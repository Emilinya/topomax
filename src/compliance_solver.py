import os

import numpy as np
from scipy import io
import dolfin as df
import dolfin_adjoint as dfa
from pyadjoint.reduced_functional_numpy import ReducedFunctionalNumPy

from designs.design_parser import parse_design
from src.utils import SidesDomain, MeshFunctionWrapper, constrain

df.set_log_level(df.LogLevel.ERROR)
# turn off redundant output in parallel
df.parameters["std_out_all_processes"] = False


class BodyForce(dfa.UserExpression):
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        self.rho = kwargs["rho"]
        self.domain_size: tuple[float, float] = kwargs["domain_size"]

    def set_rho(self, rho):
        self.rho = rho

    def eval(self, values, pos):
        values[0] = 0.0
        values[1] = -1 * self.rho(pos)

        if df.near(pos[0], 2.9) and df.near(pos[1], 0.5):
            values[1] += -1

    def value_shape(self):
        return (2,)


class ControlFilter:
    def __init__(self, control_space):
        self.control_space = control_space
        self.epsilon = 0.02

    def apply(self, input_rho):
        # solve -ε²Δξ + ξ = ρ, ∇ξ·n = 0 on ∂Ω
        # where ρ is the input and ξ is the output

        trial_rho = df.TrialFunction(self.control_space)
        test_rho = df.TestFunction(self.control_space)

        lhs = (
            -self.epsilon**2 * df.inner(df.grad(trial_rho), df.grad(test_rho))
            + df.inner(trial_rho, test_rho)
        ) * df.dx
        rhs = df.inner(input_rho, test_rho) * df.dx

        filtered_rho = dfa.Function(self.control_space)
        dfa.solve(lhs == rhs, filtered_rho)

        return filtered_rho


def expit(x):
    return 1.0 / (1.0 + np.exp(-x))


def expit_diff(x):
    expit_val = expit(x)
    return expit_val * (1 - expit_val)


def logit(x):
    return np.log(x / (1.0 - x))


class ComplianceSolver:
    def __init__(self, design_file: str, N: int):
        self.design_file = design_file
        self.parameters, *_ = parse_design(self.design_file)

        # define constants
        self.Young_modulus = 4 / 3
        self.Poisson_ratio = 1 / 3

        # define domain
        self.N = N
        self.width = self.parameters.width
        self.height = self.parameters.height
        domain_size = (self.width, self.height)

        volume_fraction = self.parameters.fraction
        self.volume = self.width * self.height * volume_fraction

        self.mesh = dfa.Mesh(
            dfa.RectangleMesh(
                df.MPI.comm_world,
                df.Point(0.0, 0.0),
                df.Point(self.width, self.height),
                int(self.width * self.N),
                int(self.height * self.N),
            )
        )

        # define function spaces space
        self.latent_space = df.FunctionSpace(self.mesh, "DG", 0)
        self.control_space = df.FunctionSpace(self.mesh, "DG", 0)
        displacement_element = df.VectorElement("CG", self.mesh.ufl_cell(), 2)
        self.solution_space = df.FunctionSpace(self.mesh, displacement_element)

        self.control_filter = ControlFilter(self.control_space)

        # create initial conditions
        self.rho = dfa.Function(self.control_space)
        self.rho.vector()[:] = volume_fraction / (self.width * self.height)
        self.body_force = BodyForce(domain_size=domain_size, rho=self.rho)

        # define boundary conditions
        marker = MeshFunctionWrapper(self.mesh)
        marker.add(SidesDomain(domain_size, ["left"]), "fixed")

        self.boundary_conditions = [
            dfa.DirichletBC(
                self.solution_space,
                dfa.Constant((0.0, 0.0)),
                *marker.get("fixed"),
            ),
        ]

        # get reduced objective function: rho --> j(rho)
        dfa.set_working_tape(dfa.Tape())
        filtered_rho = self.control_filter.apply(self.rho)
        u = self.forward(filtered_rho)
        objective = dfa.assemble(df.inner(u, self.body_force) * df.dx)

        self.control = dfa.Control(self.rho)
        self.objective_function = ReducedFunctionalNumPy(objective, self.control)

    def alpha(self, rho):
        alpha_min = 1e-6
        return alpha_min + rho**3 * (1 - alpha_min)

    def forward(self, rho):
        """Solve the forward problem for a given density distribution rho(x)."""
        self.body_force.set_rho(rho)

        w = dfa.Function(self.solution_space)
        u = df.TrialFunction(self.solution_space)
        v = df.TestFunction(self.solution_space)
        d = u.geometric_dimension()

        traction = dfa.Constant((0, 0))
        lda = self.Young_modulus * (1 + self.Poisson_ratio)
        mu = lda * self.Poisson_ratio / (1 - 2 * self.Poisson_ratio)

        sigma = lda * df.div(u) * df.Identity(d) + 2 * mu * df.sym(df.grad(u))

        a = df.inner(self.alpha(rho) * sigma, df.sym(df.grad(v))) * df.dx
        L = df.dot(self.body_force, v) * df.dx + df.dot(traction, v) * df.ds

        dfa.solve(a == L, w, bcs=self.boundary_conditions)

        return w

    def zero_solver(self, half_step, volume):
        # solve 'expit(half_step + c) * df.dx = volume' with newtons method

        expit_integral_func = dfa.Function(self.control_space)
        expit_diff_integral_func = dfa.Function(self.control_space)

        def evaluate(c):
            expit_integral_func.vector()[:] = expit(half_step + c)
            expit_diff_integral_func.vector()[:] = expit_diff(half_step + c)

            error = dfa.assemble(expit_integral_func * df.dx) - volume
            gradient = dfa.assemble(expit_diff_integral_func * df.dx)

            return error, gradient

        # is 0 a good initial guess?
        c = 0
        err, grad = evaluate(c)
        while abs(err) > 1e-6:
            c = c - err / grad
            err, grad = evaluate(c)

        return c - err / grad

    def step(self, prev_psi, alpha):
        # Latent space gradient descent
        objective_gradient = self.objective_function.derivative()
        half_step = prev_psi - alpha * objective_gradient

        # Compute Lagrange multiplier
        c = self.zero_solver(half_step, self.volume)

        # Latent space feasibility forrection
        return half_step + c

    def solve(self):
        itol = 1e-2
        ntol = 1e-5

        psi = logit(self.rho.vector()[:])
        prev_psi = None

        error = float("Infinity")
        objective = float(self.objective_function(self.rho.vector()[:]))
        print("Iteration │ Objective │   Error  ")
        print("──────────┼───────────┼──────────")

        k = 0
        while error > min(ntol, itol):
            print(f"{k:^9} │ {constrain(objective, 9)} │ {constrain(error, 9)}")

            prev_psi = psi
            psi = self.step(prev_psi, 1)

            self.rho.vector()[:] = expit(psi)
            objective = float(self.objective_function(self.rho.vector()[:]))

            self.save_control(self.rho, objective, k)
            k += 1

            # create dfa functions from psi and prev_psi to calculate error
            prev_psi_func = dfa.Function(self.control_space)
            prev_psi_func.vector()[:] = prev_psi

            psi_func = dfa.Function(self.control_space)
            psi_func.vector()[:] = psi

            error = np.sqrt(dfa.assemble((psi_func - prev_psi_func) ** 2 * df.dx))

    def save_control(self, rho, objective, k):
        design = os.path.splitext(os.path.basename(self.design_file))[0]
        filename = f"output/{design}/data/N={self.N}_{k=}.mat"

        Nx, Ny = int(self.width * self.N), int(self.height * self.N)
        data = np.array(
            [
                [rho((0.5 + xi) / Nx, (0.5 + yi) / Ny) for xi in range(Nx)]
                for yi in range(Ny)
            ]
        )

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        io.savemat(
            filename,
            mdict={
                "data": data,
                "objective": objective,
            },
        )
