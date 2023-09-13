import os

import numpy as np
from scipy import io
import dolfin as df
import dolfin_adjoint as dfa

from designs.design_parser import parse_design
from src.compliance_problem import Problem
from src.filter import HelmholtzFilter
from src.utils import constrain

df.set_log_level(df.LogLevel.ERROR)
# turn off redundant output in parallel
df.parameters["std_out_all_processes"] = False


def expit(x):
    """Sigmoid function."""
    return 1.0 / (1.0 + np.exp(-x))


def expit_diff(x):
    """Derivative of the sigmoid function."""
    expit_val = expit(x)
    return expit_val * (1 - expit_val)


def logit(x):
    """Inverse sigmoid function."""
    return np.log(x / (1.0 - x))


class Solver:
    """Class that solves a given topology optimization problem using a magical algorithm."""

    def __init__(self, design_file: str, N: int, problem: Problem):
        self.problem = problem
        self.design_file = design_file
        self.parameters, *_ = parse_design(self.design_file)

        # define domain
        self.N = N
        self.width = self.parameters.width
        self.height = self.parameters.height

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

        control_filter = HelmholtzFilter(self.N)
        self.rho, self.objective_function = self.problem.init(
            control_filter, self.mesh, self.parameters
        )

    def zero_solver(self, half_step, volume: float):
        """solve '∫expit(half_step + c)dx = volume' for c using Newton's method."""

        expit_integral_func = dfa.Function(self.problem.control_space)
        expit_diff_integral_func = dfa.Function(self.problem.control_space)

        def evaluate(c: float):
            expit_integral_func.vector()[:] = expit(half_step + c)
            expit_diff_integral_func.vector()[:] = expit_diff(half_step + c)

            error = float(dfa.assemble(expit_integral_func * df.dx) - volume)
            gradient = float(dfa.assemble(expit_diff_integral_func * df.dx))

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
        """Solve the given topology optimization problem."""
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

            self.save_rho(self.rho, objective, k)
            k += 1

            # create dfa functions from psi and prev_psi to calculate error
            prev_psi_func = dfa.Function(self.problem.control_space)
            prev_psi_func.vector()[:] = prev_psi

            psi_func = dfa.Function(self.problem.control_space)
            psi_func.vector()[:] = psi

            error = np.sqrt(dfa.assemble((psi_func - prev_psi_func) ** 2 * df.dx))

    def save_rho(self, rho, objective, k):
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
