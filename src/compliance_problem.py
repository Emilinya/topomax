import dolfin as df
import dolfin_adjoint as dfa
from pyadjoint.reduced_functional_numpy import ReducedFunctionalNumPy

from src.utils import alpha
from src.filter import Filter
from src.problem import Problem
from src.domains import SidesDomain
from designs.design_parser import SolverParameters


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
            values[1] += 0

    def value_shape(self):
        return (2,)


class ComplianceProblem(Problem):
    def __init__(self):
        self.Young_modulus = 4 / 3
        self.Poisson_ratio = 1 / 3

    def get_rho(self):
        return self.rho

    def create_objective(self):
        """get reduced objective function ϕ(rho)"""
        dfa.set_working_tape(dfa.Tape())
        filtered_rho = self.control_filter.apply(self.rho)
        u = self.forward(filtered_rho)
        objective = dfa.assemble(df.inner(u, self.body_force) * df.dx)

        control = dfa.Control(self.rho)
        return ReducedFunctionalNumPy(objective, control)

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

        a = df.inner(alpha(rho) * sigma, df.sym(df.grad(v))) * df.dx
        L = df.dot(self.body_force, v) * df.dx + df.dot(traction, v) * df.ds

        dfa.solve(a == L, w, bcs=self.boundary_conditions)

        return w

    def create_boundary_conditions(self):
        self.marker.add(SidesDomain(self.domain_size, ["left"]), "fixed")

        self.boundary_conditions = [
            dfa.DirichletBC(
                self.solution_space,
                dfa.Constant((0.0, 0.0)),
                *self.marker.get("fixed"),
            ),
        ]

    def create_function_spaces(self, mesh: dfa.Mesh):
        self.control_space = df.FunctionSpace(mesh, "DG", 0)
        displacement_element = df.VectorElement("CG", mesh.ufl_cell(), 2)
        self.solution_space = df.FunctionSpace(mesh, displacement_element)

    def create_rho(self):
        self.rho = dfa.Function(self.control_space)
        self.rho.vector()[:] = self.volume_fraction / (
            self.domain_size[0] * self.domain_size[1]
        )
        self.body_force = BodyForce(domain_size=self.domain_size, rho=self.rho)
