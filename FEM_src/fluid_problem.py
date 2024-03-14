from __future__ import annotations

import dolfin as df

from FEM_src.problem import FEMProblem
from FEM_src.domains import SidesDomain
from FEM_src.pde_solver import SmartMumpsSolver
from designs.definitions import DomainParameters, FluidParameters, Side, Flow
from src.penalizers import FluidPenalizer


class BoundaryFlows(df.UserExpression):
    def __init__(self, domain_size: tuple[float, float], flows: list[Flow], **kwargs):
        super().__init__(**kwargs)
        self.domain_size = domain_size
        self.flows = flows

    def get_flow(self, position: float, center: float, length: float, rate: float):
        t = position - center
        if -length / 2 < t < length / 2:
            return rate * (1 - (2 * t / length) ** 2)
        return 0

    def eval(self, values, pos):
        values[1] = 0.0
        values[0] = 0.0

        for side, center, length, rate in [f.to_tuple() for f in self.flows]:
            if side == Side.LEFT:
                if pos[0] == 0.0:
                    values[0] += self.get_flow(pos[1], center, length, rate)
            elif side == Side.RIGHT:
                if pos[0] == self.domain_size[0]:
                    values[0] -= self.get_flow(pos[1], center, length, rate)
            elif side == Side.TOP:
                if pos[1] == self.domain_size[1]:
                    values[1] -= self.get_flow(pos[0], center, length, rate)
            elif side == Side.BOTTOM:
                if pos[1] == 0:
                    values[1] += self.get_flow(pos[0], center, length, rate)
            else:
                raise ValueError(f"Malformed side: {side}")

    def value_shape(self):
        return (2,)


class FluidProblem(FEMProblem):
    """Elastic compliance topology optimization problem."""

    def __init__(
        self,
        mesh: df.Mesh,
        fluid_parameters: FluidParameters,
        domain_parameters: DomainParameters,
    ):
        self.parameters = fluid_parameters
        super().__init__(mesh, domain_parameters)

        self.solver = self.create_solver()

        self.viscosity = self.parameters.viscosity
        self.penalizer: FluidPenalizer = FluidPenalizer()

        self.u = None
        self.rho = None

    def create_solver(self):
        """
        The weak form of the state equation is `r(ρ)(u, v) + μ(∇u, ∇v) +
        (∇p, v) + (∇·u, q) = 0`. This gives a = r(ρ)(u, v) + μ(∇u, ∇v) +
        (∇p, v) + (∇·u, q), L = 0
        """

        def a_func(trial, test, rho):
            (u, p) = df.split(trial)
            (v, q) = df.split(test)

            return (
                self.penalizer(rho) * df.inner(u, v)
                + df.inner(df.grad(u), df.grad(v))
                + df.inner(df.grad(p), v)
                + df.inner(df.div(u), q)
            ) * df.dx

        def l_func(test, _):
            return df.inner(test, df.Constant([0, 0, 0])) * df.dx

        return SmartMumpsSolver(
            a_func,
            l_func,
            self.boundary_conditions,
            self.solution_space,
            l_has_no_args=True,
        )

    def calculate_objective_gradient(self):
        """get objective gradient ϕ'(ρ) = ½r'(ρ)|u|²."""

        if self.rho is None or self.u is None:
            raise ValueError(
                "You must call calculate_objective before "
                + "calling calculate_objective_gradient"
            )

        return df.project(
            0.5 * self.penalizer.derivative(self.rho) * self.u**2,
            self.rho.function_space(),
        )

    def calculate_objective(self, rho):
        """get objective function ϕ(ρ) = ½∫r(ρ)|u|² + μ|∇u|² dx."""
        self.rho = rho
        (self.u, _) = df.split(self.forward(rho))

        t1 = self.penalizer(rho) * self.u**2
        t2 = self.viscosity * df.grad(self.u) ** 2
        objective = df.assemble(0.5 * (t1 + t2) * df.dx)

        return float(objective)

    def forward(self, rho):
        return self.solver.solve(a_arg=rho)

    def create_boundary_conditions(self):
        flow_sides = [flow.side for flow in self.parameters.flows]
        self.marker.add(SidesDomain(self.domain_size, flow_sides), "flow")

        # assume no slip conditions where there is no flow
        no_slip_sides = list(set(Side.get_all()).difference(flow_sides))
        self.marker.add(SidesDomain(self.domain_size, no_slip_sides), "no_slip")

        self.boundary_flows = BoundaryFlows(
            self.domain_size, self.parameters.flows, degree=2
        )

        boundary_conditions = [
            df.DirichletBC(
                self.solution_space.sub(0),
                self.boundary_flows,
                *self.marker.get("flow"),
            ),
            df.DirichletBC(
                self.solution_space.sub(0),
                df.Constant((0.0, 0.0)),
                *self.marker.get("no_slip"),
            ),
        ]

        return boundary_conditions

    def create_solution_space(self):
        velocity_space = df.VectorElement("CG", self.mesh.ufl_cell(), 2)
        pressure_space = df.FiniteElement("CG", self.mesh.ufl_cell(), 1)
        return df.FunctionSpace(self.mesh, velocity_space * pressure_space)
