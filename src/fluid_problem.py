from __future__ import annotations

import numpy as np
import dolfin as df

from src.problem import Problem
from src.penalizers import FluidPenalizer
from designs.design_parser import Side, Flow
from src.domains import SidesDomain, RegionDomain, PointDomain


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

        for side, center, length, rate in self.flows:
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


class FluidProblem(Problem):
    """Elastic compliance topology optimization problem."""

    def __init__(self):
        self.viscosity = 1.0

    def calculate_objective_gradient(self):
        """does this equation work?"""

        return df.project(
            0.5 * FluidPenalizer.derivative(self.rho) * self.u**2,
            self.rho.function_space(),
        )

    def calculate_objective(self, rho):
        """get reduced objective function ϕ(rho)"""
        self.rho = rho
        (self.u, _) = df.split(self.forward(rho))

        t1 = FluidPenalizer.eval(rho) * self.u**2
        t2 = self.viscosity * df.grad(self.u) ** 2
        objective = df.assemble(0.5 * (t1 + t2) * df.dx)

        return objective

    def forward(self, rho):
        """Solve the forward problem for a given density distribution rho(x)."""
        w = df.Function(self.solution_space)
        (u, p) = df.TrialFunctions(self.solution_space)
        (v, q) = df.TestFunctions(self.solution_space)

        F = (
            FluidPenalizer.eval(rho) * df.inner(u, v)
            + df.inner(df.grad(u), df.grad(v))
            + df.inner(df.grad(p), v)
            + df.inner(df.div(u), q)
        ) * df.dx

        df.solve(df.lhs(F) == df.rhs(F), w, bcs=self.boundary_conditions)

        return w

    def create_boundary_conditions(self):
        flows, no_slip, zero_pressure, max_region = self.data

        # flow_sides = [flow.side for flow in flows]
        # self.marker.add(SidesDomain(self.domain_size, flow_sides), "flow")

        # if zero_pressure:
        #     self.marker.add(
        #         SidesDomain(self.domain_size, zero_pressure.sides), "zero_pressure"
        #     )
        # else:
        #     # default pressure boundary condition: 0 at(0, 0)
        #     self.marker.add(PointDomain((0, 0)), "zero_pressure")

        # if no_slip:
        #     self.marker.add(SidesDomain(self.domain_size, no_slip.sides), "no_slip")
        # else:
        #     # assume no slip conditions where there is no flow
        #     all_sides = [Side.LEFT, Side.RIGHT, Side.TOP, Side.BOTTOM]
        #     no_slip_sides = list(set(all_sides).difference(flow_sides))
        #     self.marker.add(SidesDomain(self.domain_size, no_slip_sides), "no_slip")

        # if max_region:
        #     self.marker.add(RegionDomain(max_region), "max")

        # self.boundary_flows = BoundaryFlows(self.domain_size, flows, degree=2)
        # self.boundary_conditions = [
        #     df.DirichletBC(
        #         self.solution_space.sub(0),
        #         self.boundary_flows,
        #         *self.marker.get("flow"),
        #     ),
        #     df.DirichletBC(
        #         self.solution_space.sub(1),
        #         df.Constant(0.0),
        #         *self.marker.get("zero_pressure"),
        #     ),
        #     df.DirichletBC(
        #         self.solution_space.sub(0),
        #         df.Constant((0.0, 0.0)),
        #         *self.marker.get("no_slip"),
        #     ),
        # ]

        self.boundary_flows = BoundaryFlows(self.domain_size, flows)
        self.boundary_conditions = [
            df.DirichletBC(
                self.solution_space.sub(0),
                self.boundary_flows,
                "on_boundary",
            ),
        ]

    def create_function_spaces(self):
        velocity_space = df.VectorElement("CG", self.mesh.ufl_cell(), 2)
        pressure_space = df.FiniteElement("CG", self.mesh.ufl_cell(), 1)
        self.solution_space = df.FunctionSpace(
            self.mesh, velocity_space * pressure_space
        )
