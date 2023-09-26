from __future__ import annotations

import numpy as np
import dolfin as df

from src.problem import Problem
from src.utils import fluid_alpha
from designs.design_parser import Side, Flow
from src.domains import SidesDomain, RegionDomain, PointDomain


class FlowBoundaryConditions(df.UserExpression):
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        self.domain_size: tuple[float, float] = kwargs["domain_size"]
        self.flows: list[Flow] = kwargs["flows"]

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
        """
        self.viscosity * lda * phi * df.dx + fluid_alpha(rho) * lda * phi * df.dx = -F'_rho(u, rho)
        """
        return self.u

    def calculate_objective(self, rho):
        """get reduced objective function ϕ(rho)"""
        (self.u, self.p) = df.split(self.forward(rho))

        if self.objective == "minimize_power":
            t1 = 0.5 * df.inner(fluid_alpha(rho) * self.u, self.u) * df.dx
            t2 = self.viscosity * df.inner(df.grad(self.u), df.grad(self.u)) * df.dx
            objective = df.assemble(t1 + t2)
        elif self.objective == "maximize_flow":
            subdomain_data, subdomain_idx = self.marker.get("max")
            ds = df.Measure("dS", domain=self.mesh, subdomain_data=subdomain_data)
            objective = df.assemble(
                -df.inner(self.u, df.Constant((-1.0, 0))) * ds(subdomain_idx)
            )

        return objective

    def forward(self, rho):
        """Solve the forward problem for a given density distribution rho(x)."""
        w = df.Function(self.solution_space)
        (u, p) = df.TrialFunctions(self.solution_space)
        (v, q) = df.TestFunctions(self.solution_space)

        F = (
            fluid_alpha(rho) * df.inner(u, v)
            + df.inner(df.grad(u), df.grad(v))
            + df.inner(df.grad(p), v)
            + df.inner(df.div(u), q)
        ) * df.dx

        df.solve(df.lhs(F) == df.rhs(F), w, bcs=self.boundary_conditions)

        return w

    def create_boundary_conditions(self):
        flows, no_slip, zero_pressure, max_region = self.data

        flow_sides = [flow.side for flow in flows]
        self.marker.add(SidesDomain(self.domain_size, flow_sides), "flow")

        if zero_pressure:
            self.marker.add(
                SidesDomain(self.domain_size, zero_pressure.sides), "zero_pressure"
            )
        else:
            # default pressure boundary condition: 0 at(0, 0)
            self.marker.add(PointDomain((0, 0)), "zero_pressure")

        if no_slip:
            self.marker.add(SidesDomain(self.domain_size, no_slip.sides), "no_slip")
        else:
            # assume no slip conditions where there is no flow
            all_sides = [Side.LEFT, Side.RIGHT, Side.TOP, Side.BOTTOM]
            no_slip_sides = list(set(all_sides).difference(flow_sides))
            self.marker.add(SidesDomain(self.domain_size, no_slip_sides), "no_slip")

        if max_region:
            self.marker.add(RegionDomain(max_region), "max")

        self.boundary_conditions = [
            df.DirichletBC(
                self.solution_space.sub(0),
                FlowBoundaryConditions(
                    degree=2, domain_size=self.domain_size, flows=flows
                ),
                *self.marker.get("flow"),
            ),
            df.DirichletBC(
                self.solution_space.sub(1),
                df.Constant(0.0),
                *self.marker.get("zero_pressure"),
            ),
            df.DirichletBC(
                self.solution_space.sub(0),
                df.Constant((0.0, 0.0)),
                *self.marker.get("no_slip"),
            ),
        ]

    def create_function_spaces(self):
        velocity_space = df.VectorElement("CG", self.mesh.ufl_cell(), 2)
        pressure_space = df.FiniteElement("CG", self.mesh.ufl_cell(), 1)
        self.solution_space = df.FunctionSpace(
            self.mesh, velocity_space * pressure_space
        )
