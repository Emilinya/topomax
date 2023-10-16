from __future__ import annotations

import numpy as np
import dolfin as df

from src.problem import Problem
from src.domains import SidesDomain
from src.penalizers import ElasticPenalizer
from designs.design_parser import Side, ForceRegion, Traction


class BodyForce(df.UserExpression):
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        self.rho = kwargs["rho"]
        self.force_region: ForceRegion = kwargs["force_region"]
        self.domain_size: tuple[float, float] = kwargs["domain_size"]

    def set_rho(self, rho):
        self.rho = rho

    def eval(self, values, pos):
        values[0] = 0.0
        values[1] = 0.0

        center = self.force_region.center
        distance = np.sqrt((pos[0] - center[0]) ** 2 + (pos[1] - center[1]) ** 2)
        if distance < self.force_region.radius:
            values[0], values[1] = self.force_region.value

    def value_shape(self):
        return (2,)


class TractionExpression(df.UserExpression):
    def __init__(self, **kwargs):
        super().__init__(kwargs)
        self.domain_size: tuple[float, float] = kwargs["domain_size"]
        self.tractions: list[Traction] = kwargs["tractions"]

    def eval(self, values, pos):
        values[0] = 0.0
        values[1] = 0.0

        for side, center, length, value in self.tractions:
            region = (center - length / 2, center + length / 2)
            if side == Side.LEFT:
                if pos[0] == 0.0 and df.between(pos[1], region):
                    values[0] += value[0]
                    values[1] += value[1]
            elif side == Side.RIGHT:
                if pos[0] == self.domain_size[0] and df.between(pos[1], region):
                    values[0] += value[0]
                    values[1] += value[1]
            elif side == Side.TOP:
                if pos[1] == self.domain_size[1] and df.between(pos[0], region):
                    values[0] += value[0]
                    values[1] += value[1]
            elif side == Side.BOTTOM:
                if pos[1] == 0 and df.between(pos[0], region):
                    values[0] += value[0]
                    values[1] += value[1]
            else:
                raise ValueError(f"Malformed side: {side}")

    def value_shape(self):
        return (2,)


class ElasticityProblem(Problem):
    """Elastic compliance topology optimization problem."""

    def __init__(self):
        super().__init__()

        self.Young_modulus = 4 / 3
        self.Poisson_ratio = 1 / 3

        # Calculate Lamé parameters from material properties
        self.lamé_lda = self.Young_modulus / (1 + self.Poisson_ratio)
        self.lamé_mu = self.lamé_lda * self.Poisson_ratio / (1 - 2 * self.Poisson_ratio)

        self.u = None
        self.body_force = None
        self.filtered_rho = None
        self.traction_term = None

    def calculate_objective_gradient(self):
        """
        Filter -α'(ξ) (λ|∇⋅u|² + 2μ|ε(u)|²), where ξ is the filtered rho
        and ε(u) = (∇u + ∇uᵀ)/2 is the symmetric gradient of u.
        """

        if self.filtered_rho is None or self.u is None:
            raise ValueError(
                "You must call calculate_objective "
                + "before calling calculate_objective_gradient"
            )

        gradient = -ElasticPenalizer.derivative(self.filtered_rho) * (
            self.lamé_lda * df.div(self.u) ** 2
            + 2 * self.lamé_mu * df.sym(df.grad(self.u)) ** 2
        )
        return self.filter.apply(gradient, self.filtered_rho.function_space())

    def calculate_objective(self, rho):
        """
        get reduced objective function ϕ(ρ) = ∫u⋅f dx + ∫u⋅t ds,
        where f is the body force and t is the traction term.
        """

        if not isinstance(self.body_force, df.Constant):
            self.body_force.set_rho(rho)
        self.filtered_rho = self.filter.apply(rho)
        self.u = self.forward(self.filtered_rho)
        objective = df.assemble(
            df.inner(self.u, self.body_force) * df.dx
            + df.inner(self.u, self.traction_term) * df.ds
        )

        return float(objective)

    def forward(self, rho):
        """
        Solve the state equation (λα(ξ)∇⋅u, ∇⋅v) + (2μα(ξ)ε(u), ε(v)) = (f, v),
        where ξ is the filtered rho and ε(u) is the symmetric gradient of u
        """

        w = df.Function(self.solution_space)
        u = df.TrialFunction(self.solution_space)
        v = df.TestFunction(self.solution_space)
        d = u.geometric_dimension()

        sigma = self.lamé_lda * df.div(u) * df.Identity(d) + 2 * self.lamé_mu * df.sym(
            df.grad(u)
        )

        a = df.inner(ElasticPenalizer.eval(rho) * sigma, df.sym(df.grad(v))) * df.dx
        L = df.dot(self.body_force, v) * df.dx + df.dot(self.traction_term, v) * df.ds

        df.solve(a == L, w, bcs=self.boundary_conditions)

        return w

    def create_boundary_conditions(self):
        force_region, fixed_sides, tractions = self.data

        self.body_force = df.Constant((0, 0))
        if force_region is not None:
            self.body_force = BodyForce(
                domain_size=self.domain_size, force_region=force_region, rho=None
            )

        self.traction_term = TractionExpression(
            domain_size=self.domain_size, tractions=tractions
        )

        self.marker.add(SidesDomain(self.domain_size, fixed_sides), "fixed")
        return [
            df.DirichletBC(
                self.solution_space,
                df.Constant((0.0, 0.0)),
                *self.marker.get("fixed"),
            )
        ]

    def create_solution_space(self):
        displacement_element = df.VectorElement("CG", self.mesh.ufl_cell(), 2)
        return df.FunctionSpace(self.mesh, displacement_element)
