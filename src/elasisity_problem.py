from __future__ import annotations

import numpy as np
import dolfin as df

from src.problem import Problem
from src.domains import SidesDomain
from designs.design_parser import ForceRegion
from src.utils import elastisity_alpha, elastisity_alpha_derivative


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


class ElasticityProblem(Problem):
    """Elastic compliance topology optimization problem."""

    def __init__(self):
        self.Young_modulus = 4 / 3
        self.Poisson_ratio = 1 / 3

        self.filtered_rho = None
        self.u = None

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

        lda = self.Young_modulus * (1 + self.Poisson_ratio)
        mu = lda * self.Poisson_ratio / (1 - 2 * self.Poisson_ratio)

        gradient = -elastisity_alpha_derivative(self.filtered_rho) * (
            lda * df.div(self.u) ** 2 + 2 * mu * df.sym(df.grad(self.u)) ** 2
        )
        return self.filter.apply(gradient, self.filtered_rho.function_space())

    def calculate_objective(self, rho):
        """get reduced objective function ϕ(rho)"""
        self.body_force.set_rho(rho)
        self.filtered_rho = self.filter.apply(rho)
        self.u = self.forward(self.filtered_rho)
        objective = float(df.assemble(df.inner(self.u, self.body_force) * df.dx))

        return objective

    def forward(self, filtered_rho):
        """
        Solve the state equation (λα(ξ)∇⋅u, ∇⋅v) + (2μα(ξ)ε(u), ε(v)) = (f, v),
        where ξ is the filtered rho and ε(u) is the symmetric gradient of u
        """

        w = df.Function(self.solution_space)
        u = df.TrialFunction(self.solution_space)
        v = df.TestFunction(self.solution_space)
        d = u.geometric_dimension()

        lda = self.Young_modulus * (1 + self.Poisson_ratio)
        mu = lda * self.Poisson_ratio / (1 - 2 * self.Poisson_ratio)

        sigma = lda * df.div(u) * df.Identity(d) + 2 * mu * df.sym(df.grad(u))

        a = df.inner(elastisity_alpha(filtered_rho) * sigma, df.sym(df.grad(v))) * df.dx
        L = df.dot(self.body_force, v) * df.dx + df.dot(self.traction, v) * df.ds

        df.solve(a == L, w, bcs=self.boundary_conditions)

        return w

    def create_boundary_conditions(self):
        force_region, fixed_sides, traction = self.data

        self.traction = df.Constant(traction)
        self.body_force = BodyForce(
            domain_size=self.domain_size, force_region=force_region, rho=None
        )

        self.marker.add(SidesDomain(self.domain_size, fixed_sides), "fixed")

        self.boundary_conditions = [
            df.DirichletBC(
                self.solution_space,
                df.Constant((0.0, 0.0)),
                *self.marker.get("fixed"),
            ),
        ]

    def create_function_spaces(self):
        displacement_element = df.VectorElement("CG", self.mesh.ufl_cell(), 2)
        self.solution_space = df.FunctionSpace(self.mesh, displacement_element)
