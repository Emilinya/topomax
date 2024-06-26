from __future__ import annotations

import numpy as np
import dolfin as df

from FEM_src.problem import FEMProblem
from FEM_src.domains import SidesDomain
from FEM_src.filter import HelmholtzFilter
from FEM_src.pde_solver import SmartMumpsSolver
from src.penalizers import ElasticPenalizer
from designs.definitions import (
    DomainParameters,
    ElasticityParameters,
    Side,
    Traction,
    Force,
)


class BodyForce(df.UserExpression):
    def __init__(self, force: Force, domain_size: tuple[float, float], **kwargs):
        super().__init__(**kwargs)
        self.force = force
        self.domain_size = domain_size

    def eval(self, values, pos):
        values[0] = 0.0
        values[1] = 0.0

        center = self.force.region.center
        distance = np.sqrt((pos[0] - center[0]) ** 2 + (pos[1] - center[1]) ** 2)
        if distance < self.force.region.radius:
            values[0], values[1] = self.force.value

    def value_shape(self):
        return (2,)


class TractionExpression(df.UserExpression):
    def __init__(
        self, domain_size: tuple[float, float], tractions: list[Traction], **kwargs
    ):
        super().__init__(**kwargs)
        self.domain_size = domain_size
        self.tractions = tractions

    def eval(self, values, pos):
        values[0] = 0.0
        values[1] = 0.0

        for side, center, length, value in [t.to_tuple() for t in self.tractions]:
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


class ElasticityProblem(FEMProblem):
    """Elastic compliance topology optimization problem."""

    def __init__(
        self,
        mesh: df.Mesh,
        control_space: df.FunctionSpace,
        domain_parameters: DomainParameters,
        elasticity_parameters: ElasticityParameters,
    ):
        self.parameters = elasticity_parameters
        super().__init__(mesh, domain_parameters)

        filter_radius = self.parameters.filter_radius
        self.filter = HelmholtzFilter(filter_radius, control_space)
        self.solver = self.create_solver()

        self.Young_modulus = self.parameters.young_modulus
        self.Poisson_ratio = self.parameters.poisson_ratio
        self.penalizer: ElasticPenalizer = ElasticPenalizer()

        # Calculate Lamé parameters from material properties
        self.lamé_mu = self.Young_modulus / (2 * (1 + self.Poisson_ratio))
        self.lamé_lda = self.lamé_mu * self.Poisson_ratio / (0.5 - self.Poisson_ratio)

        self.u = None
        self.filtered_rho = None

    def create_solver(self):
        """
        The weak form of the state equation is `(λr(ξ)∇⋅u, ∇⋅v) + (2μr(ξ)ε(u), ε(v))
        = (f, v) + (t, v)∂`, where ξ is the filtered rho and ε(u) is the symmetric
        gradient of u. This gives a = (λr(ξ)∇⋅u, ∇⋅v) + (2μr(ξ)ε(u), ε(v)),
        L = (f, v) + (t, v)∂
        """

        def a_func(trial, test, rho):
            d = trial.geometric_dimension()
            sigma = self.lamé_lda * df.div(trial) * df.Identity(
                d
            ) + 2 * self.lamé_mu * df.sym(df.grad(trial))

            return df.inner(self.penalizer(rho) * sigma, df.sym(df.grad(test))) * df.dx

        def l_func(test, _):
            return (
                df.dot(self.body_force, test) * df.dx
                + df.dot(self.traction_term, test) * df.ds
            )

        return SmartMumpsSolver(
            a_func,
            l_func,
            self.boundary_conditions,
            self.solution_space,
            l_has_no_args=True,
        )

    def calculate_objective_gradient(self):
        """
        Filter -r'(ξ) (λ|∇⋅u|² + 2μ|ε(u)|²), where ξ is the filtered rho
        and ε(u) = (∇u + ∇uᵀ)/2 is the symmetric gradient of u.
        """

        if self.filtered_rho is None or self.u is None:
            raise ValueError(
                "You must call calculate_objective "
                + "before calling calculate_objective_gradient"
            )

        gradient = -self.penalizer.derivative(self.filtered_rho) * (
            self.lamé_lda * df.div(self.u) ** 2
            + 2 * self.lamé_mu * df.sym(df.grad(self.u)) ** 2
        )
        return self.filter.apply(gradient)

    def calculate_objective(self, rho):
        """
        Get objective function ϕ(ρ) = ∫u⋅f dx + ∫u⋅t ds,
        where u is the solution to the state equation with
        the given ρ, f is the body force and t is the traction.
        """

        self.filtered_rho = self.filter.apply(rho)
        self.u = self.forward(self.filtered_rho)
        objective = df.assemble(
            df.inner(self.u, self.body_force) * df.dx
            + df.inner(self.u, self.traction_term) * df.ds
        )

        return float(objective)

    def forward(self, rho):
        return self.solver.solve(a_arg=rho)

    def create_boundary_conditions(self):
        self.body_force = df.Constant((0, 0))
        if self.parameters.body_force is not None:
            self.body_force = BodyForce(self.parameters.body_force, self.domain_size)

        if self.parameters.tractions is not None:
            self.traction_term = TractionExpression(
                self.domain_size, self.parameters.tractions
            )
        else:
            self.traction_term = TractionExpression(self.domain_size, [])

        self.marker.add(
            SidesDomain(self.domain_size, self.parameters.fixed_sides), "fixed"
        )
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
