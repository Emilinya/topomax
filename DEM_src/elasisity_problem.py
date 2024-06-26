from __future__ import annotations

import torch
import numpy as np
import numpy.typing as npt

from DEM_src.problem import DEMProblem
from DEM_src.utils import Mesh, unflatten
from DEM_src.filter import create_density_filter
from DEM_src.domains import SideDomain, CircleDomain
from DEM_src.dirichlet_enforcer import ElasticityEnforcer
from DEM_src.objective_calculator import ObjectiveCalculator
from DEM_src.integrator import boundary_integral, circular_integral
from DEM_src.deep_energy_method import NNParameters, DeepEnergyMethod
from designs.definitions import Force, Traction, ElasticityParameters
from src.penalizers import ElasticPenalizer


class TractionPoints:
    def __init__(self, mesh: Mesh, traction: Traction):
        side, center, length, self.value = traction.to_tuple()
        self.domain = SideDomain(mesh, side, center, length)


class BodyForce:
    def __init__(self, mesh: Mesh, force: Force, device: torch.device):
        self.value = force.value
        self.domain = CircleDomain(mesh, force.region, device)


def calculate_traction_integral(
    u: torch.Tensor,
    mesh: Mesh,
    traction_points_list: list[TractionPoints],
):
    external_energy = torch.tensor(0.0)

    for traction_points in traction_points_list:
        domain = traction_points.domain
        tx, ty = traction_points.value

        if abs(tx) > 1e-14:
            external_energy += boundary_integral(u[:, 0], mesh, domain) * tx
        if abs(ty) > 1e-14:
            external_energy += boundary_integral(u[:, 1], mesh, domain) * ty

    return external_energy


def calculate_body_force_integral(u: torch.Tensor, mesh: Mesh, body_force: BodyForce):
    external_energy = torch.tensor(0.0)

    fx, fy = body_force.value
    ux, uy = unflatten(u, mesh.shape)

    if abs(fx) > 1e-14:
        external_energy += circular_integral(ux, mesh, body_force.domain) * fx
    if abs(fy) > 1e-14:
        external_energy += circular_integral(uy, mesh, body_force.domain) * fy

    return external_energy


class StrainEnergy(ObjectiveCalculator):
    def __init__(
        self,
        mesh: Mesh,
        body_force: BodyForce | None,
        Young_modulus: float,
        Poisson_ratio: float,
        traction_points_list: list[TractionPoints] | None,
    ):
        super().__init__(mesh, ElasticPenalizer())
        self.body_force = body_force
        self.traction_points_list = traction_points_list
        self.lamé_mu = Young_modulus / (2 * (1 + Poisson_ratio))
        self.lamé_lda = self.lamé_mu * Poisson_ratio / (0.5 - Poisson_ratio)

    def calculate_strain_energy(self, _, grad: torch.Tensor):
        """
        Calculate σ:ε, where
        ε = (∇u + ∇uᵀ)/2, σ = λ∇⋅uI + 2με
        """

        # strain
        ε = 0.5 * (grad + torch.transpose(grad, 0, 1))

        # stress
        div_u = grad[0][0] + grad[1][1]
        identity = torch.eye(2).reshape((2, 2, 1, 1))
        σ = self.lamé_lda * div_u * identity + 2 * self.lamé_mu * ε

        return [torch.sum(σ * ε, [0, 1])]

    def calculate_energy(
        self, u: torch.Tensor, shape: tuple[int, int], density: torch.Tensor
    ):
        """Calculate ψ(u; ρ) = ½∫r(ρ)σ:ε dx - ∫f·u dx - ∫t·u ds"""

        (strain_energies,) = self.evaluate(u, shape, self.calculate_strain_energy)

        internal_energy = 0.5 * torch.sum(
            self.penalizer(density) * strain_energies * self.detJ
        )

        external_energy = torch.zeros_like(internal_energy)
        if self.traction_points_list is not None:
            external_energy += calculate_traction_integral(
                u, self.mesh, self.traction_points_list
            )
        if self.body_force is not None:
            external_energy += calculate_body_force_integral(
                u, self.mesh, self.body_force
            )

        return internal_energy - external_energy

    def calculate_objective_and_gradient(
        self, u: torch.Tensor, shape: tuple[int, int], density: torch.Tensor
    ):
        """Calculate ϕ(ρ) = ∫r(ρ)σ:ε dx and ∇ϕ(ρ) = -r'(ρ)σ:ε"""

        (strain_energies,) = self.evaluate(u, shape, self.calculate_strain_energy)
        strain_energy_at_element = strain_energies * self.detJ

        objective = torch.sum(self.penalizer(density) * strain_energy_at_element)
        gradient = -self.penalizer.derivative(density) * strain_energy_at_element

        return objective, gradient


class ElasticityProblem(DEMProblem):
    """Elastic compliance topology optimization problem."""

    def __init__(
        self,
        mesh: Mesh,
        device: torch.device,
        verbose: bool,
        parameters: ElasticityParameters,
    ):
        self.parameters = parameters
        super().__init__(mesh, device, verbose)

        # conversion factor from Helmholtz filter radius
        # to classical filter radius is 2sqrt(3)
        filter_radius = self.parameters.filter_radius * 2 * np.sqrt(3)
        self.filter = create_density_filter(filter_radius, self.mesh)
        self.objective_gradient: npt.NDArray[np.float64] | None = None

    def calculate_objective_gradient(self):
        if self.objective_gradient is None:
            raise ValueError(
                "You must call calculate_objective "
                + "before calling calculate_objective_gradient"
            )

        return self.objective_gradient

    def calculate_objective(self, rho: npt.NDArray[np.float64]):
        rho_shape = rho.shape
        filtered_rho = self.filter @ rho.flatten()

        objective, objective_gradient = self.dem.train_model(filtered_rho, self.mesh)

        objective = objective.cpu().detach().numpy()
        objective_gradient = objective_gradient.cpu().detach().numpy()

        # invert filter
        self.objective_gradient = (
            self.filter.T @ objective_gradient.flatten()
        ).reshape(rho_shape)

        return float(objective)

    def forward(self, rho: npt.NDArray[np.float64]): ...

    def create_dem_parameters(self):
        body_force = None
        if self.parameters.body_force:
            body_force = BodyForce(self.mesh, self.parameters.body_force, self.device)

        traction_points_list: list[TractionPoints] | None = None
        if self.parameters.tractions:
            traction_points_list = []
            for traction in self.parameters.tractions:
                traction_points_list.append(TractionPoints(self.mesh, traction))

        dirichlet_enforcer = ElasticityEnforcer(self.parameters, self.mesh, self.device)

        strain_energy = StrainEnergy(
            self.mesh,
            body_force,
            self.parameters.young_modulus,
            self.parameters.poisson_ratio,
            traction_points_list,
        )

        return dirichlet_enforcer, strain_energy

    def create_dem(
        self,
        dirichlet_enforcer: ElasticityEnforcer,
        objective_calculator: StrainEnergy,
    ):
        nn_parameters = NNParameters(
            layer_count=5,
            neuron_count=68,
            learning_rate=1.73553,
            iteration_count=100,
            weight_deviation=0.062264,
            fourier_deviation=0.119297,
            activation_function="rrelu",
            convergence_tolerance=5e-5,
        )

        dimension = 2
        return DeepEnergyMethod(
            self.device,
            self.verbose,
            dimension,
            nn_parameters,
            dirichlet_enforcer,
            objective_calculator,
        )
