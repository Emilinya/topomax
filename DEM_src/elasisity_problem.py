from __future__ import annotations

import torch
import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix

from DEM_src.utils import Mesh
from DEM_src.problem import DEMProblem
from DEM_src.integrator import boundary_integral
from DEM_src.domains import SideDomain, CircleDomain
from DEM_src.dirichlet_enforcer import ElasticityEnforcer
from DEM_src.ObjectiveCalculator import ObjectiveCalculator
from DEM_src.DeepEnergyMethod import NNParameters, DeepEnergyMethod
from designs.definitions import Traction, ElasticityDesign
from src.penalizers import ElasticPenalizer


class TractionPoints:
    def __init__(self, mesh: Mesh, traction: Traction):
        side, center, length, self.value = traction.to_tuple()
        self.domain = SideDomain(mesh, side, center, length)


def calculate_traction_integral(
    u: torch.Tensor,
    dxdy: tuple[float, float],
    traction_points_list: list[TractionPoints],
):
    external_energy = torch.tensor(0.0)

    for tps in traction_points_list:
        tx, ty = tps.value

        if abs(tx) > 1e-14:
            external_energy += boundary_integral(u[:, 0], dxdy, tps.domain) * tx
        if abs(ty) > 1e-14:
            external_energy += boundary_integral(u[:, 1], dxdy, tps.domain) * ty

    return external_energy


class StrainEnergy(ObjectiveCalculator):
    def __init__(
        self,
        dxdy: tuple[float, float],
        Young_modulus: float,
        Poisson_ratio: float,
        traction_points_list: list[TractionPoints],
    ):
        super().__init__(dxdy, ElasticPenalizer())
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

        return [torch.einsum("ijkl,ijkl->kl", σ, ε)]

    def calculate_potential_power(
        self, u: torch.Tensor, shape: tuple[int, int], density: torch.Tensor
    ):
        """Calculate ψ(u; ρ) = ½∫r(ρ)σ:ε dx - ∫t·u ds"""

        (strain_energies,) = self.evaluate(u, shape, self.calculate_strain_energy)

        internal_energy = 0.5 * torch.sum(
            self.penalizer(density) * strain_energies * self.detJ
        )

        external_energy = calculate_traction_integral(
            u, self.dxdy, self.traction_points_list
        )

        return internal_energy - external_energy

    def calculate_objective_and_gradient(
        self, u: torch.Tensor, shape: tuple[int, int], density: torch.Tensor
    ):
        """Calculate ∇ϕ(ρ; u) = -½r'(ρ)σ:ε and ϕ(ρ; u) = ½∫r(ρ)σ:ε dx"""

        (strain_energies,) = self.evaluate(u, shape, self.calculate_strain_energy)
        strain_energy_at_element = 0.5 * strain_energies * self.detJ

        objective = torch.sum(self.penalizer(density) * strain_energy_at_element)
        gradient = -self.penalizer.derivative(density) * strain_energy_at_element

        return objective, gradient


class ElasticityProblem(DEMProblem):
    """Elastic compliance topology optimization problem."""

    def __init__(
        self,
        domain: Mesh,
        device: torch.device,
        verbose: bool,
        input_filter: csr_matrix,
        elasticity_design: ElasticityDesign,
    ):
        self.filter = input_filter
        self.design = elasticity_design
        super().__init__(domain, device, verbose)

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

        objective, objective_gradient = self.dem.train_model(filtered_rho, self.domain)

        objective = objective.cpu().detach().numpy()
        objective_gradient = objective_gradient.cpu().detach().numpy()

        # invert filter
        self.objective_gradient = (
            self.filter.T @ objective_gradient.flatten()
        ).reshape(rho_shape)

        return float(objective)

    def forward(self, rho: npt.NDArray[np.float64]):
        ...

    def create_dem_parameters(self):
        if self.design.parameters.body_force:
            circle_domain = CircleDomain(
                self.domain, self.design.parameters.body_force.region
            )
            print(circle_domain.area_error * 100)
            exit()

        traction_points_list: list[TractionPoints] = []
        if self.design.parameters.tractions:
            for traction in self.design.parameters.tractions:
                traction_points_list.append(TractionPoints(self.domain, traction))

        dirichlet_enforcer = ElasticityEnforcer(
            self.design.parameters, self.domain, self.device
        )

        strain_energy = StrainEnergy(
            self.domain.dxdy,
            self.design.parameters.young_modulus,
            self.design.parameters.poisson_ratio,
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
            CNN_deviation=0.062264,
            rff_deviation=0.119297,
            iteration_count=100,
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
