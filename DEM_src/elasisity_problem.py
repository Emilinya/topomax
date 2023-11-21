from __future__ import annotations

import torch
import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix

from designs.definitions import ElasticityDesign
from src.penalizers import ElasticPenalizer
from DEM_src.data_structs import Domain
from DEM_src.ObjectiveCalculator import ObjectiveCalculator
from DEM_src.external_energy import calculate_external_energy
from DEM_src.bc_helpers import ElasticityEnforcer, TractionPoints
from DEM_src.DeepEnergyMethod import NNParameters, DeepEnergyMethod


class StrainEnergy(ObjectiveCalculator):
    def __init__(
        self,
        dxdy: tuple[float, float],
        Young_modulus: float,
        Poisson_ratio: float,
        traction_points_list: list[TractionPoints],
    ):
        super().__init__(dxdy)
        self.lamé_mu = Young_modulus / (2 * (1 + Poisson_ratio))
        self.lamé_lda = self.lamé_mu * Poisson_ratio / (0.5 - Poisson_ratio)
        self.traction_points_list = traction_points_list

        self.penalizer = ElasticPenalizer()
        self.penalizer.set_penalization(3)

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

        external_energy = calculate_external_energy(
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


class ElasticityProblem:
    """Elastic compliance topology optimization problem."""

    def __init__(
        self,
        domain: Domain,
        device: torch.device,
        verbose: bool,
        input_filter: csr_matrix,
        elasticity_design: ElasticityDesign,
    ):
        self.design = elasticity_design
        self.filter = input_filter
        self.domain = domain

        traction_points_list: list[TractionPoints] = []
        if self.design.parameters.tractions:
            for traction in self.design.parameters.tractions:
                traction_points_list.append(TractionPoints(self.domain, traction))

        dirichlet_enforcer = ElasticityEnforcer(
            self.design.parameters, self.domain, device
        )

        strain_energy = StrainEnergy(
            self.domain.dxdy,
            self.design.parameters.young_modulus,
            self.design.parameters.poisson_ratio,
            traction_points_list,
        )

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

        # Note: Making this code work with 3D requires a
        # lot more work than just changing the value below
        dimension = 2
        self.dem = DeepEnergyMethod(
            device,
            verbose,
            dimension,
            nn_parameters,
            dirichlet_enforcer,
            strain_energy,
        )

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
