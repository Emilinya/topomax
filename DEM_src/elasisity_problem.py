import torch
import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix

from src.penalizers import ElasticPenalizer
from designs.definitions import ElasticityDesign
from DEM_src.data_structs import Domain, NNParameters
from DEM_src.DeepEnergyMethod import DeepEnergyMethod
from DEM_src.bc_helpers import get_boundary_conditions
from DEM_src.ObjectiveCalculator import ObjectiveCalculator


class StrainEnergy(ObjectiveCalculator):
    def __init__(self, E: float, nu: float, dxdy: tuple[float, float]):
        super().__init__(dxdy)
        self.E = E
        self.nu = nu

        self.penalizer = ElasticPenalizer()
        self.penalizer.set_penalization(3)

    def calculate_strain_energy(self, grad: list[list[torch.Tensor]]):
        # strain
        e_xx = grad[0][0]
        e_yy = grad[1][1]
        e_xy = 0.5 * (grad[1][0] + grad[0][1])

        # stress
        S_xx = self.E * (e_xx + self.nu * e_yy) / (1 - self.nu**2)
        S_yy = self.E * (e_yy + self.nu * e_xx) / (1 - self.nu**2)
        S_xy = self.E * e_xy / (1 + self.nu)

        strain_energy = e_xx * S_xx + e_yy * S_yy + 2 * e_xy * S_xy

        return strain_energy

    def calculate_objective_gradient(
        self, u: torch.Tensor, shape: tuple[int, int], density: torch.Tensor
    ):
        strain_energies = self.evaluate(u, shape, self.calculate_strain_energy)
        strain_energy_at_element = 0.5 * strain_energies * self.detJ

        objective = torch.sum(self.penalizer(density) * strain_energy_at_element)
        gradient = -self.penalizer.derivative(density) * strain_energy_at_element

        return gradient, objective

    def calculate_objective(
        self, u: torch.Tensor, shape: tuple[int, int], density: torch.Tensor
    ):
        strain_energies = self.evaluate(u, shape, self.calculate_strain_energy)
        strain_energy_at_element = 0.5 * strain_energies * self.detJ

        return torch.sum(self.penalizer(density) * strain_energy_at_element)


class ElasticityProblem:
    """Elastic compliance topology optimization problem."""

    def __init__(
        self,
        domain: Domain,
        device: torch.device,
        input_filter: csr_matrix,
        nn_parameters: NNParameters,
        elasticity_design: ElasticityDesign,
    ):
        self.design = elasticity_design
        self.filter = input_filter
        self.domain = domain

        traction_points_list, dirichlet_enforcer = get_boundary_conditions(
            domain, self.design
        )
        strain_energy = StrainEnergy(
            self.design.parameters.young_modulus,
            self.design.parameters.poisson_ratio,
            self.domain.dxdy,
        )

        self.dem = DeepEnergyMethod(
            device,
            nn_parameters,
            dirichlet_enforcer,
            traction_points_list,
            strain_energy,
        )

        self.objective_gradient = None

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

        return objective
