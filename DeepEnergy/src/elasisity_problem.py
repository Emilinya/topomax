import torch
import numpy as np
import numpy.typing as npt

from designs.definitions import ElasticityDesign
from DeepEnergy.src.DeepEnergyMethod import DeepEnergyMethod
from DeepEnergy.src.bc_helpers import get_boundary_conditions
from DeepEnergy.src.data_structs import Domain, NNParameters, TopOptParameters


class ElasticityProblem:
    """Elastic compliance topology optimization problem."""

    def __init__(
        self,
        device: torch.device,
        elasticity_design: ElasticityDesign,
        test_domain: Domain,
        train_domain: Domain,
        input_filter: npt.NDArray[np.float64],
        to_parameters: TopOptParameters,
        nn_parameters: NNParameters,
    ):
        self.filter = input_filter
        self.test_domain = test_domain
        self.train_domain = train_domain
        self.to_parameters = to_parameters

        traction_points_list, dirichlet_enforcer = get_boundary_conditions(
            train_domain, elasticity_design
        )
        self.dem = DeepEnergyMethod(
            device,
            to_parameters,
            nn_parameters,
            dirichlet_enforcer,
            traction_points_list,
        )

        self.objective_gradient = None

    def calculate_objective_gradient(self):
        if self.objective_gradient is None:
            raise ValueError(
                "You must call calculate_objective "
                + "before calling calculate_objective_gradient"
            )

        return self.objective_gradient

    def calculate_objective(self, rho):
        filtered_rho = self.filter @ rho

        objective, objective_gradient = self.dem.train_model(
            filtered_rho, self.train_domain
        )

        objective = objective.cpu().detach().numpy()
        objective_gradient = objective_gradient.cpu().detach().numpy()

        # invert filter
        self.objective_gradient = self.filter.T @ objective_gradient.flatten()

        return objective
