import torch
import numpy as np
import numpy.typing as npt
from scipy.sparse import csr_matrix

from designs.definitions import ElasticityDesign
from DEM_src.DeepEnergyMethod import DeepEnergyMethod
from DEM_src.bc_helpers import get_boundary_conditions
from DEM_src.data_structs import Domain, NNParameters


class ElasticityProblem:
    """Elastic compliance topology optimization problem."""

    def __init__(
        self,
        E: float,
        nu: float,
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
            domain, elasticity_design
        )
        self.dem = DeepEnergyMethod(
            E,
            nu,
            device,
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
