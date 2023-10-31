import torch
import numpy as np
import numpy.typing as npt

from DeepEnergy.src.DeepEnergyMethod import DeepEnergyMethod
from DeepEnergy.src.data_structs import Domain, TopOptParameters, NNParameters


def get_boundary_load(
    domain: Domain, side: str, center: float, length: float, value: list[float]
):
    if side == "left":
        side_condition = domain.coordinates[:, 0] == 0
        side_points = domain.coordinates[:, 1]
    elif side == "right":
        side_condition = domain.coordinates[:, 0] == domain.length
        side_points = domain.coordinates[:, 1]
    elif side == "top":
        side_condition = domain.coordinates[:, 1] == domain.height
        side_points = domain.coordinates[:, 0]
    elif side == "bottom":
        side_condition = domain.coordinates[:, 1] == 0
        side_points = domain.coordinates[:, 0]
    else:
        raise ValueError(f"Unknown side: '{side}'")

    left_condition = side_points >= center - length / 2.0
    right_condition = side_points <= center + length / 2.0
    load_idxs = np.where(side_condition & left_condition & right_condition)
    load_points = domain.coordinates[load_idxs, :][0]
    load_values = np.ones(np.shape(load_points)) * value

    return load_idxs, load_points, load_values


def get_boundary_conditions(domain: Domain, example: int):
    if example == 1:
        # downward load on the top of the domain
        load_idxs, load_points, load_values = get_boundary_load(
            domain,
            side="top",
            center=domain.length / 2,
            length=0.5,
            value=[0.0, -2000.0],
        )

        # fixed on left and right side
        def u0(x: torch.Tensor, y: torch.Tensor):
            return 0

        def m(x: torch.Tensor, y: torch.Tensor):
            return x * (1 - x)

    elif example == 2:
        # downward load on the right side of the domain
        _, dy = domain.dxdy
        load_idxs, load_points, load_values = get_boundary_load(
            domain,
            side="right",
            center=domain.height / 2,
            length=dy,
            value=[0.0, -2000.0],
        )

        # fixed on left side
        def u0(x: torch.Tensor, y: torch.Tensor):
            return 0

        def m(x: torch.Tensor, y: torch.Tensor):
            return x

    else:
        raise ValueError(f"Unknown example: {example}")

    neumannBC = {
        "neumann_1": {
            "coord": load_points,
            "known_value": load_values,
            "penalty": 1.0,
            "idx": np.asarray(load_idxs),
        }
    }

    dirichletBC = {"m": m, "u0": u0}

    return neumannBC, dirichletBC


class ElasticityProblem:
    """Elastic compliance topology optimization problem."""

    def __init__(
        self,
        device: torch.device,
        example: int,
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

        neumannBC, dirichletBC = get_boundary_conditions(train_domain, example)
        self.dem = DeepEnergyMethod(
            device, neumannBC, dirichletBC, to_parameters, nn_parameters
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
