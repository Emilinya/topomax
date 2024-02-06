from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import rff.layers
import numpy as np
import numpy.typing as npt

from DEM_src.utils import Mesh, flatten
from DEM_src.dirichlet_enforcer import DirichletEnforcer
from DEM_src.ObjectiveCalculator import ObjectiveCalculator


@dataclass
class NNParameters:
    layer_count: int
    neuron_count: int
    learning_rate: float
    CNN_deviation: float
    rff_deviation: float
    iteration_count: int
    activation_function: str
    convergence_tolerance: float


class DeepEnergyMethod:
    def __init__(
        self,
        device: torch.device,
        verbose: bool,
        output_size: int,
        nn_parameters: NNParameters,
        dirichlet_enforcer: DirichletEnforcer,
        objective_calculator: ObjectiveCalculator,
    ):
        self.model = MultiLayerNet(2, output_size, nn_parameters)
        self.model = self.model.to(device)

        self.device = device
        self.verbose = verbose
        self.output_size = output_size
        self.nn_parameters = nn_parameters
        self.dirichlet_enforcer = dirichlet_enforcer
        self.objective_calculator = objective_calculator

        self.loss_array = []

    def set_nn_parameters(self, nn_parameters: NNParameters):
        self.model = MultiLayerNet(2, self.output_size, nn_parameters)
        self.model = self.model.to(self.device)
        self.nn_parameters = nn_parameters

    def train_model(self, rho: npt.NDArray[np.float64], mesh: Mesh):
        x = torch.from_numpy(flatten([mesh.x_grid, mesh.y_grid])).float()
        x = x.to(self.device)
        x.requires_grad_(True)

        density = torch.from_numpy(rho).float()
        density = torch.reshape(density, mesh.intervals).to(self.device)

        optimizer_LBFGS = torch.optim.LBFGS(
            self.model.parameters(),
            lr=self.nn_parameters.learning_rate,
            max_iter=20,
            line_search_fn="strong_wolfe",
        )

        def closure_generator(t: int):
            def closure():
                u_pred = self.get_u(x)
                u_pred.double()

                loss = self.objective_calculator.calculate_energy_form(
                    u_pred, mesh.shape, density
                )
                optimizer_LBFGS.zero_grad()
                loss.backward()

                if self.verbose:
                    line = f"Iter: {t+1:^3} - Loss: {loss.item():.6g}"
                    twidth = os.get_terminal_size().columns
                    print(f"\r{line}{' '*(twidth - len(line))}", end="")
                self.loss_array.append(loss.data.cpu())

                return float(loss)

            return closure

        for t in range(self.nn_parameters.iteration_count):
            optimizer_LBFGS.step(closure_generator(t))

            if self.convergence_check(
                self.loss_array,
                self.nn_parameters.convergence_tolerance,
            ):
                break

        if self.verbose:
            print()

        return self.objective_calculator.calculate_objective_and_gradient(
            self.get_u(x), mesh.shape, density
        )

    def get_loss(self, rho: npt.NDArray[np.float64], mesh: Mesh):
        density = torch.from_numpy(rho).float()
        density = torch.reshape(density, mesh.intervals).to(self.device)

        u_pred = self.get_u(mesh=mesh)
        loss = self.objective_calculator.calculate_energy_form(
            u_pred, mesh.shape, density
        )

        return float(loss)

    def get_u(self, x: torch.Tensor | None = None, mesh: Mesh | None = None):
        if x is None:
            if mesh is None:
                raise ValueError(
                    "get_u must get either x or mesh, they can't both be None"
                )

            x = torch.from_numpy(flatten([mesh.x_grid, mesh.y_grid])).float()
            x = x.to(self.device)

        return self.dirichlet_enforcer(self.model(x))

    def convergence_check(self, loss_array: list[float], tolerance: float):
        num_check = 10

        # Run minimum of 2 * num_check iterations
        if len(loss_array) < 2 * num_check:
            return False

        mean1 = np.mean(loss_array[-2 * num_check : -num_check])
        mean2 = np.mean(loss_array[-num_check:])

        if np.abs(mean2) < 1e-6:
            return True

        if (np.abs(mean1 - mean2) / np.abs(mean2)) < tolerance:
            return True

        return False


class MultiLayerNet(torch.nn.Module):
    def __init__(self, input_size, output_size, parameters: NNParameters):
        super().__init__()

        neuron_count = parameters.neuron_count
        rff_deviation = parameters.rff_deviation
        CNN_deviation = parameters.CNN_deviation

        self.layer_count = parameters.layer_count
        self.activation_function = getattr(torch, parameters.activation_function)
        self.encoding = rff.layers.GaussianEncoding(
            sigma=rff_deviation, input_size=input_size, encoded_size=neuron_count // 2
        )

        if self.layer_count < 2:
            raise ValueError(
                "Can't create MultiLayerNet with less than 2 layers "
                + f"(tried to create one with {self.layer_count} layers)"
            )

        self.linears = torch.nn.ModuleList()
        for i in range(self.layer_count):
            linear_inputs = input_size if i == 0 else neuron_count
            linear_outputs = output_size if i == self.layer_count - 1 else neuron_count
            self.linears.append(torch.nn.Linear(linear_inputs, linear_outputs))

            torch.nn.init.constant_(self.linears[i].bias, 0.0)
            torch.nn.init.normal_(self.linears[i].weight, mean=0, std=CNN_deviation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.encoding(x)
        for i in range(1, self.layer_count - 1):
            y = self.activation_function(self.linears[i](y))
        y = self.linears[-1](y)

        return y
