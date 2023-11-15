from __future__ import annotations

from dataclasses import dataclass

import torch
import rff.layers
import numpy as np
import numpy.typing as npt

from DEM_src.data_structs import Domain
from DEM_src.bc_helpers import DirichletEnforcer
from DEM_src.ObjectiveCalculator import ObjectiveCalculator


@dataclass
class NNParameters:
    verbose: bool
    input_size: int
    output_size: int
    layer_count: int
    neuron_count: int
    learning_rate: float
    CNN_deviation: float
    rff_deviation: float
    iteration_count: int
    activation_function: str
    convergence_tolerance: float


class DeepEnergyMethod:
    # Instance attributes
    def __init__(
        self,
        device: torch.device,
        nn_parameters: NNParameters,
        dirichlet_enforcer: DirichletEnforcer,
        objective_calculator: ObjectiveCalculator,
    ):
        # self.data = data
        self.model = MultiLayerNet(nn_parameters)
        self.model = self.model.to(device)

        self.device = device
        self.nn_parameters = nn_parameters
        self.dirichlet_enforcer = dirichlet_enforcer
        self.objective_calculator = objective_calculator

        self.loss_array = []

    def train_model(self, rho: npt.NDArray[np.float64], domain: Domain):
        x = torch.from_numpy(
            np.array([domain.x_grid.T.flat, domain.y_grid.T.flat]).T
        ).float()
        x = x.to(self.device)
        x.requires_grad_(True)

        density = torch.from_numpy(rho).float()
        density = torch.reshape(density, domain.intervals).to(self.device)

        optimizer_LBFGS = torch.optim.LBFGS(
            self.model.parameters(),
            lr=self.nn_parameters.learning_rate,
            max_iter=20,
            line_search_fn="strong_wolfe",
        )

        def closure_generator(t: int):
            def closure():
                u_pred = self.get_U(x, domain)
                u_pred.double()

                # ---- Calculate internal and external energies------
                loss = self.objective_calculator.calculate_potential_power(
                    u_pred, domain.shape, density
                )
                optimizer_LBFGS.zero_grad()
                loss.backward()

                if self.nn_parameters.verbose:
                    print(f"Iter: {t+1:d} - Loss: {loss.item():.6e}")
                self.loss_array.append(loss.data.cpu())

                return float(loss)

            return closure

        for t in range(self.nn_parameters.iteration_count):
            # Zero gradients, perform a backward pass, and update the weights.

            optimizer_LBFGS.step(closure_generator(t))

            # Check convergence
            if self.convergence_check(
                self.loss_array,
                self.nn_parameters.convergence_tolerance,
            ):
                break

        u_pred = self.get_U(x, domain)
        return self.objective_calculator.calculate_objective_and_gradient(
            u_pred, domain.shape, density
        )

    def convergence_check(self, loss_array: list[float], tolerance: float):
        num_check = 10

        # Run minimum of 2*num_check iterations
        if len(loss_array) < 2 * num_check:
            return False

        mean1 = np.mean(loss_array[-2 * num_check : -num_check])
        mean2 = np.mean(loss_array[-num_check:])

        if np.abs(mean2) < 1e-6:
            return True

        if (np.abs(mean1 - mean2) / np.abs(mean2)) < tolerance:
            return True

        return False

    def get_U(
        self,
        x: torch.Tensor,
        domain: Domain,
    ):
        u_tilde: torch.Tensor = self.model(x)

        normed_x = (x[:, 0] / domain.length).unsqueeze(1)
        normed_y = (x[:, 1] / domain.height).unsqueeze(1)

        return self.dirichlet_enforcer(u_tilde, normed_x, normed_y)


class MultiLayerNet(torch.nn.Module):
    def __init__(self, parameters: NNParameters):
        super().__init__()

        input_size = parameters.input_size
        output_size = parameters.output_size
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
