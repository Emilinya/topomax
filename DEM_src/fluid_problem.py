from __future__ import annotations

import torch
import numpy as np
import numpy.typing as npt

from designs.definitions import FluidDesign
from src.penalizers import FluidPenalizer
from DEM_src.data_structs import Domain
from DEM_src.bc_helpers import FluidEnforcer
from DEM_src.ObjectiveCalculator import ObjectiveCalculator
from DEM_src.DeepEnergyMethod import NNParameters, DeepEnergyMethod


class FluidEnergy(ObjectiveCalculator):
    def __init__(self, dxdy: tuple[float, float], viscocity: float):
        super().__init__(dxdy)
        self.viscocity = viscocity

        self.penalizer = FluidPenalizer()
        self.penalizer.set_penalization(0.1)

    def calculate_potential(self, u: torch.Tensor, grad_u: torch.Tensor):
        """Calculate |u|², |∇u|² and |∇·u|²."""

        u_norm, grad_u_norm = self.calculate_norms(u, grad_u)
        div_u = grad_u[0][0] + grad_u[1][1]

        return [u_norm, grad_u_norm, div_u**2]

    def calculate_norms(self, u: torch.Tensor, grad_u: torch.Tensor):
        """Calculate |u|² and |∇u|²."""

        return [torch.sum(u**2, 0), torch.sum(grad_u**2, [0, 1])]

    def calculate_potential_power(
        self, u: torch.Tensor, shape: tuple[int, int], density: torch.Tensor
    ):
        """Calculate ψ(u; ρ) = ∫½(r(ρ)|u|² + μ|∇u|²) + γ|∇·u|² dx."""

        u_norm, grad_norm, div_norm = self.evaluate(u, shape, self.calculate_potential)
        gamma = 1000000

        potential = (
            0.5 * (self.penalizer(density) * u_norm + self.viscocity * grad_norm)
            + gamma * div_norm
        )

        return torch.sum(potential * self.detJ)

    def calculate_objective_and_gradient(
        self, u: torch.Tensor, shape: tuple[int, int], density: torch.Tensor
    ):
        """Calculate ∇ϕ(ρ; u) = ½r'(ρ)|u|² and ϕ(ρ; u) = ½∫r(ρ)|u|² + μ|∇u|² dx."""

        u_norm, grad_norm = self.evaluate(u, shape, self.calculate_norms)
        potential = self.penalizer(density) * u_norm + self.viscocity * grad_norm

        objective = 0.5 * torch.sum(potential * self.detJ)
        gradient = 0.5 * self.penalizer.derivative(density) * u_norm

        return objective, gradient


class FluidProblem:
    """Elastic compliance topology optimization problem."""

    def __init__(
        self,
        domain: Domain,
        device: torch.device,
        verbose: bool,
        fluid_design: FluidDesign,
    ):
        self.design = fluid_design
        self.domain = domain

        dirichlet_enforcer = FluidEnforcer(self.design.parameters, self.domain, device)
        fluid_energy = FluidEnergy(self.domain.dxdy, self.design.parameters.viscosity)

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
        self.dem = DeepEnergyMethod(
            device,
            verbose,
            dimension,
            nn_parameters,
            dirichlet_enforcer,
            fluid_energy,
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
        objective, objective_gradient = self.dem.train_model(rho, self.domain)

        objective = objective.cpu().detach().numpy()
        self.objective_gradient = objective_gradient.cpu().detach().numpy()

        return float(objective)
