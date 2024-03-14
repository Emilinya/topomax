from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import numpy as np
import numpy.typing as npt

from DEM_src.utils import Mesh, flatten
from designs.definitions import Side, Flow, FluidParameters, ElasticityParameters


class DirichletEnforcer(ABC):
    def create_zero_enforcer(
        self,
        sides: list[Side],
        mesh: Mesh,
        device: torch.device,
        output_dimension: int,
    ):
        # ensure that sides does not contain any duplicates
        sides = sorted(list(set(sides)), key=lambda v: v.value)

        # normalize x- and y-values
        flat_norm_x = mesh.x_grid.T.flatten().reshape((-1, 1)) / mesh.width
        flat_norm_y = mesh.y_grid.T.flatten().reshape((-1, 1)) / mesh.height

        zero_enforcer = np.ones((flat_norm_x.size, output_dimension))
        for side in sides:
            if side == Side.LEFT:
                zero_enforcer *= flat_norm_x
            elif side == Side.RIGHT:
                zero_enforcer *= 1 - flat_norm_x
            elif side == Side.TOP:
                zero_enforcer *= 1 - flat_norm_y
            elif side == Side.BOTTOM:
                zero_enforcer *= flat_norm_y
            else:
                raise ValueError(f"Unknown side: '{side}'")

        return torch.from_numpy(zero_enforcer).to(device).float()

    @abstractmethod
    def __call__(self, u: torch.Tensor) -> torch.Tensor: ...


class ElasticityEnforcer(DirichletEnforcer):
    def __init__(
        self, parameters: ElasticityParameters, mesh: Mesh, device: torch.device
    ):
        self.zero_enforcer = self.create_zero_enforcer(
            parameters.fixed_sides, mesh, device, 2
        )

    def __call__(self, u: torch.Tensor):
        return u * self.zero_enforcer


class FluidEnforcer(DirichletEnforcer):
    def __init__(
        self, fluid_parameters: FluidParameters, mesh: Mesh, device: torch.device
    ):
        self.zero_enforcer = self.create_zero_enforcer(Side.get_all(), mesh, device, 2)

        self.flow_enforcer = self.create_flow_enforcer(
            fluid_parameters.flows, mesh, device
        )

    def get_flow(
        self,
        position: npt.NDArray[np.float64],
        center: float,
        length: float,
        rate: float,
    ):
        output = np.zeros_like(position)

        t = position - center

        flow_idxs = np.where(((-length / 2) < t) & (t < (length / 2)))
        output[flow_idxs] = rate * (1 - (2 * t[flow_idxs] / length) ** 2)

        return output

    def create_flow_enforcer(self, flows: list[Flow], mesh: Mesh, device: torch.device):
        flow_enforcer_ux = np.zeros_like(mesh.x_grid)
        flow_enforcer_uy = np.zeros_like(mesh.y_grid)

        for flow in flows:
            side, center, length, rate = flow.to_tuple()

            if side == Side.LEFT:
                flow_enforcer_ux[:, 0] += self.get_flow(
                    mesh.y_grid[:, 0], center, length, rate
                )
            elif side == Side.RIGHT:
                flow_enforcer_ux[:, -1] -= self.get_flow(
                    mesh.y_grid[:, -1], center, length, rate
                )
            elif side == Side.TOP:
                flow_enforcer_uy[-1, :] -= self.get_flow(
                    mesh.x_grid[-1, :], center, length, rate
                )
            elif side == Side.BOTTOM:
                flow_enforcer_uy[0, :] = self.get_flow(
                    mesh.x_grid[0, :], center, length, rate
                )
            else:
                raise ValueError(f"Unknown side: '{side}'")

        return (
            torch.from_numpy(flatten([flow_enforcer_ux, flow_enforcer_uy]))
            .float()
            .to(device)
        )

    def __call__(self, u: torch.Tensor):
        return u * self.zero_enforcer + self.flow_enforcer
