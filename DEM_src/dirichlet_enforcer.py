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
        domain: Mesh,
        device: torch.device,
        output_dimension: int,
    ):
        # ensure that sides does not contain any duplicates
        sides = sorted(list(set(sides)), key=lambda v: v.value)

        # normalize x- and y-values
        flat_norm_x = domain.x_grid.T.flatten().reshape((-1, 1)) / domain.length
        flat_norm_y = domain.y_grid.T.flatten().reshape((-1, 1)) / domain.height

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
    def __call__(self, u: torch.Tensor) -> torch.Tensor:
        ...


class ElasticityEnforcer(DirichletEnforcer):
    def __init__(
        self, parameters: ElasticityParameters, domain: Mesh, device: torch.device
    ):
        self.zero_enforcer = self.create_zero_enforcer(
            parameters.fixed_sides, domain, device, 2
        )

    def __call__(self, u: torch.Tensor):
        return u * self.zero_enforcer


class FluidEnforcer(DirichletEnforcer):
    def __init__(
        self, fluid_parameters: FluidParameters, domain: Mesh, device: torch.device
    ):
        flow_sides = [flow.side for flow in fluid_parameters.flows]
        if fluid_parameters.no_slip is None:
            all_sides = [Side.LEFT, Side.RIGHT, Side.TOP, Side.BOTTOM]
            no_slips = list(set(all_sides).difference(flow_sides))
        else:
            no_slips = fluid_parameters.no_slip

        self.zero_enforcer = self.create_zero_enforcer(
            flow_sides + no_slips, domain, device, 2
        )

        self.flow_enforcer = self.create_flow_enforcer(
            fluid_parameters.flows, domain, device
        )

        if fluid_parameters.zero_pressure is not None:
            raise ValueError("TODO: handle zero pressure boundary condition")

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

    def create_flow_enforcer(
        self, flows: list[Flow], domain: Mesh, device: torch.device
    ):
        flow_enforcer_ux = np.zeros_like(domain.x_grid)
        flow_enforcer_uy = np.zeros_like(domain.y_grid)

        for flow in flows:
            side, center, length, rate = flow.to_tuple()

            if side == Side.LEFT:
                flow_enforcer_ux[:, 0] += self.get_flow(
                    domain.y_grid[:, 0], center, length, rate
                )
            elif side == Side.RIGHT:
                flow_enforcer_ux[:, -1] -= self.get_flow(
                    domain.y_grid[:, -1], center, length, rate
                )
            elif side == Side.TOP:
                flow_enforcer_uy[-1, :] -= self.get_flow(
                    domain.x_grid[-1, :], center, length, rate
                )
            elif side == Side.BOTTOM:
                flow_enforcer_uy[0, :] = self.get_flow(
                    domain.x_grid[0, :], center, length, rate
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
