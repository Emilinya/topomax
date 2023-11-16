from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import numpy as np

from DEM_src.data_structs import Domain
from designs.definitions import Side, Traction, ElasticityParameters


class DirichletEnforcer(ABC):
    def create_zero_enforcer(
        self, sides: list[Side], domain: Domain, device: torch.device
    ):
        # ensure that sides does not contain any duplicates
        sides = sorted(list(set(sides)), key=lambda v: v.value)

        # normalize x- and y-values
        flat_norm_x = domain.x_grid.T.flatten().reshape((-1, 1)) / domain.length
        flat_norm_y = domain.y_grid.T.flatten().reshape((-1, 1)) / domain.height

        zero_enforcer = np.ones((flat_norm_x.size, 2))
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
        self, parameters: ElasticityParameters, domain: Domain, device: torch.device
    ):
        self.zero_enforcer = self.create_zero_enforcer(
            parameters.fixed_sides, domain, device
        )

    def __call__(self, u: torch.Tensor):
        return u * self.zero_enforcer


class TractionPoints:
    def __init__(self, domain: Domain, traction: Traction):
        side, center, length, value = traction.to_tuple()
        left = center - length / 2
        right = center + length / 2

        flat_x = domain.x_grid.T.flatten()
        flat_y = domain.y_grid.T.flatten()

        if side == Side.LEFT:
            side_condition = flat_x == 0
            side_points = flat_y
        elif side == Side.RIGHT:
            side_condition = flat_x == domain.length
            side_points = flat_y
        elif side == Side.TOP:
            side_condition = flat_y == domain.height
            side_points = flat_x
        elif side == Side.BOTTOM:
            side_condition = flat_y == 0
            side_points = flat_x
        else:
            raise ValueError(f"Unknown side: '{side}'")

        if side in (Side.LEFT, Side.RIGHT):
            self.side_index = 1
            self.stride = 1
            self.width = domain.Ny + 1
        elif side in (Side.TOP, Side.BOTTOM):
            self.side_index = 0
            self.stride = domain.Ny + 1
            self.width = domain.Nx + 1

        left_condition = side_points >= left
        right_condition = side_points <= right
        (load_indices,) = np.where(side_condition & left_condition & right_condition)

        load_points = np.array([flat_x[load_indices], flat_y[load_indices]]).T

        self.value = value
        self.points = load_points
        self.indices = load_indices
        self.left_error = load_points[0, self.side_index] - left
        self.right_error = right - load_points[-1, self.side_index]
