import torch
import numpy as np

from DEM_src.data_structs import Domain
from designs.definitions import Side, Traction, ElasticityDesign


class DirichletEnforcer:
    def __init__(self, fixed_sides: list[Side]):
        self.fixed_sides = sorted(list(set(fixed_sides)), key=lambda v: v.value)

    def __call__(self, u: torch.Tensor, x: torch.Tensor, y: torch.Tensor):
        zero_enforcer = torch.ones_like(u)
        for side in self.fixed_sides:
            if side == Side.LEFT:
                zero_enforcer *= x
            elif side == Side.RIGHT:
                zero_enforcer *= 1 - x
            elif side == Side.TOP:
                zero_enforcer *= 1 - y
            elif side == Side.BOTTOM:
                zero_enforcer *= y
            else:
                raise ValueError(f"Unknown side: '{side}'")

        return u * zero_enforcer


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


def get_boundary_conditions(domain: Domain, elasticity_design: ElasticityDesign):
    assert elasticity_design.parameters.tractions is not None

    traction_points_list: list[TractionPoints] = []
    for traction in elasticity_design.parameters.tractions:
        traction_points_list.append(TractionPoints(domain, traction))

    dirichlet_enforcer = DirichletEnforcer(elasticity_design.parameters.fixed_sides)

    return traction_points_list, dirichlet_enforcer
