import torch
import numpy as np

from DeepEnergy.src.data_structs import Domain
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
        if side == Side.LEFT:
            side_condition = domain.coordinates[:, 0] == 0
            side_points = domain.coordinates[:, 1]
        elif side == Side.RIGHT:
            side_condition = domain.coordinates[:, 0] == domain.length
            side_points = domain.coordinates[:, 1]
        elif side == Side.TOP:
            side_condition = domain.coordinates[:, 1] == domain.height
            side_points = domain.coordinates[:, 0]
        elif side == Side.BOTTOM:
            side_condition = domain.coordinates[:, 1] == 0
            side_points = domain.coordinates[:, 0]
        else:
            raise ValueError(f"Unknown side: '{side}'")

        left_condition = side_points >= center - length / 2.0
        right_condition = side_points <= center + length / 2.0
        load_idxs = np.where(side_condition & left_condition & right_condition)
        load_points = domain.coordinates[load_idxs, :][0]
        load_values = np.ones(np.shape(load_points)) * value

        self.penalty = 1.0
        self.coord = load_points
        self.known_value = load_values
        self.idx = np.asarray(load_idxs)


def get_boundary_conditions(domain: Domain, elasticity_design: ElasticityDesign):
    assert elasticity_design.parameters.tractions is not None

    traction_points_list: list[TractionPoints] = []
    for traction in elasticity_design.parameters.tractions:
        traction_points_list.append(TractionPoints(domain, traction))

    dirichlet_enforcer = DirichletEnforcer(elasticity_design.parameters.fixed_sides)

    return traction_points_list, dirichlet_enforcer
