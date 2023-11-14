from __future__ import annotations

import torch

from DEM_src.bc_helpers import TractionPoints


def lerp(a, b, t: float):
    """
    A function that interpolates between a and b linearly.
    t∈[0, 1] represents  how far along the segment [a, b] you are
    """
    return a * t + b * (1 - t)


def calculate_external_energy(
    u: torch.Tensor,
    dxdy: tuple[float, float],
    traction_points_list: list[TractionPoints],
):
    external_energy = torch.tensor(0.0)

    for tps in traction_points_list:
        ds = dxdy[tps.side_index]
        traction = torch.tensor(tps.value)
        dot_products = torch.sum(traction * u[tps.indices], dim=1)

        left_correction, right_correction = 0.0, 0.0

        # if we are not on the left edge, add left lerp correction
        if (tps.indices[0] / tps.stride) % tps.width != 0:
            left_t = tps.left_error / ds
            left_u = u[tps.indices[0] - tps.stride]
            left_lerp = lerp(torch.dot(traction, left_u), dot_products[0], left_t)
            left_correction = tps.left_error * (left_lerp + dot_products[0]) / 2

        # if we are not on the right edge, add right lerp correction
        if ((tps.indices[-1] / tps.stride) + 1) % tps.width != 0:
            right_t = tps.right_error / ds
            right_u = u[tps.indices[-1] + tps.stride]
            right_lerp = lerp(torch.dot(traction, right_u), dot_products[-1], right_t)
            right_correction = tps.right_error * (right_lerp + dot_products[-1]) / 2

        return left_correction + right_correction + torch.trapezoid(dot_products, dx=ds)

    return external_energy
