import torch
import numpy as np
import numpy.typing as npt

from DEM_src.utils import Mesh
from DEM_src.domains import SideDomain, CircleDomain


def integrate(data: npt.NDArray[np.float64], mesh: Mesh) -> float:
    dx, dy = mesh.dxdy

    if data.shape == mesh.shape:
        """
        Data is evaluated at the nodes of the mesh
        x─x─x─x
        │ │ │ │
        x─x─x─x
        """
        return np.trapz(np.trapz(data, dx=dx), dx=dy)

    if data.shape == mesh.intervals:
        """
        Data is evaluated at the centers of the mesh
        ┌─┬─┬─┐
        │x│x│x│
        └─┴─┴─┘
        """
        return float(np.sum(data)) * dx * dy

    raise ValueError(
        f"Got data with shape {data.shape}, "
        + f"but it must have shape {mesh.shape} or {mesh.intervals}"
    )


def lerp(a, b, t: float):
    """
    Interpolate between a and b linearly. t∈[0, 1]
    represents how far along the segment [a, b] you are
    """
    return a * t + b * (1 - t)


def circular_integral(
    data: torch.Tensor,
    mesh: Mesh,
    domain: CircleDomain,
):
    if data.shape != mesh.shape:
        raise ValueError(
            f"Got data with shape {data.shape}, "
            + f"but it must have shape {mesh.shape}"
        )

    if domain.torch_areas is None:
        raise ValueError(
            "Tried to integrate over CircleDomain with torch_areas=None. "
            + "You must give CircleDomain a device!"
        )

    # this is inefficient, how to fix?
    center_data = (data[1:, 1:] + data[1:, :-1] + data[:-1, 1:] + data[:-1, :-1]) / 4

    values = center_data[tuple(domain.indices.T)]

    return torch.sum(values * domain.torch_areas)


def boundary_integral(
    data: torch.Tensor,
    mesh: Mesh,
    domain: SideDomain,
):
    if data.shape != (np.prod(mesh.shape)):
        raise ValueError(
            f"Got data with shape {data.shape}, "
            + f"but it must have shape ({np.prod(mesh.shape)})"
        )

    ds = mesh.dxdy[domain.side_index]
    boundary_data = data[domain.indices]

    left_correction, right_correction = 0.0, 0.0

    # if we are not on the left edge, add left lerp correction
    if (domain.indices[0] / domain.stride) % domain.width != 0:
        left_t = domain.left_error / ds
        left_data = data[domain.indices[0] - domain.stride]
        left_lerp = lerp(left_data, boundary_data[0], left_t)
        left_correction = domain.left_error * (left_lerp + boundary_data[0]) / 2

    # if we are not on the right edge, add right lerp correction
    if ((domain.indices[-1] / domain.stride) + 1) % domain.width != 0:
        right_t = domain.right_error / ds
        right_data = data[domain.indices[-1] + domain.stride]
        right_lerp = lerp(right_data, boundary_data[-1], right_t)
        right_correction = domain.right_error * (right_lerp + boundary_data[-1]) / 2

    return left_correction + right_correction + torch.trapezoid(boundary_data, dx=ds)
