import numpy as np
import numpy.typing as npt

from DEM_src.data_structs import Domain


def integrate(data: npt.NDArray[np.float64], domain: Domain) -> float:
    dx, dy = domain.dxdy

    if data.shape == domain.shape:
        """
        Data is evaluated at the nodes of the domain
        ╳─╳─╳─╳
        │ │ │ │
        ╳─╳─╳─╳
        """
        return np.trapz(np.trapz(data, dx=dx), dx=dy)

    if data.shape == domain.intervals:
        """
        Data is evaluated at the centers of the domain
        ┌─┬─┬─┐
        │╳│╳│╳│
        └─┴─┴─┘
        """
        return float(np.sum(data)) * dx * dy

    raise ValueError(
        f"Got data with shape {data.shape}, "
        + f"but it must have shape {domain.shape} or {domain.intervals}"
    )
