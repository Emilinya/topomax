import numpy as np
import numpy.typing as npt

from DEM_src.data_structs import Domain


def integrate(data: npt.NDArray[np.float64], domain: Domain) -> float:
    if data.shape != domain.shape:
        raise ValueError(
            f"Got data with shape {data.shape}, "
            + f"but it must have shape {domain.shape}"
        )

    dx, dy = domain.dxdy
    return np.trapz(np.trapz(data, dx=dx), dx=dy)
