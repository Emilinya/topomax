from __future__ import annotations

from typing import Any, Iterable

import torch
import numpy as np
import numpy.typing as npt


class Mesh:
    def __init__(
        self,
        Nx: int,
        Ny: int,
        width: float,
        height: float,
    ):
        self.Nx = Nx
        self.Ny = Ny
        self.width = width
        self.height = height

        # create points
        x_ray = np.linspace(0.0, self.width, self.Nx + 1)
        y_ray = np.linspace(0.0, self.height, self.Ny + 1)

        self.x_grid, self.y_grid = np.meshgrid(x_ray, y_ray)

        self.intervals = (self.Ny, self.Nx)
        self.shape = (self.Ny + 1, self.Nx + 1)
        self.dxdy = (self.width / self.Nx, self.height / self.Ny)


def flatten(grids: Iterable[npt.NDArray[Any]]):
    """
    Turns a list of n grids [g_i] of shape (Ny, Nx) into one
    array of (g_0, ..., g_n) pairs with shape (Nx*Ny, n)

    Example
    --------
    >>> flatten([np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]])])
    array([[1, 5], [3, 7], [2, 6], [4, 8]])
    """

    return np.array([grid.T.flat for grid in grids]).T


def unflatten(flat_values: npt.NDArray | torch.Tensor, shape: tuple[int, int]):
    """
    Turns an array with shape (Nx*Ny, n), together with an input shape (Ny, Nx),
    into an array of shape (n, Ny, Nx)

    Examples
    --------
    >>> unflatten(np.array([[1, 5], [3, 7], [2, 6], [4, 8]]), (2, 2))
    array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    """

    Ny, Nx = shape
    NxNy, dim = flat_values.shape

    if Ny * Nx != NxNy:
        raise ValueError(
            f"Got an input array with {NxNy} elements, but input shape only has "
            + f"{Nx}*{Ny}={Nx*Ny} elements"
        )

    reshaped_values = flat_values.reshape(Nx, Ny, dim)

    if isinstance(reshaped_values, torch.Tensor):
        # The use of `x.T` on tensors of dimension other
        # than 2 to reverse their shape is deprecated :(
        return reshaped_values.permute((2, 1, 0))
    return reshaped_values.T
