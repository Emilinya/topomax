from __future__ import annotations

import sys
from typing import Callable
from abc import ABC, abstractmethod

import torch
import numpy as np
import numpy.typing as npt


class ObjectiveCalculator(ABC):
    def __init__(self, dxdy: tuple[float, float]):
        self.dxdy = dxdy
        self.Jinv, self.detJ = self.calculate_jacobian(dxdy)
        self.shape_derivatives = self.get_shape_derivatives()

    def calculate_jacobian(self, dxdy: tuple[float, float]):
        dx, dy = dxdy
        dxds = dx / 2
        dydt = dy / 2

        J = np.array([[dxds, 0], [0, dydt]])
        Jinv = np.linalg.inv(J)
        detJ: float = np.linalg.det(J)

        return Jinv, detJ

    def get_shape_derivatives(self) -> list[npt.NDArray[np.float64]]:
        """differentiation of shape functions at gauss quadrature points"""
        dN_dsy_1 = np.array(
            [
                [-0.394337567, -0.105662433, -0.105662433, -0.394337567],
                [-0.394337567, -0.394337567, -0.105662433, -0.105662433],
            ]
        )
        dN_dsy_2 = np.array(
            [
                [-0.105662433, -0.394337567, -0.394337567, -0.105662433],
                [0.394337567, 0.394337567, 0.105662433, 0.105662433],
            ]
        )
        dN_dsy_3 = np.array(
            [
                [0.394337567, 0.105662433, 0.105662433, 0.394337567],
                [-0.105662433, -0.105662433, -0.394337567, -0.394337567],
            ]
        )
        dN_dsy_4 = np.array(
            [
                [0.105662433, 0.394337567, 0.394337567, 0.105662433],
                [0.105662433, 0.105662433, 0.394337567, 0.394337567],
            ]
        )

        return [dN_dsy_1, dN_dsy_2, dN_dsy_3, dN_dsy_4]

    def get_gauss_points(self, U: torch.Tensor):
        """what does this function do? I don't know"""
        Ny, _, dim = U.shape

        axis = -1
        slice1 = [slice(None)] * 2
        slice2 = [slice(None)] * 2
        slice1[axis] = slice(1, None)
        slice2[axis] = slice(None, -1)

        point_lists: list[list[torch.Tensor]] = []
        for i in range(dim):
            Ux = U[:, :, i]

            UxN1 = Ux[: (Ny - 1)][tuple(slice2)]
            UxN2 = Ux[1:Ny][tuple(slice2)]
            UxN3 = Ux[0 : (Ny - 1)][tuple(slice1)]
            UxN4 = Ux[1:Ny][tuple(slice1)]
            point_lists.append([UxN1, UxN2, UxN3, UxN4])

        return point_lists

    def get_grad_U(
        self, gauss_point: int, point_lists: list[list[torch.Tensor]]
    ) -> torch.Tensor:
        dxdy_list = np.array(
            [
                np.matmul(self.Jinv, dN_dsy[:, gauss_point].reshape((2, 1)))
                for dN_dsy in self.shape_derivatives
            ]
        ).reshape((len(self.shape_derivatives), 2))
        Ux_list, Uy_list = point_lists

        gradient = torch.zeros((len(point_lists), 2, *Ux_list[0].shape))

        for i in range(len(self.shape_derivatives)):
            dx, dy = dxdy_list[i, :]
            Ux, Uy = Ux_list[i], Uy_list[i]

            gradient[0, 0, :, :] += Ux * dx
            gradient[0, 1, :, :] += Ux * dy
            gradient[1, 0, :, :] += Uy * dx
            gradient[1, 1, :, :] += Uy * dy

        return gradient

    def value_at_gauss_point(
        self,
        function: Callable[[torch.Tensor], torch.Tensor],
        gauss_point: int,
        point_lists: list[list[torch.Tensor]],
    ):
        grad = self.get_grad_U(gauss_point, point_lists)
        return function(grad)

    def evaluate(
        self,
        u: torch.Tensor,
        shape: tuple[int, int],
        function: Callable[[torch.Tensor], torch.Tensor],
    ):
        _, dim = u.shape
        U = torch.transpose(u.reshape(shape[1], shape[0], dim), 0, 1)

        point_lists = self.get_gauss_points(U)

        value = sum(
            self.value_at_gauss_point(function, i, point_lists)
            for i in range(len(self.shape_derivatives))
        )

        # this only happens if len(self.shape_derivatives) == 0, which it should never be.
        # Nevertheless, the type checker gets angry if I don't do this, so I might as well
        if isinstance(value, int):
            sys.exit(f"evaluate returned {value}, this should not happen!")

        return value

    @abstractmethod
    def calculate_potential_power(
        self, u: torch.Tensor, shape: tuple[int, int], density: torch.Tensor
    ) -> torch.Tensor:
        ...

    @abstractmethod
    def calculate_objective_and_gradient(
        self, u: torch.Tensor, shape: tuple[int, int], density: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns the tuple (objective, gradient)"""
