import torch
import numpy as np
import numpy.typing as npt


class StrainEnergy:
    def __init__(self, E: float, nu: float, dxdy: tuple[float, float]):
        self.E = E
        self.nu = nu

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
    ) -> list[list[torch.Tensor]]:
        dN_dxy_list = np.array(
            [
                np.matmul(self.Jinv, dN_dsy[:, gauss_point].reshape((2, 1)))
                for dN_dsy in self.shape_derivatives
            ]
        ).reshape((len(self.shape_derivatives), 2))

        Ux_list, Uy_list = point_lists
        dUxdx = sum(dx * Ux for dx, Ux in zip(dN_dxy_list[:, 0], Ux_list))
        dUxdy = sum(dy * Ux for dy, Ux in zip(dN_dxy_list[:, 1], Ux_list))
        dUydx = sum(dx * Uy for dx, Uy in zip(dN_dxy_list[:, 0], Uy_list))
        dUydy = sum(dy * Uy for dy, Uy in zip(dN_dxy_list[:, 1], Uy_list))

        return [[dUxdx, dUxdy], [dUydx, dUydy]]

    def strain_at_gauss_point(
        self,
        gauss_point: int,
        point_lists: list[list[torch.Tensor]],
        density: torch.Tensor,
    ):
        grad = self.get_grad_U(gauss_point, point_lists)

        # strains at all gauss quadrature points
        e_xx = grad[0][0]
        e_yy = grad[1][1]
        e_xy = 0.5 * (grad[1][0] + grad[0][1])

        # stresses at all gauss quadrature points
        S_xx = (self.E * (e_xx + self.nu * e_yy) / (1 - self.nu**2)) * density
        S_yy = (self.E * (e_yy + self.nu * e_xx) / (1 - self.nu**2)) * density
        S_xy = (self.E * e_xy / (1 + self.nu)) * density

        strain_energy = e_xx * S_xx + e_yy * S_yy + 2 * e_xy * S_xy

        return strain_energy

    def calculate_objective_gradient(
        self, u: torch.Tensor, shape: tuple[int, int], density: torch.Tensor
    ):
        _, dim = u.shape
        U = torch.transpose(u.reshape(shape[1], shape[0], dim), 0, 1)

        point_lists = self.get_gauss_points(U)

        # SIMP
        simp_exponent = 3.0
        SIMP_derivative = simp_exponent * torch.pow(density, simp_exponent - 1.0)

        strain_energies: torch.Tensor = sum(
            self.strain_at_gauss_point(i, point_lists, SIMP_derivative)
            for i in range(len(self.shape_derivatives))
        )
        strain_energy_at_element = 0.5 * strain_energies * self.detJ

        return -strain_energy_at_element, torch.sum(
            strain_energy_at_element * density / simp_exponent
        )

    def calculate_objective(
        self, u: torch.Tensor, shape: tuple[int, int], density: torch.Tensor
    ):
        _, dim = u.shape
        U = torch.transpose(u.reshape(shape[1], shape[0], dim), 0, 1)

        point_lists = self.get_gauss_points(U)

        # SIMP
        simp_exponent = 3.0
        SIMP = torch.pow(density, simp_exponent)

        strain_energies: torch.Tensor = sum(
            self.strain_at_gauss_point(i, point_lists, SIMP)
            for i in range(len(self.shape_derivatives))
        )
        strain_energy_at_element = 0.5 * strain_energies * self.detJ

        return torch.sum(strain_energy_at_element)
