import torch
import numpy as np


class InternalEnergy:
    def __init__(self, E, nu):
        self.E = E
        self.nu = nu

    def strain_at_gauss_point(self, Jinv, gauss_point, lists, density_SIMP):
        dN_dsy_list, UxN_list, UyN_list = lists

        # strain energy at gauss point
        dN_dxy_list = [
            np.matmul(Jinv, dN_dsy[:, gauss_point].reshape((2, 1)))
            for dN_dsy in dN_dsy_list
        ]

        dUxdx = sum(dN_dxy[0][0] * UxN for dN_dxy, UxN in zip(dN_dxy_list, UxN_list))
        dUxdy = sum(dN_dxy[1][0] * UxN for dN_dxy, UxN in zip(dN_dxy_list, UxN_list))
        dUydx = sum(dN_dxy[0][0] * UyN for dN_dxy, UyN in zip(dN_dxy_list, UyN_list))
        dUydy = sum(dN_dxy[1][0] * UyN for dN_dxy, UyN in zip(dN_dxy_list, UyN_list))

        # strains at all gauss quadrature points
        e_xx = dUxdx
        e_yy = dUydy
        e_xy = 0.5 * (dUydx + dUxdy)

        # stresses at all gauss quadrature points
        S_xx = (self.E * (e_xx + self.nu * e_yy) / (1 - self.nu**2)) * density_SIMP
        S_yy = (self.E * (e_yy + self.nu * e_xx) / (1 - self.nu**2)) * density_SIMP
        S_xy = (self.E * e_xy / (1 + self.nu)) * density_SIMP

        strain_energy = e_xx * S_xx + e_yy * S_yy + 2 * e_xy * S_xy

        return strain_energy

    def Elastic2DGaussQuad(self, u, x, dxdydz, shape, density, SENSITIVITY):
        Ux = torch.transpose(u[:, 0].unsqueeze(1).reshape(shape[0], shape[1]), 0, 1)
        Uy = torch.transpose(u[:, 1].unsqueeze(1).reshape(shape[0], shape[1]), 0, 1)

        axis = -1

        nd = Ux.ndimension()
        slice1 = [slice(None)] * nd
        slice2 = [slice(None)] * nd
        slice1[axis] = slice(1, None)
        slice2[axis] = slice(None, -1)

        UxN1 = Ux[: (shape[1] - 1)][tuple(slice2)]
        UxN2 = Ux[1 : shape[1]][tuple(slice2)]
        UxN3 = Ux[0 : (shape[1] - 1)][tuple(slice1)]
        UxN4 = Ux[1 : shape[1]][tuple(slice1)]
        UxN_list = [UxN1, UxN2, UxN3, UxN4]

        UyN1 = Uy[: (shape[1] - 1)][tuple(slice2)]
        UyN2 = Uy[1 : shape[1]][tuple(slice2)]
        UyN3 = Uy[0 : (shape[1] - 1)][tuple(slice1)]
        UyN4 = Uy[1 : shape[1]][tuple(slice1)]
        UyN_list = [UyN1, UyN2, UyN3, UyN4]

        # SIMP
        simp_exponent = 3.0
        if SENSITIVITY:
            density_SIMP = simp_exponent * torch.pow(density, simp_exponent - 1.0)
        else:
            density_SIMP = torch.pow(density, simp_exponent)

        ## Differentiation of shape functions at gauss quadrature points
        dN1_dsy = np.array(
            [
                [-0.394337567, -0.105662433, -0.105662433, -0.394337567],
                [-0.394337567, -0.394337567, -0.105662433, -0.105662433],
            ]
        )
        dN2_dsy = np.array(
            [
                [-0.105662433, -0.394337567, -0.394337567, -0.105662433],
                [0.394337567, 0.394337567, 0.105662433, 0.105662433],
            ]
        )
        dN3_dsy = np.array(
            [
                [0.394337567, 0.105662433, 0.105662433, 0.394337567],
                [-0.105662433, -0.105662433, -0.394337567, -0.394337567],
            ]
        )
        dN4_dsy = np.array(
            [
                [0.105662433, 0.394337567, 0.394337567, 0.105662433],
                [0.105662433, 0.105662433, 0.394337567, 0.394337567],
            ]
        )
        dN_dsy_list = [dN1_dsy, dN2_dsy, dN3_dsy, dN4_dsy]

        lists = [dN_dsy_list, UxN_list, UyN_list]

        dx = dxdydz[0]
        dy = dxdydz[1]
        dxds = dx / 2
        dydt = dy / 2

        J = np.array([[dxds, 0], [0, dydt]])
        Jinv = np.linalg.inv(J)
        detJ = np.linalg.det(J)

        strain_energies = (
            self.strain_at_gauss_point(Jinv, 0, lists, density_SIMP)
            + self.strain_at_gauss_point(Jinv, 1, lists, density_SIMP)
            + self.strain_at_gauss_point(Jinv, 2, lists, density_SIMP)
            + self.strain_at_gauss_point(Jinv, 3, lists, density_SIMP)
        )

        # Strain energy at element
        strainEnergy_at_elem = 0.5 * strain_energies * detJ

        if SENSITIVITY:
            return -strainEnergy_at_elem, torch.sum(
                strainEnergy_at_elem * density / simp_exponent
            )
        return torch.sum(strainEnergy_at_elem)
