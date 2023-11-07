import numpy as np
import torch

from DEM_src.external_energy import calculate_external_energy
from DEM_src.bc_helpers import TractionPoints
from DEM_src.data_structs import Domain
from designs.definitions import Traction, Side


def test_calculate_external_energy():
    device = torch.device("cpu")

    Ns = [1, 5, 10, 20]
    xy_err_list = []
    yx_err_list = []

    l = 0.5
    for N in Ns:
        domain = Domain(12 * N, 3 * N, 12, 2)

        # we want to calculate int_\delta\Omega u \cdot t dx
        # set t = (0, -1) if (L-l)/2 < x < (L+l)/2, y=H
        traction_points_list = [
            TractionPoints(
                domain, Traction(Side.TOP, domain.length / 2, l, (0.0, -1.0))
            )
        ]

        # if u = (x, y), then int_\delta\Omega u \cdot t = int_{(L-l)/2}^{(L+l)/2} -H dx = -lH
        u = torch.from_numpy(domain.coordinates).float()
        numeric = float(
            calculate_external_energy(u, domain.dxdy, device, traction_points_list)
        )
        analytic = -l * domain.height
        xy_err_list.append(abs(numeric - analytic) / abs(analytic) * 100)

        # if u = (y, x), then int_\delta\Omega u \cdot t = int_{(L-l)/2}^{(L+l)/2} -x dx = -lL/2
        u_rev = torch.from_numpy(domain.coordinates[:, ::-1].copy()).float()
        numeric = float(
            calculate_external_energy(u_rev, domain.dxdy, device, traction_points_list)
        )
        analytic = -l * domain.length / 2
        yx_err_list.append(abs(numeric - analytic) / abs(analytic) * 100)

    xy_degree = np.polynomial.Polynomial.fit(np.log(Ns), np.log(xy_err_list), 1).coef[1]
    yx_degree = np.polynomial.Polynomial.fit(np.log(Ns), np.log(yx_err_list), 1).coef[1]

    assert xy_degree <= -2
    assert yx_degree <= -2
