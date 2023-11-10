import sys

import torch
import numpy as np

from DEM_src.data_structs import Domain
from DEM_src.bc_helpers import TractionPoints
from designs.definitions import Traction, Side
from DEM_src.external_energy import calculate_external_energy


def f_linear(x, y):
    return x * y


def F_linear(H: float, c: float, l: float):
    return H * c * l


def f_trig(x, y):
    return np.sin(9 * x) * np.sin(9 * y)


def F_trig(H: float, c: float, l: float):
    return 2 / 9 * np.sin(9 * H) * np.sin(9 * c) * np.sin(9 * l / 2)


def calculate_error(
    f,
    F,
    domain: Domain,
    traction: Traction,
    traction_points_list: list[TractionPoints],
):
    values = f(domain.x_grid, domain.y_grid)
    u = torch.from_numpy(np.array([values.T.flat, values.T.flat]).T).float()

    numeric = float(calculate_external_energy(u, domain.dxdy, traction_points_list))

    if traction.side in (Side.LEFT, Side.RIGHT):
        analytic = F(domain.length, traction.center, traction.length)
    elif traction.side in (Side.TOP, Side.BOTTOM):
        analytic = F(domain.height, traction.center, traction.length)
    else:
        sys.exit("???")

    return abs((numeric - analytic) / analytic) * 100


def get_errors(Ns: list[int], domain: Domain, traction: Traction):
    linear_err_list = []
    trig_err_list = []

    for N in Ns:
        N_domain = Domain(domain.Nx * N, domain.Ny * N, domain.length, domain.height)
        traction_points_list = [TractionPoints(N_domain, traction)]

        linear_err_list.append(
            calculate_error(
                f_linear, F_linear, N_domain, traction, traction_points_list
            )
        )

        trig_err_list.append(
            calculate_error(f_trig, F_trig, N_domain, traction, traction_points_list)
        )

    trig_degree = np.polynomial.Polynomial.fit(
        np.log(Ns), np.log(np.array(trig_err_list) + 1e-14), 1
    ).coef[1]

    return np.average(linear_err_list), trig_degree


def test_calculate_external_energy():
    Ns = list(range(2, 100))

    bridge_domain = Domain(4, 1, 12, 2)
    bridge_traction = Traction(Side.TOP, bridge_domain.length / 2, 0.5, (0.0, 1.0))

    cantilever_domain = Domain(10, 5, 2, 1)
    cantilever_traction = Traction(
        Side.RIGHT, cantilever_domain.height / 2, 0.5, (0.0, 1.0)
    )

    bridge_linear_average, bridge_trig_degree = get_errors(
        Ns, bridge_domain, bridge_traction
    )
    assert bridge_trig_degree <= -3
    assert bridge_linear_average < 1e-5

    cantilever_linear_average, cantilever_trig_degree = get_errors(
        Ns, cantilever_domain, cantilever_traction
    )

    assert cantilever_trig_degree <= -3
    assert cantilever_linear_average < 1e-5
