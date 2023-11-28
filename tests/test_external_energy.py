import sys

import torch
import numpy as np

from DEM_src.utils import Mesh
from DEM_src.elasisity_problem import TractionPoints, calculate_traction_integral
from tests.utils import get_average, get_convergance
from designs.definitions import Traction, Side


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
    domain: Mesh,
    traction: Traction,
    traction_points_list: list[TractionPoints],
):
    values = f(domain.x_grid, domain.y_grid)
    u = torch.from_numpy(np.array([values.T.flat, values.T.flat]).T).float()

    numeric = float(calculate_traction_integral(u, domain.dxdy, traction_points_list))

    if traction.side in (Side.LEFT, Side.RIGHT):
        analytic = F(domain.length, traction.center, traction.length)
    elif traction.side in (Side.TOP, Side.BOTTOM):
        analytic = F(domain.height, traction.center, traction.length)
    else:
        sys.exit("???")

    return abs((numeric - analytic) / analytic) * 100


def test_calculate_traction_integral():
    Ns = list(range(2, 100))

    bridge_domain = Mesh(4, 1, 12, 2)
    bridge_traction = Traction(Side.TOP, bridge_domain.length / 2, 0.5, (0.0, 1.0))

    cantilever_domain = Mesh(10, 5, 2, 1)
    cantilever_traction = Traction(
        Side.RIGHT, cantilever_domain.height / 2, 0.5, (0.0, 1.0)
    )

    # this is not hacky at all, don't worry
    def get_get_error_function(domain, traction):
        def get_error_function(f, F):
            def error_function(N):
                N_domain = Mesh(
                    domain.Nx * N, domain.Ny * N, domain.length, domain.height
                )
                traction_points_list = [TractionPoints(N_domain, traction)]

                return calculate_error(f, F, N_domain, traction, traction_points_list)

            return error_function

        return get_error_function

    for domain, traction in [
        (bridge_domain, bridge_traction),
        (cantilever_domain, cantilever_traction),
    ]:
        get_error_function = get_get_error_function(domain, traction)

        assert get_convergance(Ns, get_error_function(f_trig, F_trig)) <= -3
        assert get_average(Ns, get_error_function(f_linear, F_linear)) < 1e-5
