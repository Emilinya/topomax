import numpy as np

from DEM_src.utils import Mesh
from DEM_src.integrator import integrate
from tests.utils import get_convergance


def linear(x, y):
    return x * y


def sine(x, y):
    return np.sin(x) * np.cos(y)


def linear_solution(mesh: Mesh):
    return mesh.width**2 * mesh.height**2 / 4


def sine_solution(mesh: Mesh):
    return np.sin(mesh.height) * (1 - np.cos(mesh.width))


def comp_results(mesh: Mesh, function, solution):
    numeric = integrate(function(mesh.x_grid, mesh.y_grid), mesh)
    analytic = solution(mesh)

    return abs(numeric - analytic) / abs(analytic)


def test_integrate():
    mesh = Mesh(5, 10, 5, 5)

    # Integration with the trapezoidal rule should give the exact answer for linear functions
    assert comp_results(mesh, linear, linear_solution) == 0

    # For more complicated functions, we must see if the error decreases when N increases
    def error_func(N):
        mesh = Mesh(5 * N, 10 * N, 5, 5)
        return comp_results(mesh, sine, sine_solution)

    Ns = [1, 3, 6, 10, 15, 21]
    assert get_convergance(Ns, error_func) <= -3
