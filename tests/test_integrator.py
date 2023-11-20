import numpy as np

from DEM_src.integrator import integrate
from DEM_src.data_structs import Domain
from tests.utils import get_convergance


def linear(x, y):
    return x * y


def sine(x, y):
    return np.sin(x) * np.cos(y)


def linear_solution(domain: Domain):
    return domain.length**2 * domain.height**2 / 4


def sine_solution(domain: Domain):
    return np.sin(domain.height) * (1 - np.cos(domain.length))


def comp_results(domain: Domain, function, solution):
    numeric = integrate(function(domain.x_grid, domain.y_grid), domain)
    analytic = solution(domain)

    return abs(numeric - analytic) / abs(analytic)


def test_integrate():
    domain = Domain(5, 10, 5, 5)

    # Integration with the trapezoidal rule should give the exact answer for linear functions
    assert comp_results(domain, linear, linear_solution) == 0

    # For more complicated functions, we must see if the error decreases when N increases
    def error_func(N):
        domain = Domain(5 * N, 10 * N, 5, 5)
        return comp_results(domain, sine, sine_solution)

    Ns = [1, 3, 6, 10, 15, 21]
    assert get_convergance(Ns, error_func) <= -3
