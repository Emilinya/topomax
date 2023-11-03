import numpy as np

from DEM_src.integrator import integrate
from DEM_src.data_structs import Domain


def linear(x, y):
    return x * y


def sine(x, y):
    return np.sin(x) * np.cos(y)


def linear_solution(domain: Domain):
    return domain.length**2 * domain.height**2 / 4


def sine_solution(domain: Domain):
    return np.sin(domain.height) * (1 - np.cos(domain.length))


def comp_results(X, Y, domain: Domain, function, solution):
    numeric = integrate(function(X, Y), domain)
    analytic = solution(domain)

    return abs(numeric - analytic) / abs(analytic)


def test_integrate():
    domain = Domain(5, 10, 5, 5)
    X, Y = np.meshgrid(domain.x_ray, domain.y_ray)

    # Integration with the trapezoidal rule should give the exact answer for linear functions
    assert comp_results(X, Y, domain, linear, linear_solution) == 0

    # For more complicated functions, we must see if the error is proportional to 1/N^2
    Ns = [1, 3, 6, 10, 15, 21]
    errors = []
    for N in Ns:
        domain = Domain(5 * N, 10 * N, 5, 5)
        X, Y = np.meshgrid(domain.x_ray, domain.y_ray)
        errors.append(comp_results(X, Y, domain, sine, sine_solution))

    poly = np.polynomial.Polynomial.fit(np.log(Ns), np.log(errors), 1)
    degree = poly.coef[1]

    assert degree <= -2
