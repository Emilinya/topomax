import numpy as np
import dolfin as df

from FEM_src.filter import HelmholtzFilter
from tests.utils import get_convergance


def initialize(N):
    mesh = df.Mesh(
        df.RectangleMesh(
            df.MPI.comm_world,
            df.Point(0.0, 0.0),
            df.Point(1.0, 1.0),
            N,
            N,
        )
    )

    control_space = df.FunctionSpace(mesh, "CG", 1)
    rho = df.Function(control_space)

    return mesh, rho


def test_HelmholtzFilter():
    _, rho = initialize(10)

    # if epsilon is 0, filter should not do anything
    design_filter = HelmholtzFilter(epsilon=0)

    np.random.seed(198)
    rho.vector()[:] = np.random.random(len(rho.vector()[:]))

    filtered_rho = design_filter.apply(rho)
    assert df.errornorm(rho, filtered_rho, "L2", degree_rise=2) < 1e-14

    # if rho = (8ε²π² + 1)cos(2π x)cos(2π y), then filtered_rho = cos(2π y)cos(2π x)
    design_filter.epsilon = np.e / np.pi
    rho_expression = df.Expression(
        "(8*eps*eps*pi*pi + 1)*cos(2*pi*x[0])*cos(2*pi*x[1])",
        eps=design_filter.epsilon,
        degree=2,
    )
    filtered_rho_expression = df.Expression("cos(2*pi*x[0])*cos(2*pi*x[1])", degree=2)

    # even if the filter works, the error will still be relatively large due to the smoothness
    # of the analytical solution. If the filter works, the error will be proporitonal to 1/N^2
    def error_func(N):
        mesh, rho = initialize(N)

        rho.interpolate(rho_expression)
        filtered_rho = design_filter.apply(rho)

        return df.errornorm(
            filtered_rho, filtered_rho_expression, "L2", degree_rise=2, mesh=mesh
        )

    Ns = list(range(10, 90 + 1, 20))
    assert get_convergance(Ns, error_func) <= -2
