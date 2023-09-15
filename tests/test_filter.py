import numpy as np
import dolfin as df
import dolfin_adjoint as dfa

from src.filter import HelmholtzFilter


def test_HelmholtzFilter():
    mesh = dfa.Mesh(
        dfa.RectangleMesh(
            df.MPI.comm_world,
            df.Point(0.0, 0.0),
            df.Point(1.0, 1.0),
            10,
            10,
        )
    )

    control_space = df.FunctionSpace(mesh, "DG", 0)
    design_filter = HelmholtzFilter(1)

    np.random.seed(198)
    rho = dfa.Function(control_space)
    rho.vector()[:] = np.random.random(len(rho.vector()[:]))

    # if epsilon is 0, filter should not do anything
    design_filter.epsilon = 0
    filtered_rho = design_filter.apply(rho)
    assert df.errornorm(rho, filtered_rho, "L2", degree_rise=2) < 1e-14

    # if rho = (8ε²π² + 1)cos(2π x)cos(2π y), then filtered_rho = cos(2π y)cos(2π x)
    design_filter.epsilon = np.random.random() * 5
    rho_expression = dfa.Expression(
        "(8*eps*eps*pi*pi + 1)*cos(2*pi*x[0])*cos(2*pi*x[1])",
        eps=design_filter.epsilon,
        degree=2,
    )
    filtered_rho_expression = dfa.Expression("cos(2*pi*x[0])*cos(2*pi*x[1])", degree=2)

    rho.interpolate(rho_expression)
    filtered_rho = design_filter.apply(rho)

    error = df.errornorm(
        filtered_rho, filtered_rho_expression, "L2", degree_rise=2, mesh=mesh
    )
    assert error < 1e-14
