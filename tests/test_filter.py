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
    rho = dfa.Function(control_space)
    k = len(rho.vector()[:])

    rho.vector()[:] = np.random.random(k)

    design_filter = HelmholtzFilter(10)
    design_filter.epsilon = 0
    filtered_rho = design_filter.apply(rho)

    assert abs(np.average(rho.vector()[:] - filtered_rho.vector()[:])) < 1e-14
