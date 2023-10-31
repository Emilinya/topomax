import os
import pytest
import numpy as np
import dolfin as df

from src.df_utils import save_function, load_function


@pytest.fixture()
def cleanup():
    yield
    os.remove("tests/test_data/temp.dat")


def test_save_load(cleanup):
    N = 20
    mesh = df.Mesh(
        df.RectangleMesh(
            df.MPI.comm_world,
            df.Point(0.0, 0.0),
            df.Point(1.0, 1.0),
            N,
            N,
        )
    )
    velocity_space = df.VectorElement("CG", mesh.ufl_cell(), 2)
    pressure_space = df.FiniteElement("CG", mesh.ufl_cell(), 1)
    function_space = df.FunctionSpace(mesh, velocity_space * pressure_space)

    f = df.Function(function_space)

    np.random.seed(198)
    f.vector()[:] = np.random.random(len(f.vector()[:]))

    save_function(f, "tests/test_data/temp.dat", "fluid")

    assert os.path.isfile("tests/test_data/temp.dat")

    saved_f, *_ = load_function("tests/test_data/temp.dat")

    assert df.errornorm(f, saved_f, "L2", degree_rise=2) < 1e-14
