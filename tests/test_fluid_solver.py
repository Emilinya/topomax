import os
import pickle

import pytest
import dolfin as df

from FEM_src.solver import FEMSolver
from FEM_src.utils import load_function


@pytest.fixture()
def data_path():
    return os.path.join("tests", "test_data")


@pytest.fixture()
def output_folder(data_path):
    return os.path.join(data_path, "FEM", "diffuser", "data")


@pytest.fixture()
def cleanup(output_folder):
    yield
    for filename in os.listdir(output_folder):
        if filename in ["correct_data.dat", "correct_rho.dat"]:
            continue
        os.remove(os.path.join(output_folder, filename))


def test_fluid_solver(data_path, output_folder, cleanup):
    solver = FEMSolver(
        20, "designs/diffuser.json", data_path=data_path, skip_multiple=999
    )
    solver.solve()

    solver_data = os.path.join(output_folder, "N=20_p=0.1_k=19.dat")
    assert os.path.isfile(solver_data)

    solver_rho = os.path.join(output_folder, "N=20_p=0.1_k=19_rho.dat")
    assert os.path.isfile(solver_rho)

    with open(solver_data, "rb") as datafile:
        solver_obj = pickle.load(datafile)
    solver_rho, mesh, function_space = load_function(solver_rho)
    solver_objective = solver_obj["objective"]

    with open(os.path.join(output_folder, "correct_data.dat"), "rb") as datafile:
        correct_obj = pickle.load(datafile)
    correct_rho, *_ = load_function(
        os.path.join(output_folder, "correct_rho.dat"), mesh, function_space
    )
    correct_objective = correct_obj["objective"]

    assert abs(solver_objective - correct_objective) < 1e-14
    assert df.assemble((solver_rho - correct_rho) ** 2 * df.dx) < 1e-14
