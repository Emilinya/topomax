import os
import pickle

import pytest
import dolfin as df

from src.solver import Solver
from src.utils import load_function


@pytest.fixture()
def cleanup():
    yield
    for filename in os.listdir("tests/test_data/diffuser/data"):
        if filename in ["correct_data.dat", "correct_rho.dat"]:
            continue
        os.remove(os.path.join("tests/test_data/diffuser/data", filename))


def test_fluid_solver(cleanup):
    solver = Solver(
        20, "designs/diffuser.json", data_path="tests/test_data", skip_multiple=999
    )
    solver.solve()

    data_path = "tests/test_data/diffuser/data"

    solver_out = os.path.join(data_path, "N=20_k=19.dat")
    assert os.path.isfile(solver_out)

    with open(solver_out, "rb") as datafile:
        solver_obj = pickle.load(datafile)
    solver_rho, mesh, function_space = load_function(solver_obj["rho_file"])
    solver_objective = solver_obj["objective"]

    with open(os.path.join(data_path, "correct_data.dat"), "rb") as datafile:
        correct_obj = pickle.load(datafile)
    correct_rho, *_ = load_function(
        os.path.join(data_path, "correct_rho.dat"), mesh, function_space
    )
    correct_objective = correct_obj["objective"]

    assert solver_objective == correct_objective
    assert df.assemble((solver_rho - correct_rho) ** 2 * df.dx) < 1e-14
