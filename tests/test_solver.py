import os

import pytest
import numpy as np
from scipy import io

from src.solver import Solver
from src.elasisity_problem import ElasticityProblem


@pytest.fixture()
def cleanup():
    yield
    for filename in os.listdir("tests/test_data/triangle/data"):
        if filename == "correct_result.mat":
            continue
        os.remove(os.path.join("tests/test_data/triangle/data", filename))


def test_elasticity_problem(cleanup):
    problem = ElasticityProblem()
    solver = Solver("designs/triangle.json", 10, problem, "tests/test_data")
    solver.solve()

    solver_out = "tests/test_data/triangle/data/N=10_k=26.mat"
    assert os.path.isfile(solver_out)

    solver_mat = io.loadmat(solver_out)
    solver_data = solver_mat["data"]
    solver_objective = solver_mat["objective"][0, 0]

    correct_mat = io.loadmat("tests/test_data/triangle/data/correct_result.mat")
    correct_data = correct_mat["data"]
    correct_objective = correct_mat["objective"][0, 0]

    assert solver_objective == correct_objective
    assert np.average(np.abs(solver_data - correct_data)) < 1e-14
