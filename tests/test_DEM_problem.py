import os
import pickle

import numpy as np

from DEM_src.solver import Solver


def data_path(design):
    return os.path.join("tests", "test_data", "DEM", design, "problem_data.dat")


def compare_output(design):
    solver = Solver(os.path.join("designs", f"{design}.json"))

    objective = solver.problem.calculate_objective(solver.rho)
    gradient = solver.problem.calculate_objective_gradient()

    saved_data = {}
    with open(data_path(design), "rb") as datafile:
        saved_data = pickle.load(datafile)

    assert objective == saved_data["objective"]
    assert np.all(gradient == saved_data["gradient"])


def test_elasticity_problem():
    compare_output("short_cantilever")
    compare_output("bridge")
