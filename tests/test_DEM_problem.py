import os
import pickle

import torch
import numpy as np

from DEM_src.solver import Solver


def data_path(design):
    return os.path.join("tests", "test_data", "DEM", design, "problem_data.dat")


def compare_output(design):
    solver = Solver(os.path.join("designs", f"{design}.json"))

    x = torch.from_numpy(
        np.array([solver.domain.x_grid.T.flat, solver.domain.y_grid.T.flat]).T
    ).float()

    density = torch.from_numpy(solver.rho).float()
    density = torch.reshape(density, solver.domain.intervals)

    objective_calculator = solver.problem.dem.objective_calculator
    objective, gradient = objective_calculator.calculate_objective_and_gradient(
        x, solver.domain.shape, density
    )

    objective = float(objective)
    gradient = gradient.detach().numpy()

    saved_data = {}
    with open(data_path(design), "rb") as datafile:
        saved_data = pickle.load(datafile)

    assert abs(objective - saved_data["objective"]) < 1e-14
    assert np.all(np.abs(gradient - saved_data["gradient"]) < 1e-14)


def test_elasticity_problem():
    compare_output("short_cantilever")
    compare_output("bridge")
