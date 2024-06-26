import os
import pickle

import torch
import numpy as np

from DEM_src.solver import DEMSolver


def data_path(design):
    return os.path.join("tests", "test_data", "DEM", design, "problem_data.dat")


def compare_output(design, N):
    solver = DEMSolver(N, os.path.join("designs", f"{design}.json"))

    x = torch.from_numpy(
        np.array([solver.mesh.x_grid.T.flat, solver.mesh.y_grid.T.flat]).T
    ).float()

    density = torch.from_numpy(solver.rho).float()
    density = torch.reshape(density, solver.mesh.intervals)

    solver.problem.set_penalization(solver.parameters.penalties[-1])
    objective_calculator = solver.problem.dem.objective_calculator
    objective, gradient = objective_calculator.calculate_objective_and_gradient(
        x, solver.mesh.shape, density
    )

    objective = float(objective)
    gradient = gradient.detach().numpy()

    saved_data = {}
    with open(data_path(design), "rb") as datafile:
        saved_data = pickle.load(datafile)

    assert abs(objective - saved_data["objective"]) < 1e-14
    assert np.all(np.abs(gradient - saved_data["gradient"]) < 1e-14)


def test_elasticity_problem():
    compare_output("short_cantilever", 45)
    compare_output("bridge", 30)
