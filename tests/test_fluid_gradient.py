import dolfin as df
import numpy as np

from src.solver import Solver
from src.penalizers import FluidPenalizer

def test_fluid_problem():
    solver = Solver(10, "designs/twin_pipe.json")

    objective = solver.problem.calculate_objective(solver.rho)
    direction = df.project(
        solver.volume / solver.problem.volume_fraction, solver.rho.function_space()
    )
    gradient = df.assemble(
        0.5 * FluidPenalizer.derivative(solver.rho) * direction * solver.problem.u**2 * df.dx
    )

    # we can't make t arbitrarily small as a small t results in numerical errors. Instead,
    # we use multiple t values and test if the error is converging to 0
    ts = [1e-3, 1e-7]
    errors= []
    for t in ts:
        moved_rho = df.project(solver.rho + t * direction, solver.rho.function_space())

        moved_objective = solver.problem.calculate_objective(moved_rho)
        almost_gradient = (moved_objective - objective) / t

        errors.append(abs(almost_gradient - gradient))

    poly = np.polynomial.Polynomial.fit(np.log(ts), np.log(errors), 1)
    degree = poly.coef[1]

    # this sure does converge fast
    assert degree >= 4
