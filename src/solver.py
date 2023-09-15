import os

import numpy as np
from scipy import io
import dolfin as df
import dolfin_adjoint as dfa

from designs.design_parser import parse_design
from src.filter import HelmholtzFilter
from src.problem import Problem
from src.utils import constrain

df.set_log_level(df.LogLevel.ERROR)
# turn off redundant output in parallel
df.parameters["std_out_all_processes"] = False


def expit(x):
    """Sigmoid function."""
    return 1.0 / (1.0 + np.exp(-x))


def expit_diff(x):
    """Derivative of the sigmoid function."""
    expit_val = expit(x)
    return expit_val * (1 - expit_val)


def logit(x):
    """Inverse sigmoid function."""
    return np.log(x / (1.0 - x))


class Solver:
    """Class that solves a given topology optimization problem using a magical algorithm."""

    def __init__(self, design_file: str, N: int, problem: Problem):
        self.problem = problem
        self.design_file = design_file
        self.parameters, *extra_data = parse_design(self.design_file)

        # define domain
        self.N = N
        self.width = self.parameters.width
        self.height = self.parameters.height

        volume_fraction = self.parameters.fraction
        self.volume = self.width * self.height * volume_fraction

        self.mesh = dfa.Mesh(
            dfa.RectangleMesh(
                df.MPI.comm_world,
                df.Point(0.0, 0.0),
                df.Point(self.width, self.height),
                int(self.width * self.N),
                int(self.height * self.N),
            )
        )

        control_filter = HelmholtzFilter(self.N)
        self.rho, self.objective_function = self.problem.init(
            control_filter, self.mesh, self.parameters, extra_data
        )

    def project(self, half_step, volume: float):
        """
        Project half_step so the volume constraint is fulfilled by
        solving '∫expit(half_step + c)dx = volume' for c using Newton's method,
        and then adding c to half_step.
        """

        expit_integral_func = dfa.Function(self.problem.control_space)
        expit_diff_integral_func = dfa.Function(self.problem.control_space)

        c = 0
        max_iterations = 10
        for _ in range(max_iterations):
            expit_integral_func.vector()[:] = expit(half_step + c)
            expit_diff_integral_func.vector()[:] = expit_diff(half_step + c)

            error = float(dfa.assemble(expit_integral_func * df.dx) - volume)
            derivative = float(dfa.assemble(expit_diff_integral_func * df.dx))
            if derivative == 0.0:
                print("Warning: Got derivative equal to zero during gradient descent.")
                raise ValueError("Can't project psi")

            newton_step = error / derivative
            c = c - newton_step
            if abs(newton_step) < 1e-12:
                break
        else:
            print(
                "Warning: Projection reached maximum iteration "
                + "without converging. Result may not be accurate."
            )

        return half_step + c

    def step(self, previous_psi, step_size):
        """Take a entropic mirror descent step with a given step size."""
        # Latent space gradient descent
        objective_gradient = self.objective_function.derivative()
        half_step = previous_psi - step_size * objective_gradient
        return self.project(half_step, self.volume)

    def step_size(self, k: int):
        return 25 * k

    def solve(self):
        """Solve the given topology optimization problem."""
        itol = 1e-2
        ntol = 1e-5

        psi = logit(self.rho.vector()[:])
        previous_psi = None

        error = float("Infinity")
        objective = float(self.objective_function(self.rho.vector()[:]))
        print("Iteration │ Objective │   Error  ")
        print("──────────┼───────────┼──────────")

        k = 0
        while error > min(self.step_size(k) * ntol, itol):
            print(f"{k:^9} │ {constrain(objective, 9)} │ {constrain(error, 9)}")
            if k % 10 == 0:
                self.save_rho(self.rho, objective, k)

            previous_psi = psi
            psi = self.step(previous_psi, self.step_size(k))
            k += 1

            self.rho.vector()[:] = expit(psi)
            objective = float(self.objective_function(self.rho.vector()[:]))

            # create dfa functions from psi and previous_psi to calculate error
            previous_psi_func = dfa.Function(self.problem.control_space)
            previous_psi_func.vector()[:] = previous_psi

            psi_func = dfa.Function(self.problem.control_space)
            psi_func.vector()[:] = psi

            error = np.sqrt(dfa.assemble((psi_func - previous_psi_func) ** 2 * df.dx))

        print(f"{k:^9} │ {constrain(objective, 9)} │ {constrain(error, 9)}")
        print("EXIT: Optimal solution found")
        self.save_rho(self.rho, objective, k)

    def save_rho(self, rho, objective, k):
        design = os.path.splitext(os.path.basename(self.design_file))[0]
        filename = f"output/{design}/data/N={self.N}_{k=}.mat"

        Nx, Ny = int(self.width * self.N), int(self.height * self.N)
        data = np.array(
            [
                [rho((0.5 + xi) / self.N, (0.5 + yi) / self.N) for xi in range(Nx)]
                for yi in range(Ny)
            ]
        )

        os.makedirs(os.path.dirname(filename), exist_ok=True)
        io.savemat(
            filename,
            mdict={
                "data": data,
                "objective": objective,
            },
        )
