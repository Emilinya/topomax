import os
import pickle

import numpy as np
import dolfin as df

from src.problem import Problem
from src.filter import HelmholtzFilter
from src.utils import constrain, save_function
from designs.design_parser import parse_design

df.set_log_level(df.LogLevel.WARNING)
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

    def __init__(
        self,
        N: int,
        design_file: str,
        problem: Problem,
        data_path: str = "data",
        skip_multiple: int = 1,
    ):
        self.N = N
        self.problem = problem
        self.data_path = data_path
        self.design_file = design_file
        self.skip_multiple = skip_multiple
        self.parameters, *extra_data = parse_design(self.design_file)

        # define domain
        self.width = self.parameters.width
        self.height = self.parameters.height

        volume_fraction = self.parameters.fraction
        self.volume = self.width * self.height * volume_fraction

        self.mesh = df.Mesh(
            df.RectangleMesh(
                df.MPI.comm_world,
                df.Point(0.0, 0.0),
                df.Point(self.width, self.height),
                int(self.width * self.N),
                int(self.height * self.N),
            )
        )

        self.control_space = df.FunctionSpace(self.mesh, "CG", 1)

        self.rho = df.Function(self.control_space)
        self.rho.vector()[:] = volume_fraction

        control_filter = HelmholtzFilter(epsilon=0.02)
        self.problem.init(control_filter, self.mesh, self.parameters, extra_data)

    def project(self, half_step, volume: float):
        """
        Project half_step so the volume constraint is fulfilled by
        solving '∫expit(half_step + c)dx = volume' for c using Newton's method,
        and then adding c to half_step.
        """

        expit_integral_func = df.Function(self.control_space)
        expit_diff_integral_func = df.Function(self.control_space)

        c = 0
        max_iterations = 10
        for _ in range(max_iterations):
            expit_integral_func.vector()[:] = expit(half_step + c)
            expit_diff_integral_func.vector()[:] = expit_diff(half_step + c)

            error = float(df.assemble(expit_integral_func * df.dx) - volume)
            derivative = float(df.assemble(expit_diff_integral_func * df.dx))
            if derivative == 0.0:
                raise ValueError("Got derivative equal to zero while projecting psi")

            newton_step = error / derivative
            c = c - newton_step
            if abs(newton_step) < 1e-12:
                break
        else:
            raise ValueError("Projection reached maximum iteration without converging.")

        return half_step + c

    def step(self, previous_psi, step_size):
        """Take a entropic mirror descent step with a given step size."""
        # Latent space gradient descent
        objective_gradient = self.problem.calculate_objective_gradient().vector()[:]

        half_step = previous_psi - step_size * objective_gradient
        return self.project(half_step, self.volume)

    def step_size(self, k: int) -> float:
        if self.parameters.problem == "elasticity":
            return 25 * (k + 1)
        else:
            return min(0.0015 * (k + 1), 0.015)

    def tolerance(self, k: int) -> float:
        itol = 1e-2
        ntol = 1e-5
        return min(25 * (k + 1) * ntol, itol)

    def solve(self):
        """Solve the given topology optimization problem."""

        psi = logit(self.rho.vector()[:])
        previous_psi = None

        difference = float("Infinity")
        objective = float(self.problem.calculate_objective(self.rho))
        objective_difference = None

        print("Iteration │ Objective │ ΔObjective │     Δρ    │ Tolerance ")
        print("──────────┼───────────┼────────────┼───────────┼───────────")

        def print_values(k, objective, objective_difference, difference):
            print(
                f"{k:^9} │ {constrain(objective, 9)} │ "
                + f"{constrain(objective_difference, 10)} │ "
                + f"{constrain(difference, 9)} │ "
                + f"{constrain(self.tolerance(k), 9)}",
                flush=True,
            )

        for k in range(100):
            print_values(k, objective, objective_difference, difference)
            if k % self.skip_multiple == 0:
                self.save_rho(self.rho, objective, k)

            previous_psi = psi.copy()
            try:
                psi = self.step(previous_psi, self.step_size(k))
            except ValueError as e:
                print_values(k + 1, objective, objective_difference, difference)
                print(f"EXIT: {e}")
                break

            self.rho.vector()[:] = expit(psi)
            previous_objective = objective
            objective = float(self.problem.calculate_objective(self.rho))
            objective_difference = previous_objective - objective

            if np.isnan(objective):
                print_values(k + 1, objective, objective_difference, difference)
                print("EXIT: Objective is NaN!")
                break

            # create dfa functions from previous_psi to calculate difference
            previous_rho = df.Function(self.control_space)
            previous_rho.vector()[:] = expit(previous_psi)

            difference = np.sqrt(df.assemble((self.rho - previous_rho) ** 2 * df.dx))

            if difference < self.tolerance(k):
                print_values(k + 1, objective, objective_difference, difference)
                print("EXIT: Optimal solution found")
                break
        else:
            print_values(k + 1, objective, objective_difference, difference)
            print("EXIT: Iteration did not converge")

        self.save_rho(self.rho, objective, k + 1)

    def save_rho(self, rho, objective, k):
        design = os.path.splitext(os.path.basename(self.design_file))[0]
        file_root = self.data_path + f"/{design}/data/N={self.N}_{k=}"
        os.makedirs(file_root, exist_ok=True)

        rho_file = file_root + "_rho.dat"
        save_function(rho, rho_file, "design")

        data = {"objective": objective, "iteration": k, "rho_file": rho_file}
        with open(file_root + ".dat", "wb") as datafile:
            pickle.dump(data, datafile)
