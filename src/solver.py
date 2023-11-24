import os
import pickle
from typing import Any
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from src.utils import constrain
from src.problem import Problem
from designs.design_parser import parse_design
from designs.definitions import FluidDesign, ElasticityDesign


def expit(x: npt.NDArray):
    """Sigmoid function."""
    return 1.0 / (1.0 + np.exp(-x))


def expit_diff(x: npt.NDArray):
    """Derivative of the sigmoid function."""
    expit_val = expit(x)
    return expit_val * (1 - expit_val)


def logit(x: npt.NDArray):
    """Inverse sigmoid function."""
    return np.log(x / (1.0 - x))


class Solver(ABC):
    """
    Class that solves a given topology optimization problem using the
    entropic mirror descent algorithm by Brendan Keith and Thomas M. Surowiec

    This abstract base class contains the logic for the EMD algorithm,
    without making any assumptions about what object the design function
    is, or how the objective and objective gradient is calculated. To use
    this class, you must inherit from it and define all the abstract functions.
    """

    def __init__(self, N: int, design_file: str, data_path="output"):
        self.N = N
        self.design_file = design_file
        self.design_str = os.path.splitext(os.path.basename(design_file))[0]
        self.output_folder = f"{data_path}/{self.get_name()}/{self.design_str}/data"

        self.parameters, design = parse_design(design_file)

        self.width = self.parameters.width
        self.height = self.parameters.height

        volume_fraction = self.parameters.volume_fraction
        self.volume = self.width * self.height * volume_fraction

        self.prepare_domain()
        self.rho = self.create_rho(volume_fraction)
        self.problem = self.create_problem(design)

        self.step_size_multiplier = 1

    @abstractmethod
    def get_name(self) -> str:
        """
        Returns the name of the solver, which defines
        the folder the output data is saved in.
        """

    @abstractmethod
    def prepare_domain(self) -> None:
        """
        Create all the objects you need when creating rho and the problem.
        When this function is called, you have acces to the domain parameters.
        """

    @abstractmethod
    def create_rho(self, volume_fraction: float) -> Any:
        """Create and return the design function."""

    @abstractmethod
    def create_problem(self, design: FluidDesign | ElasticityDesign) -> Problem:
        ...

    @abstractmethod
    def to_array(self, rho: Any) -> npt.NDArray:
        """Convert the design function into a numpy array."""

    @abstractmethod
    def set_from_array(self, rho: Any, values: npt.NDArray) -> None:
        """
        Update the design function by setting it's value from a numpy array.
        """

    @abstractmethod
    def integrate(self, values: npt.NDArray) -> float:
        """Integrate the given values over the domain."""

    @abstractmethod
    def save_rho(self, rho: Any, file_root: str):
        """
        Save the design function to a file. File root
        is equal to {output_folder}/{N=}_{p=}_{k=}
        """

    def project(self, half_step: npt.NDArray, volume: float):
        """
        Project half_step so the volume constraint is fulfilled
        by first solving '∫expit(half_step + c)dx = volume' for c
        using Newton's method, and then adding c to half_step.
        """

        c = 0
        max_iterations = 10
        for _ in range(max_iterations):
            error = self.integrate(expit(half_step + c)) - volume
            derivative = self.integrate(expit_diff(half_step + c))

            if derivative == 0.0:
                raise ValueError(
                    "Got derivative equal to zero while projecting psi."
                    + "Your step size is probably too high."
                )

            newton_step = error / derivative
            c = c - newton_step
            if abs(newton_step) < 1e-12:
                break
        else:
            raise ValueError("Projection reached maximum iteration without converging.")

        return half_step + c

    def step(self, previous_psi: npt.NDArray, step_size: float):
        """Take a entropic mirror descent step with a given step size."""
        # Latent space gradient descent
        objective_gradient = self.to_array(self.problem.calculate_objective_gradient())
        half_step = previous_psi - step_size * objective_gradient
        # Volume correction
        return self.project(half_step, self.volume)

    def tolerance(self, k: int):
        itol = 1e-2
        ntol = 1e-5
        return min(25 * (k + 1) * ntol, itol)

    def step_size(self, k: int):
        return self.parameters.step_size * (k + 1) * self.step_size_multiplier

    def solve(self):
        """Solve the given topology optimization problem."""

        psi = logit(self.to_array(self.rho))
        previous_psi = None

        def print_values(k, objective, objective_difference, difference):
            print(
                f"{k:^9} │ {constrain(objective, 9)} │ "
                + f"{constrain(objective_difference, 10)} │ "
                + f"{constrain(difference, 9)} │ "
                + f"{constrain(self.tolerance(k), 9)}",
                flush=True,
            )

        def abort(reason: str, k: int):
            print_values(k + 1, objective, objective_difference, difference)
            print(f"EXIT: {reason}")

        for penalty in self.parameters.penalties:
            self.problem.set_penalization(penalty)

            print(f"{f'Penalty: {constrain(penalty, 6)}':^59}")
            print("Iteration │ Objective │ ΔObjective │     Δρ    │ Tolerance ")
            print("──────────┼───────────┼────────────┼───────────┼───────────")

            objective = self.problem.calculate_objective(self.rho)
            difference = float("Infinity")
            objective_difference = float("Infinity")

            k = 0
            for k in range(100):
                print_values(k, objective, objective_difference, difference)
                self.save_data(self.rho, objective, k, penalty)

                previous_psi = psi.copy()
                try:
                    psi = self.step(previous_psi, self.step_size(k))
                except ValueError as e:
                    print(f"EXIT: {e}")
                    break

                self.set_from_array(self.rho, expit(psi))

                previous_objective = objective
                objective = self.problem.calculate_objective(self.rho)
                objective_difference = previous_objective - objective

                if np.isnan(objective):
                    abort("Objective is NaN!", k + 1)
                    break

                difference = np.sqrt(
                    self.integrate((self.to_array(self.rho) - expit(previous_psi)) ** 2)
                )

                if difference < self.tolerance(k):
                    abort("Optimal solution found", k + 1)
                    break
            else:
                abort("Iteration did not converge", k + 1)

            self.save_data(self.rho, objective, k + 1, penalty)

    def save_data(self, rho, objective: float, k: int, penalty: float):
        file_root = f"{self.output_folder}/N={self.N}_p={penalty}_{k=}"
        os.makedirs(os.path.dirname(file_root), exist_ok=True)

        self.save_rho(rho, file_root)

        data = {
            "objective": objective,
            "iteration": k,
            "penalty": penalty,
            "domain_size": (self.width, self.height),
        }
        with open(f"{file_root}.dat", "wb") as datafile:
            pickle.dump(data, datafile)
