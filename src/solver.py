import os
import pickle
from typing import Any
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from src.problem import Problem
from src.utils import constrain, Timer
from src.printer import Printer, ColumnType
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

    def __init__(self, N: int, design_file: str, data_path="output", skip_multiple=1):
        self.N = N
        self.design_file = design_file
        self.design_str = os.path.splitext(os.path.basename(design_file))[0]
        self.output_folder = f"{data_path}/{self.get_name()}/{self.design_str}/data"
        self.skip_multiple = skip_multiple

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

        print_columns = [
            ColumnType.ITERATION,
            ColumnType.OBJECTIVE,
            ColumnType.DELTA_OBJECTIVE,
            ColumnType.DELRA_RHO,
            ColumnType.TOLERANCE,
            ColumnType.TIME,
            ColumnType.TOTAL_TIME,
        ]
        timer = Timer()

        psi = logit(self.to_array(self.rho))
        previous_psi = None

        for penalty in self.parameters.penalties:
            self.problem.set_penalization(penalty)

            printer = Printer(print_columns)
            title_length = printer.title_length()
            print(f"{f'Penalty: {constrain(penalty, 6)}':^{title_length - 6}}")
            printer.print_title()

            timer.restart()
            objectives = [self.problem.calculate_objective(self.rho)]
            printer.set_time(timer.get_time_seconds())
            printer.set_objective(objectives[0])

            k = 0
            for k in range(100):
                printer.set_tolerance(self.tolerance(k))
                printer.set_iteration(k)
                printer.print_values()

                if k % self.skip_multiple == 0:
                    self.save_data(self.rho, objectives[-1], k, penalty)

                timer.restart()
                previous_psi = psi.copy()
                try:
                    psi = self.step(previous_psi, self.step_size(k))
                except ValueError as e:
                    print(f"EXIT: {e}")
                    break

                self.set_from_array(self.rho, expit(psi))

                objectives.append(self.problem.calculate_objective(self.rho))
                printer.set_time(timer.get_time_seconds())
                printer.set_objective(objectives[-1])

                if np.isnan(objectives[-1]):
                    printer.set_iteration(k + 1)
                    printer.print_values()
                    print("EXIT: Objective is NaN!")
                    break

                difference = np.sqrt(
                    self.integrate((self.to_array(self.rho) - expit(previous_psi)) ** 2)
                )
                printer.set_delta_rho(difference)

                if difference < self.tolerance(k):
                    printer.set_iteration(k + 1)
                    printer.print_values()
                    print("EXIT: Optimal solution found")
                    break
            else:
                printer.print_values()
                print("EXIT: Iteration did not converge")

            self.save_data(self.rho, objectives[-1], k + 1, penalty)

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
