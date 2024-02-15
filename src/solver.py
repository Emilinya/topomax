import os
import pickle
from typing import Any
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from scipy import optimize

from src.problem import Problem
from src.printer import Printer, ColumnType
from src.utils import (
    Timer,
    constrain,
    smart_brentq,
    SolverResult,
    IterationData,
    typeify_optimize,
)
from designs.definitions import FluidDesign, ElasticityDesign
from designs.design_parser import parse_design


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
    entropic mirror descent algorithm (EMD) by Brendan Keith and Thomas M. Surowiec

    This abstract base class contains the logic for the EMD algorithm,
    without making any assumptions about what object the design function
    is, or how the objective and objective gradient is calculated. To use
    this class, you must inherit from it and define all the abstract functions.
    """

    def __init__(self, N: int, design_file: str, data_path="output", skip_multiple=1):
        self.full_N = N
        self.design_file = design_file
        self.design_str = os.path.splitext(os.path.basename(design_file))[0]
        self.output_folder = f"{data_path}/{self.get_name()}/{self.design_str}/data"
        self.skip_multiple = skip_multiple

        self.parameters, design = parse_design(design_file)

        self.width = self.parameters.width
        self.height = self.parameters.height

        # we want N to be the number of elements in a unit length
        self.N = int(self.full_N / min(self.width, self.height))
        self.full_N = int(self.N * min(self.width, self.height))

        volume_fraction = self.parameters.volume_fraction
        self.volume = self.width * self.height * volume_fraction

        self.step_size = self.get_step_size()

        self.prepare_domain()
        self.rho = self.create_rho(volume_fraction)
        self.problem = self.create_problem(design)

        self.penalty_formatter = self.get_penalty_formatter(self.parameters.penalties)

    @abstractmethod
    def get_name(self) -> str:
        """
        Returns the name of the solver, which defines
        the folder the output data is saved in.
        """

    @abstractmethod
    def get_step_size(self) -> float:
        """
        Return self.parameters.fem_step_size or self.parameters.dem_step_size
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
    def create_problem(self, design: FluidDesign | ElasticityDesign) -> Problem: ...

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
    def save_rho(self, rho: Any, file_root: str) -> str:
        """
        Save the design function to a file, and return the basename.
        File root is equal to {output_folder}/{N=}_{p=}_{k=}
        """

    def get_penalty_formatter(self, penalties: list[float]):
        """
        To ensure that results are sorted correctly, we must format
        the penalties to always have the same number of characters.
        This is done by adding zeros to the start and end where neccesary.
        """

        padding_values = np.array(
            [[len(v) for v in str(p).split(".")] for p in penalties]
        )
        left_pad = np.max(padding_values[:, 0])
        right_pad = np.max(padding_values[:, 1])

        def penalty_formatter(penalty: float):
            left = len(str(penalty).split(".", maxsplit=1)[0])
            return f"{'0'*(left_pad-left)}{penalty:.{right_pad}f}"

        return penalty_formatter

    def project(self, half_step: npt.NDArray, volume: float):
        """
        Project half_step so the volume constraint is fulfilled
        by first solving '∫expit(half_step + c)dx = volume' for c
        using Newton's method, and then adding c to half_step.
        Newton's method might fail. In that case, Brent's method
        is used as a fallback.
        """

        def error(c: float):
            return self.integrate(expit(half_step + c)) - volume

        def error_derivative(c: float):
            return self.integrate(expit_diff(half_step + c))

        try:
            # First try Newton's method
            c, result = typeify_optimize(
                optimize.newton(
                    error,
                    0,
                    error_derivative,
                    tol=1e-12,
                    full_output=True,
                )
            )
            if result.converged:
                return half_step + c
        except RuntimeError:
            pass

        # Newton either did not converge or raised an exception.
        # Try with Brent's method instead
        c, result = smart_brentq(error, 2, 2000)
        if not result.converged:
            raise ValueError("Projection failed to converge")

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

    def step_size_at_iter(self, k: int):
        # this is a bit hacky, fix this?
        if len(self.parameters.penalties) > 1:
            return self.step_size * min((k + 1), 10)

        return self.step_size * (k + 1)

    def solve(self):
        """Solve the given topology optimization problem."""

        max_iterations = 500
        objective_increasing_factor = 2
        max_iterations_without_improvement = 50

        print_columns = [
            ColumnType.ITERATION,
            ColumnType.OBJECTIVE,
            ColumnType.DELTA_OBJECTIVE,
            ColumnType.DELRA_RHO,
            ColumnType.TOLERANCE,
            ColumnType.TIME,
            ColumnType.TOTAL_TIME,
        ]
        total_timer = Timer()
        objective_timer = Timer()

        psi = logit(self.to_array(self.rho))
        previous_psi = None

        for penalty in self.parameters.penalties:
            self.problem.set_penalization(penalty)

            printer = Printer(print_columns)
            title_length = printer.title_length()
            print(f"{f'Penalty: {constrain(penalty, 6)}':^{title_length - 6}}")
            printer.print_title()

            objective_timer.restart()
            objectives = [self.problem.calculate_objective(self.rho)]
            times = [objective_timer.get_time_seconds()]
            printer.set_objective(objectives[0])
            printer.set_time(times[0])
            printer.set_iteration(0)

            k = 0
            exit_condition = ""
            for k in range(max_iterations):
                printer.print_values()

                if k % self.skip_multiple == 0:
                    self.save_iteration(self.rho, objectives[-1], k, penalty)

                objective_timer.restart()
                previous_psi = psi.copy()
                try:
                    psi = self.step(previous_psi, self.step_size_at_iter(k))
                except ValueError as e:
                    exit_condition = str(e)
                    print(f"EXIT: {exit_condition}!")
                    break

                self.set_from_array(self.rho, expit(psi))

                objectives.append(self.problem.calculate_objective(self.rho))
                times.append(objective_timer.get_time_seconds())
                printer.set(self.tolerance(k), objectives[-1], k + 1, times[-1])

                if np.isnan(objectives[-1]):
                    exit_condition = "Objective is NaN"
                    printer.exit(exit_condition)
                    break

                min_index = int(np.argmin(objectives))
                min_objective = objectives[min_index]

                if objectives[-1] > objective_increasing_factor * min_objective:
                    exit_condition = "Objective is increasing"
                    printer.exit(exit_condition)
                    break

                if k >= min_index + max_iterations_without_improvement:
                    exit_condition = "Objective is not decreasing"
                    printer.exit(exit_condition)
                    break

                difference = np.sqrt(
                    self.integrate((self.to_array(self.rho) - expit(previous_psi)) ** 2)
                )
                printer.set_delta_rho(difference)

                if difference < self.tolerance(k):
                    exit_condition = "Convergence treshold reached"
                    printer.exit(exit_condition)
                    break
            else:
                exit_condition = "Iteration did not converge"
                printer.exit(exit_condition)

            self.save_iteration(self.rho, objectives[-1], k + 1, penalty)
            self.save_result(objectives, times, penalty, exit_condition)

        print(f"\nTopology optimization took {total_timer.get_time_string()}")

    def save_iteration(self, rho, objective: float, k: int, penalty: float):
        penalty_str = self.penalty_formatter(penalty)

        file_root = f"{self.output_folder}/N={self.full_N}_p={penalty_str}_{k=}"
        os.makedirs(os.path.dirname(file_root), exist_ok=True)

        rho_basename = self.save_rho(rho, file_root)

        data = IterationData(
            (self.width, self.height),
            objective,
            k,
            rho_basename,
            penalty,
        )
        with open(f"{file_root}.dat", "wb") as datafile:
            pickle.dump(data, datafile)

    def save_result(
        self,
        objectives: list[float],
        times: list[float],
        penalty: float,
        exit_condition: str,
    ):
        penalty_str = self.penalty_formatter(penalty)
        file_root = f"{self.output_folder}/N={self.full_N}_p={penalty_str}_result"

        min_index = int(np.argmin(objectives))
        if min_index != len(objectives) - 1:
            print(
                "\nWARNING: Final objective is not optimal. "
                + f"Lowest objective was achived at iteration {min_index}, "
                + f"with a value of {objectives[min_index]:.6g}."
            )

        data = SolverResult(
            exit_condition,
            objectives[min_index],
            objectives,
            len(objectives),
            min_index,
            times,
        )
        with open(f"{file_root}.dat", "wb") as datafile:
            pickle.dump(data, datafile)
