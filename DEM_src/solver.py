import os
import sys
import pickle
import warnings

import torch
import numpy as np
import numpy.random as npr
from sklearn.preprocessing import normalize
from scipy.sparse import coo_matrix, csr_matrix

from src.utils import constrain
from DEM_src.data_structs import Domain
from DEM_src.integrator import integrate
from DEM_src.fluid_problem import FluidProblem
from DEM_src.elasisity_problem import ElasticityProblem
from designs.definitions import FluidDesign, ElasticityDesign
from designs.design_parser import parse_design


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


def create_density_filter(radius: float, domain: Domain) -> csr_matrix:
    # we can't use domain.x_grid as it has shape (Nx+1, Ny+1)
    x_ray = np.linspace(0, domain.length, domain.Nx)
    y_ray = np.linspace(0, domain.height, domain.Ny)
    x_grid, y_grid = np.meshgrid(x_ray, y_ray)
    X = x_grid.flatten()
    Y = y_grid.flatten()

    total = domain.Nx * domain.Ny

    wi, wj, wv = [], [], []
    for eid in range(total):
        my_X = X[eid]
        my_Y = Y[eid]

        dist = np.sqrt((X - my_X) ** 2 + (Y - my_Y) ** 2)
        neighbours = np.where(dist <= radius)[0]
        wi += [eid] * len(neighbours)
        wj += list(neighbours)
        wv += list(radius - dist[neighbours])

    W = normalize(
        coo_matrix((wv, (wi, wj)), shape=(total, total)), norm="l1", axis=1
    ).tocsr()  # Normalize row-wise

    return W


class Solver:
    """Class that solves a given topology optimization problem using a magical algorithm."""

    def __init__(self, design_file: str, data_path="output", verbose=False):
        warnings.filterwarnings("ignore")
        npr.seed(2022)
        torch.manual_seed(2022)
        np.random.seed(2022)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
            print("CUDA is available, running on GPU")
        else:
            self.device = torch.device("cpu")
            print("CUDA not available, running on CPU")

        self.design_file = design_file

        self.parameters, design = parse_design(design_file)

        # define domain
        self.width = self.parameters.width
        self.height = self.parameters.height

        self.design_str = os.path.splitext(os.path.split(design_file)[1])[0]
        if self.design_str == "bridge":
            Nx = 120
            Ny = 30
        elif self.design_str == "short_cantilever":
            Nx = 90
            Ny = 45
        else:
            N = 40
            Nx = int(self.width * N)
            Ny = int(self.height * N)

        self.domain = Domain(Nx, Ny, self.width, self.height)

        volume_fraction = self.parameters.volume_fraction
        self.volume = self.width * self.height * volume_fraction

        self.output_folder = f"{data_path}/DEM/{self.design_str}/data"

        self.rho = np.ones(self.domain.intervals) * volume_fraction

        if isinstance(design, FluidDesign):
            self.problem = FluidProblem(
                self.domain,
                self.device,
                verbose,
                design,
            )
        elif isinstance(design, ElasticityDesign):
            control_filter = create_density_filter(0.25, self.domain)
            self.problem = ElasticityProblem(
                self.domain,
                self.device,
                verbose,
                control_filter,
                design,
            )
        else:
            raise ValueError(
                f"Got unknown problem '{self.parameters.problem}' "
                + f"with design of type '{type(design)}'"
            )

    def project(self, half_step, volume: float):
        """
        Project half_step so the volume constraint is fulfilled by
        solving '∫expit(half_step + c)dx = volume' for c using Newton's method,
        and then adding c to half_step.
        """

        c = 0
        max_iterations = 10
        for _ in range(max_iterations):
            error = integrate(expit(half_step + c), self.domain) - volume
            derivative = integrate(expit_diff(half_step + c), self.domain)

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

    def step(self, previous_psi, step_size):
        """Take a entropic mirror descent step with a given step size."""
        # Latent space gradient descent
        objective_gradient = self.problem.calculate_objective_gradient()

        half_step = previous_psi - step_size * objective_gradient
        return self.project(half_step, self.volume)

    def step_size(self, k: int) -> float:
        return self.parameters.step_size * (k + 1)

    def tolerance(self, k: int) -> float:
        itol = 1e-2
        ntol = 1e-5
        return min(25 * (k + 1) * ntol, itol)

    def solve(self):
        """Solve the given topology optimization problem."""

        psi = logit(self.rho)
        previous_psi = None

        for penalty in self.parameters.penalties:
            difference = float("Infinity")
            objective = self.problem.calculate_objective(self.rho)
            objective_difference = None

            print(f"{f'Penalty: {constrain(penalty, 6)}':^59}")
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

            k = 0
            for k in range(100):
                print_values(k, objective, objective_difference, difference)
                self.save_rho(self.rho, objective, k, penalty)

                previous_psi = psi.copy()
                try:
                    psi = self.step(previous_psi, self.step_size(k))
                except ValueError as e:
                    print(f"EXIT: {e}")
                    break

                self.rho = expit(psi)
                previous_objective = objective
                objective = self.problem.calculate_objective(self.rho)
                objective_difference = previous_objective - objective

                if np.isnan(objective):
                    print_values(k + 1, objective, objective_difference, difference)
                    print("EXIT: Objective is NaN!")
                    break

                previous_rho = expit(previous_psi)
                difference = np.sqrt(
                    integrate((self.rho - previous_rho) ** 2, self.domain)
                )

                if difference < self.tolerance(k):
                    print_values(k + 1, objective, objective_difference, difference)
                    print("EXIT: Optimal solution found")
                    break
            else:
                print_values(k + 1, objective, objective_difference, difference)
                print("EXIT: Iteration did not converge")

            self.save_rho(self.rho, objective, k + 1, penalty)

    def save_rho(self, rho, objective: float, k: int, penalty: float):
        file_root = f"{self.output_folder}/{k=}"
        os.makedirs(os.path.dirname(file_root), exist_ok=True)

        np.save(
            f"{file_root}_rho.npy",
            rho.reshape(self.domain.intervals),
        )

        data = {
            "objective": objective,
            "iteration": k,
            "penalty": penalty,
            "domain_size": (self.domain.length, self.domain.height),
        }
        with open(f"{file_root}.dat", "wb") as datafile:
            pickle.dump(data, datafile)
