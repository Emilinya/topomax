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
from designs.design_parser import parse_design
from designs.definitions import FluidDesign, ElasticityDesign, ProblemType
from DeepEnergy.src.integrator import integrate
from DeepEnergy.src.data_structs import Domain, NNParameters
from DeepEnergy.src.elasisity_problem import ElasticityProblem


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
    nex = domain.Nx - 1
    ney = domain.Ny - 1
    Lx = domain.length
    Ly = domain.height

    xx = np.linspace(0, Lx, nex)
    yy = np.linspace(0, Ly, ney)
    X, Y = np.meshgrid(xx, yy)
    X = X.flatten()
    Y = Y.flatten()

    wi, wj, wv = [], [], []
    for eid in range(nex * ney):
        my_X = X[eid]
        my_Y = Y[eid]

        dist = np.sqrt((X - my_X) ** 2 + (Y - my_Y) ** 2)
        neighbours = np.where(dist <= radius)[0]
        wi += [eid] * len(neighbours)
        wj += list(neighbours)
        wv += list(radius - dist[neighbours])

    W = normalize(
        coo_matrix((wv, (wi, wj)), shape=(nex * ney, nex * ney)), norm="l1", axis=1
    ).tocsr()  # Normalize row-wise

    return W


class Solver:
    """Class that solves a given topology optimization problem using a magical algorithm."""

    def __init__(self, design_file: str):
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

        self.design_str = os.path.splitext(os.path.split(design_file)[1])[0]
        if self.design_str == "bridge":
            self.width = 12
            self.height = 2
            self.domain = Domain(120 + 1, 30 + 1, 0, 0, 12, 2)
        elif self.design_str == "short_cantilever":
            self.width = 10
            self.height = 5
            self.domain = Domain(90 + 1, 45 + 1, 0, 0, 10, 5)
        else:
            sys.exit("example must be bridge or short_cantilever")

        self.reduced_domain = Domain(
            self.domain.Nx - 1,
            self.domain.Ny - 1,
            self.domain.x_min,
            self.domain.y_min,
            self.domain.length,
            self.domain.height,
        )

        volume_fraction = self.parameters.volume_fraction
        self.volume = self.width * self.height * volume_fraction

        self.output_folder = f"DeepEnergy/output/{self.design_str}/data"

        nn_parameters = NNParameters(
            verbose=False,
            input_size=2,
            output_size=2,
            layer_count=5,
            neuron_count=68,
            learning_rate=1.73553,
            CNN_deviation=0.062264,
            rff_deviation=0.119297,
            iteration_count=100,
            activation_function="rrelu",
            convergence_tolerance=5e-5,
        )

        self.rho = np.ones(self.reduced_domain.shape) * volume_fraction

        control_filter = create_density_filter(0.25, self.domain)

        if isinstance(design, FluidDesign):
            sys.exit(1)
        elif isinstance(design, ElasticityDesign):
            self.problem = ElasticityProblem(
                2e5,
                0.3,
                self.domain,
                self.device,
                control_filter,
                nn_parameters,
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
            error = integrate(expit(half_step + c), self.reduced_domain) - volume
            derivative = integrate(expit_diff(half_step + c), self.reduced_domain)

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
        if self.parameters.problem == ProblemType.ELASTICITY:
            if self.design_str == "bridge":
                multiplier = 1 / 50
            else:
                multiplier = 2
            return 25 * (k + 1) * multiplier
        if self.parameters.problem == ProblemType.FLUID:
            return min(0.0015 * (k + 1), 0.015)
        raise ValueError(f"Unknown problem: {self.parameters.problem}")

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
            objective = float(self.problem.calculate_objective(self.rho))
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
                objective = float(self.problem.calculate_objective(self.rho))
                objective_difference = previous_objective - objective

                if np.isnan(objective):
                    print_values(k + 1, objective, objective_difference, difference)
                    print("EXIT: Objective is NaN!")
                    break

                previous_rho = expit(previous_psi)
                difference = np.sqrt(
                    integrate((self.rho - previous_rho) ** 2, self.reduced_domain)
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
            rho.reshape(self.reduced_domain.shape),
        )

        data = {
            "objective": objective,
            "iteration": k,
            "penalty": penalty,
            "domain_size": (self.domain.length, self.domain.height),
        }
        with open(f"{file_root}.dat", "wb") as datafile:
            pickle.dump(data, datafile)
