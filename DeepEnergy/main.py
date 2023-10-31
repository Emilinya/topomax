import io
import os
import sys
import copy
import warnings

import torch
import numpy as np
import numpy.typing as npt
import numpy.random as npr
from hyperopt import fmin, tpe, hp, Trials
from sklearn.preprocessing import normalize
from scipy.sparse import coo_matrix, csr_matrix

from DeepEnergy.src.utils import Timer
from DeepEnergy.src.MMA import optimize
from designs.design_parser import parse_design
from DeepEnergy.src.DeepEnergyMethod import DeepEnergyMethod
from DeepEnergy.src.elasisity_problem import ElasticityProblem
from DeepEnergy.src.data_structs import Domain, TopOptParameters, NNParameters


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


def volume_constraint_generator(volume_fraction: float):
    def volume_constraint(rho: npt.NDArray[np.float64]):
        return np.mean(rho) - volume_fraction, np.ones_like(rho) / len(rho)

    return volume_constraint


def hyperopt_main_generator(
    device: torch.device,
    example: int,
    datafile: io.TextIOWrapper,
    train_domain: Domain,
    to_parameters: TopOptParameters,
):
    def hyperopt_main(x_var: dict[str, int | float | str]):
        nn_parameters = NNParameters(
            input_size=2,
            output_size=2,
            layer_count=int(x_var["layer_count"]),
            neuron_count=int(x_var["neuron_count"]),
            learning_rate=float(x_var["learning_rate"]),
            CNN_deviation=float(x_var["CNN_deviation"]),
            rff_deviation=float(x_var["rff_deviation"]),
            iteration_count=int(x_var["iteration_count"]),
            activation_function=float(x_var["activation_function"]),
        )

        # STEP 1: SETUP DOMAIN - COLLECT CLEAN DATABASE
        neumannBC, dirichletBC = get_boundary_conditions(train_domain, example)

        density = (
            np.ones((train_domain.Ny - 1) * (train_domain.Nx - 1))
            * to_parameters.volume_fraction
        )

        # STEP 2: SETUP MODEL
        dem = DeepEnergyMethod(
            device, neumannBC, dirichletBC, to_parameters, nn_parameters
        )

        # STEP 3: TRAIN MODEL
        _, loss, _ = dem.train_model(density, train_domain, 0, [], [])
        print(f"{nn_parameters}\n  loss: {loss:.5e}")
        datafile.write(f"{nn_parameters} - loss: {loss:.5e}\n")

        # STEP 4: TEST MODEL
        # dem.evaluate_model(x, y, E, nu, layer_count, activation_function)

        return float(loss)

    return hyperopt_main


def optimize_hyperparameters(
    domain: Domain, example: int, device: torch.device, to_parameters: TopOptParameters
):
    datafile_path = "hyperopt_runs.txt"
    with open(datafile_path, "w") as datafile:
        hyperopt_main = hyperopt_main_generator(
            device, example, datafile, domain, to_parameters
        )

        space = {
            "layer_count": hp.quniform("layer_count", 3, 5, 1),
            "neuron_count": 2 * hp.quniform("neuron_count", 10, 60, 1),
            "learning_rate": hp.loguniform("learning_rate", 0, 2),
            "CNN_deviation": hp.uniform("CNN_deviation", 0, 1),
            "rff_deviation": hp.uniform("rff_deviation", 0, 1),
            "iteration_count": hp.quniform("iteration_count", 40, 100, 1),
            "activation_function": hp.choice(
                "activation_function", ["tanh", "relu", "rrelu", "sigmoid"]
            ),
        }

        best = fmin(
            hyperopt_main,
            space,
            algo=tpe.suggest,
            max_evals=100,
            trials=Trials(),
            rstate=np.random.default_rng(2019),
            max_queue_len=2,
        )
        print(best)
        datafile.write(f"\n--- Optimal parameters ---\n{best}")
        sys.exit(
            "got optimal hyperparameters, change nn_parameters "
            + "and restart with optimize_hyperparameters=False"
        )


class DeepEnergySolver:
    def __init__(self, design_file: str):
        domain_parameters, elasticity_design = parse_design(design_file)

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

        design = os.path.splitext(os.path.split(design_file)[1])[0]

        if design == "bridge":
            self.train_domain = Domain(120 + 1, 30 + 1, 0, 0, 12, 2)
        elif design == "short_cantilever":
            self.train_domain = Domain(90 + 1, 45 + 1, 0, 0, 10, 5)
        else:
            sys.exit("example must be bridge or short_cantilever")

        self.test_domain = copy.deepcopy(self.train_domain)

        self.to_parameters = TopOptParameters(
            E=2e5,
            nu=0.3,
            verbose=False,
            filter_radius=0.25,
            output_folder=f"DeepEnergy/{design}",
            max_iterations=10,
            volume_fraction=domain_parameters.volume_fraction,
            convergence_tolerances=5e-5 * np.ones(80),
        )

        nn_parameters = NNParameters(
            input_size=2,
            output_size=2,
            layer_count=5,
            neuron_count=68,
            learning_rate=1.73553,
            CNN_deviation=0.062264,
            rff_deviation=0.119297,
            iteration_count=100,
            activation_function="rrelu",
        )

        with Timer("Generating density filter"):
            density_filter = create_density_filter(
                self.to_parameters.filter_radius, self.train_domain
            )
        print()

        self.problem = ElasticityProblem(
            self.device,
            elasticity_design,
            self.test_domain,
            self.train_domain,
            density_filter,
            self.to_parameters,
            nn_parameters,
        )

    def solve(self):
        # initial density: constant density equal to the volume fraction
        density = (
            np.ones((self.train_domain.Ny - 1) * (self.train_domain.Nx - 1))
            * self.to_parameters.volume_fraction
        )

        timer = Timer()
        optimizationParams = {
            "maxIters": self.to_parameters.max_iterations,
            "minIters": 2,
            "relTol": 0.0001,
        }

        def val_and_grad(rho, iteration):
            timer = Timer()
            objective = self.problem.calculate_objective(rho)
            objective_gradient = self.problem.calculate_objective_gradient()
            training_time = timer.get_time_seconds()

            Nx, Ny = self.problem.train_domain.shape
            timer.restart()
            np.save(
                f"{self.to_parameters.output_folder}/density_{iteration}.npy",
                density.reshape((Nx - 1, Ny - 1)),
            )
            io_time = timer.get_time_seconds()

            return objective, objective_gradient, training_time, io_time

        volume_constraint = volume_constraint_generator(
            self.to_parameters.volume_fraction
        )

        os.makedirs(self.to_parameters.output_folder, exist_ok=True)
        _, _, total_io_time = optimize(
            density,
            optimizationParams,
            val_and_grad,
            volume_constraint,
            self.train_domain.Ny - 1,
            self.train_domain.Nx - 1,
        )

        total_time = timer.get_time_seconds()
        print(
            f"\nTopology optimization took {Timer.prettify_seconds(total_time - total_io_time)} "
            + f"(IO took {Timer.prettify_seconds(total_io_time)})"
        )
