import io
import sys
import copy
import warnings

import torch
import numpy as np
import numpy.typing as npt
import numpy.random as npr
import matplotlib.pyplot as plt
from hyperopt import fmin, tpe, hp, Trials
from sklearn.preprocessing import normalize
from scipy.sparse import coo_matrix, csr_matrix

from DeepEnergy.src.MMA import optimize
from DeepEnergy.src.utils import Timer, smart_savefig
from DeepEnergy.src.DeepEnergyMethod import DeepEnergyMethod
from DeepEnergy.src.data_structs import Domain, TopOptParameters, NNParameters


def get_boundary_load(
    domain: Domain, side: str, center: float, length: float, value: list[float]
):
    if side == "left":
        side_condition = domain.coordinates[:, 0] == 0
        side_points = domain.coordinates[:, 1]
    elif side == "right":
        side_condition = domain.coordinates[:, 0] == domain.length
        side_points = domain.coordinates[:, 1]
    elif side == "top":
        side_condition = domain.coordinates[:, 1] == domain.height
        side_points = domain.coordinates[:, 0]
    elif side == "bottom":
        side_condition = domain.coordinates[:, 1] == 0
        side_points = domain.coordinates[:, 0]
    else:
        raise ValueError(f"Unknown side: '{side}'")

    left_condition = side_points >= center - length / 2.0
    right_condition = side_points <= center + length / 2.0
    load_idxs = np.where(side_condition & left_condition & right_condition)
    load_points = domain.coordinates[load_idxs, :][0]
    load_values = np.ones(np.shape(load_points)) * value

    return load_idxs, load_points, load_values


def get_boundary_conditions(domain: Domain, example: int):
    if example == 1:
        # downward load on the top of the domain
        load_idxs, load_points, load_values = get_boundary_load(
            domain,
            side="top",
            center=domain.length / 2,
            length=0.5,
            value=[0.0, -2000.0],
        )

        # fixed on left and right side
        def u0(x: torch.Tensor, y: torch.Tensor):
            return 0

        def m(x: torch.Tensor, y: torch.Tensor):
            return x * (1 - x)

    elif example == 2:
        # downward load on the right side of the domain
        _, dy = domain.dxdy
        load_idxs, load_points, load_values = get_boundary_load(
            domain,
            side="right",
            center=domain.height / 2,
            length=dy,
            value=[0.0, -2000.0],
        )

        # fixed on left side
        def u0(x: torch.Tensor, y: torch.Tensor):
            return 0

        def m(x: torch.Tensor, y: torch.Tensor):
            return x

    else:
        raise ValueError(f"Unknown example: {example}")

    neumannBC = {
        "neumann_1": {
            "coord": load_points,
            "known_value": load_values,
            "penalty": 1.0,
            "idx": np.asarray(load_idxs),
        }
    }

    dirichletBC = {"m": m, "u0": u0}

    return neumannBC, dirichletBC


def density_filter(radius: float, domain: Domain) -> csr_matrix:
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


def val_and_grad_generator(
    dem: DeepEnergyMethod,
    dfilter: csr_matrix,
    test_domain: Domain,
    train_domain: Domain,
    to_parameters: TopOptParameters,
    training_times: list[float],
    objective_values: list[float],
):
    def val_and_grad(density: npt.NDArray[np.float64], iteration: int):
        rho_tilda = dfilter @ density

        dfdrho_tilda, compliance, training_time = dem.train_model(
            rho_tilda,
            train_domain,
            iteration,
            training_times,
            objective_values,
        )
        dfdrho_tilda_npy = dfdrho_tilda.cpu().detach().numpy()

        # Invert filter
        ss = dfdrho_tilda_npy.shape
        sensitivity = dfilter.T @ dfdrho_tilda_npy.flatten()

        # Save plot
        timer = Timer()
        fig, axs = plt.subplots(2, 1, sharex="col")
        ff = axs[0].imshow(
            np.flipud(density.reshape(ss)),
            extent=train_domain.extent,
            cmap="binary",
        )
        plt.colorbar(ff, ax=axs[0])
        axs[0].set_aspect("equal")
        axs[0].set_title("Element density")
        ff = axs[1].imshow(
            np.flipud(sensitivity.reshape(ss)),
            extent=train_domain.extent,
            cmap="jet",
        )
        plt.colorbar(ff, ax=axs[1])
        axs[1].set_aspect("equal")
        axs[1].set_title("Sensitivity")
        smart_savefig(
            f"{to_parameters.output_folder}/design_{iteration}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)
        np.save(
            f"{to_parameters.output_folder}/density_{iteration}.npy",
            density.reshape(ss),
        )

        # Save some data
        dem.evaluate_model(test_domain, density.reshape(ss), iteration)

        compliance = compliance.cpu().detach().numpy()
        return compliance, sensitivity, training_time, timer.get_time_seconds()

    return val_and_grad


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
    def __init__(self, example: int):
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

        self.example = example

    def solve(self):
        if self.example == 1:
            train_domain = Domain(120 + 1, 30 + 1, 0, 0, 12, 2)
        elif self.example == 2:
            train_domain = Domain(90 + 1, 45 + 1, 0, 0, 10, 5)
        else:
            sys.exit("example must be set to 1 or 2")

        test_domain = copy.deepcopy(train_domain)

        to_parameters = TopOptParameters(
            E=2e5,
            nu=0.3,
            verbose=False,
            filter_radius=0.25,
            output_folder=f"DeepEnergy/example{self.example}",
            max_iterations=80,
            volume_fraction=0.4,
            convergence_tolerances=5e-5 * np.ones(80),
        )

        # perform hyper parameter optimization (this should be in a separate program ...)
        # optimize_hyperparameters(train_domain, example, device, to_parameters)

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

        neumannBC, dirichletBC = get_boundary_conditions(train_domain, self.example)

        dem = DeepEnergyMethod(
            self.device, neumannBC, dirichletBC, to_parameters, nn_parameters
        )

        with Timer("Generating density filter"):
            W = density_filter(to_parameters.filter_radius, train_domain)
        print()

        # initial density: constant density equal to the volume fraction
        density = (
            np.ones((train_domain.Ny - 1) * (train_domain.Nx - 1))
            * to_parameters.volume_fraction
        )

        timer = Timer()
        training_times, objective_values = [], []
        optimizationParams = {
            "maxIters": to_parameters.max_iterations,
            "minIters": 2,
            "relTol": 0.0001,
        }

        val_and_grad = val_and_grad_generator(
            dem,
            W,
            test_domain,
            train_domain,
            to_parameters,
            training_times,
            objective_values,
        )
        volume_constraint = volume_constraint_generator(to_parameters.volume_fraction)

        _, _, total_io_time = optimize(
            density,
            optimizationParams,
            val_and_grad,
            volume_constraint,
            train_domain.Ny - 1,
            train_domain.Nx - 1,
        )

        total_time = timer.get_time_seconds()
        print(
            f"\nTopology optimization took {Timer.prettify_seconds(total_time - total_io_time)} "
            + f"(IO took {Timer.prettify_seconds(total_io_time)})"
        )

        plt.figure()
        plt.plot(np.arange(len(training_times)) + 1, training_times)
        plt.xlabel("iteration")
        plt.ylabel("DEM training time [s]")
        plt.title(f"Total time = {total_time}s")
        smart_savefig(to_parameters.output_folder + "/training_times.png", dpi=200)
        training_times.append(total_time)
        np.save(to_parameters.output_folder + "/training_times.npy", training_times)

        plt.figure()
        plt.plot(np.arange(len(objective_values)) + 1, objective_values)
        plt.xlabel("iteration")
        plt.ylabel("compliance [J]")
        smart_savefig(to_parameters.output_folder + "/objective_values.png", dpi=200)
        np.save(
            to_parameters.output_folder + "/objective_values.npy",
            objective_values,
        )
        plt.close()
