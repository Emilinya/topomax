import os
import sys
import copy
import warnings

import torch
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from hyperopt import fmin, tpe, hp, Trials
from sklearn.preprocessing import normalize

from Sub_Functions.data_structs import Domain, TopOptParameters, NNParameters
from Sub_Functions.DeepEnergyMethod import DeepEnergyMethod
from Sub_Functions.utils import Timer, smart_savefig
from Sub_Functions.MMA import optimize


def get_train_domain(domain: Domain, example: int):
    # create points
    lin_x = np.linspace(domain.x_min, domain.length, domain.Nx)
    lin_y = np.linspace(domain.y_min, domain.height, domain.Ny)
    domain_array = np.zeros((domain.Nx * domain.Ny, 2))

    # Assign nodal coordinates to all the points in the domain array
    for c, x in enumerate(np.nditer(lin_x)):
        tb = domain.Ny * c
        te = tb + domain.Ny
        domain_array[tb:te, 0] = x
        domain_array[tb:te, 1] = lin_y

    # create boundary conditions
    if example == 1:
        # downward load on the top of the domain
        len_load = 0.5
        bcr_t_pts_idx = np.where(
            (domain_array[:, 0] >= domain.length / 2.0 - len_load / 2.0)
            & (domain_array[:, 0] <= domain.length / 2.0 + len_load / 2.0)
            & (domain_array[:, 1] == domain.height)
        )
        bcr_t_pts = domain_array[bcr_t_pts_idx, :][0]
        bcr_t = np.ones(np.shape(bcr_t_pts)) * [0.0, -2000.0]

        boundary_neumann = {
            "neumann_1": {
                "coord": bcr_t_pts,
                "known_value": bcr_t,
                "penalty": 1.0,
                "idx": np.asarray(bcr_t_pts_idx),
            }
        }
    elif example == 2:
        # downward load on the right side of the domain
        _, dy = domain.dxdy
        bcr_t_pts_idx = np.where(
            (domain_array[:, 0] == domain.length)
            & (domain_array[:, 1] >= domain.height / 2.0 - dy / 2.0)
            & (domain_array[:, 1] <= domain.height / 2.0 + dy / 2.0)
        )
        bcr_t_pts = domain_array[bcr_t_pts_idx, :][0]
        bcr_t = np.ones(np.shape(bcr_t_pts)) * [0.0, -2000.0]

        boundary_neumann = {
            "neumann_1": {
                "coord": bcr_t_pts,
                "known_value": bcr_t,
                "penalty": 1.0,
                "idx": np.asarray(bcr_t_pts_idx),
            }
        }
    else:
        raise ValueError(f"Unknown example: {example}")

    return domain_array, boundary_neumann, {}


def get_test_datatest(domain: Domain):
    # create points
    x_space = np.linspace(domain.x_min, domain.length, domain.Nx)
    y_space = np.linspace(domain.y_min, domain.height, domain.Ny)
    xGrid, yGrid = np.meshgrid(x_space, y_space)
    data_test = np.concatenate(
        (np.array([xGrid.flatten()]).T, np.array([yGrid.flatten()]).T), axis=1
    )
    return x_space, y_space, data_test


def density_filter(radius: float, domain: Domain):
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
    test_domain: Domain,
    train_domain: Domain,
    example: int,
    dfilter,
    neumannBC,
    dirichletBC,
    domain_array,
    training_times: list[float],
    objective_values: list[float],
):
    x, y, _ = get_test_datatest(test_domain)

    def val_and_grad(density, iteration):
        rho_tilda = dfilter @ density

        dfdrho_tilda, compliance, training_time = dem.train_model(
            rho_tilda,
            domain_array,
            iteration,
            neumannBC,
            dirichletBC,
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
            f"./example{example}/design_{iteration}.png",
            dpi=200,
            bbox_inches="tight",
        )
        plt.close(fig)
        np.save(
            f"./example{example}/density_{iteration}.npy",
            density.reshape(ss),
        )

        # Save some data
        dem.evaluate_model(x, y, density.reshape(ss), iteration)

        compliance = compliance.cpu().detach().numpy()
        return compliance, sensitivity, training_time, timer.get_time_seconds()

    return val_and_grad


def volume_constraint_generator(volume_fraction):
    def volume_constraint(rho):
        return np.mean(rho) - volume_fraction, np.ones_like(rho) / len(rho)

    return volume_constraint


def hyperopt_main_generator(
    train_domain: Domain,
    example: int,
    device,
    to_parameters: TopOptParameters,
    datafile,
):
    def hyperopt_main(x_var: dict[str, int | float | str]):
        nn_parameters = NNParameters(
            input_size=2,
            output_size=2,
            layer_count=int(x_var["layer_count"]),
            neuron_count=int(x_var["neuron_count"]),
            learning_rate=x_var["learning_rate"],
            CNN_deviation=x_var["CNN_deviation"],
            rff_deviation=x_var["rff_deviation"],
            iteration_count=int(x_var["iteration_count"]),
            activation_function=x_var["activation_function"],
        )

        # STEP 1: SETUP DOMAIN - COLLECT CLEAN DATABASE
        dom, boundary_neumann, boundary_dirichlet = get_train_domain(
            train_domain, example
        )
        # x, y, _ = get_test_datatest(test_domain)

        density = (
            np.ones((train_domain.Ny - 1) * (train_domain.Nx - 1))
            * to_parameters.volume_fraction
        )

        # STEP 2: SETUP MODEL
        dem = DeepEnergyMethod(
            device, train_domain, example, to_parameters, nn_parameters
        )

        # STEP 3: TRAIN MODEL
        _, loss, _ = dem.train_model(
            density, dom, 0, boundary_neumann, boundary_dirichlet, [], []
        )
        print(f"{nn_parameters}\n  loss: {loss:.5e}")
        datafile.write(f"{nn_parameters} - loss: {loss:.5e}\n")

        # STEP 4: TEST MODEL
        # dem.evaluate_model(x, y, E, nu, layer_count, activation_function)

        return float(loss)

    return hyperopt_main


def main():
    warnings.filterwarnings("ignore")
    npr.seed(2022)
    torch.manual_seed(2022)
    np.random.seed(2022)

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        print("CUDA is available, running on GPU")
    else:
        device = torch.device("cpu")
        print("CUDA not available, running on CPU")

    example = 2

    if example == 1:
        train_domain = Domain(120 + 1, 30 + 1, 0, 0, 12, 2)
    elif example == 2:
        train_domain = Domain(90 + 1, 45 + 1, 0, 0, 10, 5)
    else:
        sys.exit("example must be set to 1 or 2")

    test_domain = copy.deepcopy(train_domain)

    optimization_parameters = TopOptParameters(
        E=2e5,
        nu=0.3,
        verbose=False,
        filter_radius=0.25,
        output_folder=f"./example{example}",
        max_iterations=80,
        volume_fraction=0.4,
        convergence_tolerances=5e-5 * np.ones(80),
    )

    # perform hyper parameter optimization or not
    optimize_hyperparameters = False
    if optimize_hyperparameters:
        datafile_path = "hyperopt_runs.txt"
        with open(datafile_path, "w") as datafile:
            hyperopt_main = hyperopt_main_generator(
                train_domain, example, device, optimization_parameters, datafile
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
                "got optimal hyperparameters, change neural_network_parameters "
                + "and restart with optimize_hyperparameters=False"
            )

    neural_network_parameters = NNParameters(
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

    dom, boundary_neumann, boundary_dirichlet = get_train_domain(train_domain, example)

    dem = DeepEnergyMethod(
        device,
        train_domain,
        example,
        optimization_parameters,
        neural_network_parameters,
    )

    with Timer("Generating density filter"):
        W = density_filter(optimization_parameters.filter_radius, train_domain)
    print()

    # Passive Density elements for topology opt.
    density = (
        np.ones((train_domain.Ny - 1) * (train_domain.Nx - 1))
        * optimization_parameters.volume_fraction
    )

    timer = Timer()
    training_times, objective_values = [], []
    optimizationParams = {
        "maxIters": optimization_parameters.max_iterations,
        "minIters": 2,
        "relTol": 0.0001,
    }

    val_and_grad = val_and_grad_generator(
        dem,
        test_domain,
        train_domain,
        example,
        W,
        boundary_neumann,
        boundary_dirichlet,
        dom,
        training_times,
        objective_values,
    )
    volume_constraint = volume_constraint_generator(
        optimization_parameters.volume_fraction
    )

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
    smart_savefig(
        optimization_parameters.output_folder + "/training_times.png", dpi=200
    )
    training_times.append(total_time)
    np.save(
        optimization_parameters.output_folder + "/training_times.npy", training_times
    )

    plt.figure()
    plt.plot(np.arange(len(objective_values)) + 1, objective_values)
    plt.xlabel("iteration")
    plt.ylabel("compliance [J]")
    smart_savefig(
        optimization_parameters.output_folder + "/objective_values.png", dpi=200
    )
    np.save(
        optimization_parameters.output_folder + "/objective_values.npy",
        objective_values,
    )
    plt.close()


if __name__ == "__main__":
    main()
