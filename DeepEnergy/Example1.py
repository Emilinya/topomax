import os
import sys
import copy
import time
import warnings

import torch
import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from hyperopt import fmin, tpe, hp, Trials
from sklearn.preprocessing import normalize

from Sub_Functions.DeepEnergyMethod import DeepEnergyMethod
from Sub_Functions.data_structs import Domain, TopOptParameters, NNParameters
from Sub_Functions import Utility as util
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

        dfdrho_tilda, compliance = dem.train_model(
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
        start_io = time.time()
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
        util.smart_savefig(
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

        return compliance.cpu().detach().numpy(), sensitivity, time.time() - start_io

    return val_and_grad


def volume_constraint_generator(volume_fraction):
    def volume_constraint(rho):
        return np.mean(rho) - volume_fraction, np.ones_like(rho) / len(rho)

    return volume_constraint


def hyperopt_main(x_var):
    lr = x_var["x_lr"]
    neuron = int(x_var["neuron"])
    CNN_dev = x_var["CNN_dev"]
    rff_dev = x_var["rff_dev"]
    iteration = int(x_var["No_iteration"])
    N_layers = int(x_var["N_layers"])
    act_func = x_var["act_func"]

    # ----------------------------------------------------------------------
    #                   STEP 1: SETUP DOMAIN - COLLECT CLEAN DATABASE
    # ----------------------------------------------------------------------
    dom, boundary_neumann, boundary_dirichlet = get_train_domain(train_domain, example)
    x, y, _ = get_test_datatest(test_domain)

    # --- Activate for circular inclusion-----
    # density= get_density()
    density = 1
    # ----------------------------------------------------------------------
    #                   STEP 2: SETUP MODEL
    # ----------------------------------------------------------------------

    dem = DeepEnergyMethod(device, train_domain, example, to_parameters, nn_parameters)

    # ----------------------------------------------------------------------
    #                   STEP 3: TRAINING MODEL
    # ----------------------------------------------------------------------
    Loss = dem.train_model(
        density,
        dom,
        shape,
        dxdy,
        boundary_neumann,
        boundary_dirichlet,
        0,
    )
    print(
        "lr: %.5e,\t neuron: %.3d, \t CNN_Sdev: %.5e, \t RNN_Sdev: %.5e, \t Itertions: %.3d, \t Layers: %d, \t Act_fn : %s,\t Loss: %.5e"
        % (lr, neuron, CNN_dev, rff_dev, iteration, N_layers, act_func, Loss)
    )

    f.write(
        "lr: %.5e,\t neuron: %.3d, \t CNN_Sdev: %.5e, \t RNN_Sdev: %.5e, \t Itertions: %.3d, \t Layers: %d, \t Act_fn : %s,\t Loss: %.5e"
        % (lr, neuron, CNN_dev, rff_dev, iteration, N_layers, act_func, Loss)
    )

    # ----------------------------------------------------------------------
    #                   STEP 4: TEST MODEL
    # ----------------------------------------------------------------------

    # dem.evaluate_model(x, y, E, nu, N_layers, act_func)

    return Loss


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

    # ------------------------- Perform hyper parameter optimization or not -----------------
    if False:
        # -------------------------- File to write results in ---------
        if os.path.exists("HOpt_Runs.txt"):
            os.remove("HOpt_Runs.txt")

        f = open("HOpt_Runs.txt", "a")

        # -------------------------- Variable HyperParameters-----------------------------
        space = {
            "x_lr": hp.loguniform("x_lr", 0, 2),
            "neuron": 2 * hp.quniform("neuron", 10, 60, 1),
            "act_func": hp.choice("act_func", ["tanh", "relu", "rrelu", "sigmoid"]),
            "CNN_dev": hp.uniform("CNN_dev", 0, 1),
            "rff_dev": hp.uniform("rff_dev", 0, 1),
            "No_iteration": hp.quniform("No_iteration", 40, 100, 1),
            "N_layers": hp.quniform("N_layers", 3, 5, 1),
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
        exit()

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

    # ----------------------------------------------------------------------
    #                   STEP 1: SETUP DOMAIN - COLLECT CLEAN DATABASE
    # ----------------------------------------------------------------------
    dom, boundary_neumann, boundary_dirichlet = get_train_domain(train_domain, example)

    # ----------------------------------------------------------------------
    #                   STEP 2: SETUP MODEL
    # ----------------------------------------------------------------------
    dem = DeepEnergyMethod(
        device,
        train_domain,
        example,
        optimization_parameters,
        neural_network_parameters,
    )

    # TO using MMA
    start_t = time.time()
    W = density_filter(optimization_parameters.filter_radius, train_domain)
    end_t = time.time()
    print("Generating density filter took " + str(end_t - start_t) + " s")

    # Passive Density elements for topology opt.
    density = (
        np.ones((train_domain.Ny - 1) * (train_domain.Nx - 1))
        * optimization_parameters.volume_fraction
    )

    start_t = time.time()
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
    end_t = time.time()
    t_tot = end_t - start_t - total_io_time
    print(f"Topology optimization took {t_tot:.3f} s ({total_io_time:3f} s io time)")

    plt.figure()
    plt.plot(np.arange(len(training_times)) + 1, training_times)
    plt.xlabel("iteration")
    plt.ylabel("DEM training time [s]")
    plt.title("Total time = " + str(t_tot) + "s")
    util.smart_savefig(
        optimization_parameters.output_folder + "/training_times.png", dpi=600
    )
    training_times.append(t_tot)
    np.save(
        optimization_parameters.output_folder + "/training_times.npy", training_times
    )

    plt.figure()
    plt.plot(np.arange(len(objective_values)) + 1, objective_values)
    plt.xlabel("iteration")
    plt.ylabel("compliance [J]")
    util.smart_savefig(
        optimization_parameters.output_folder + "/objective_values.png", dpi=600
    )
    np.save(
        optimization_parameters.output_folder + "/objective_values.npy",
        objective_values,
    )
    plt.close()


if __name__ == "__main__":
    main()
