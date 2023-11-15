import io
import os

import torch
import numpy as np
from hyperopt import fmin, tpe, hp, Trials

from DEM_src.solver import Solver
from DEM_src.data_structs import Domain
from DEM_src.bc_helpers import get_boundary_conditions
from DEM_src.DeepEnergyMethod import NNParameters, DeepEnergyMethod
from designs.definitions import ElasticityDesign


def hyperopt_main_generator(
    E: float,
    nu: float,
    density,
    domain: Domain,
    device: torch.device,
    datafile: io.TextIOWrapper,
    elasticity_design: ElasticityDesign,
):
    extended_domain = Domain(domain.Nx + 1, domain.Ny + 1, domain.length, domain.height)

    def hyperopt_main(x_var: dict[str, int | float | str]):
        nn_parameters = NNParameters(
            verbose=False,
            input_size=2,
            output_size=2,
            layer_count=int(x_var["layer_count"]),
            neuron_count=int(x_var["neuron_count"]),
            learning_rate=float(x_var["learning_rate"]),
            CNN_deviation=float(x_var["CNN_deviation"]),
            rff_deviation=float(x_var["rff_deviation"]),
            iteration_count=int(x_var["iteration_count"]),
            activation_function=str(x_var["activation_function"]),
            convergence_tolerance=5e-5,
        )

        # STEP 1: SETUP DOMAIN - COLLECT CLEAN DATABASE
        traction_points_list, dirichlet_enforcer = get_boundary_conditions(
            extended_domain, elasticity_design
        )

        # STEP 2: SETUP MODEL
        dem = DeepEnergyMethod(
            E, nu, device, nn_parameters, dirichlet_enforcer, traction_points_list
        )

        # STEP 3: TRAIN MODEL
        loss, _ = dem.train_model(density, extended_domain)
        datafile.write(f"{x_var} - loss: {loss:.5e}\n")

        return float(loss)

    return hyperopt_main


def optimize_hyperparameters(
    E: float,
    nu: float,
    density,
    domain: Domain,
    device: torch.device,
    datafile_path: str,
    elasticity_design: ElasticityDesign,
):
    os.makedirs(os.path.dirname(datafile_path), exist_ok=True)
    with open(datafile_path, "w") as datafile:
        hyperopt_main = hyperopt_main_generator(
            E, nu, density, domain, device, datafile, elasticity_design
        )

        activation_functions = ["tanh", "relu", "rrelu", "sigmoid"]
        space = {
            "layer_count": hp.quniform("layer_count", 3, 5, 1),
            "neuron_count": 2 * hp.quniform("neuron_count", 10, 60, 1),
            "learning_rate": hp.loguniform("learning_rate", 0, 2),
            "CNN_deviation": hp.uniform("CNN_deviation", 0, 1),
            "rff_deviation": hp.uniform("rff_deviation", 0, 1),
            "iteration_count": hp.quniform("iteration_count", 40, 100, 1),
            "activation_function": hp.choice(
                "activation_function", activation_functions
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
        assert best is not None

        best["activation_function"] = activation_functions[best["activation_function"]]
        print("--- Optimal parameters ---")
        print(best)
        datafile.write(f"\n--- Optimal parameters ---\n{best}")


def run(design_path: str):
    solver = Solver(design_path)

    design = os.path.splitext(os.path.basename(design_path))[0]
    datafile_path = f"output/hyperopt/{design}_hyperopt.txt"
    optimize_hyperparameters(
        solver.problem.dem.E,
        solver.problem.dem.nu,
        solver.rho,
        solver.domain,
        solver.device,
        datafile_path,
        solver.problem.design,
    )
