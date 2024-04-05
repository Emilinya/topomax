import io
import os

import numpy as np
import numpy.typing as npt
from hyperopt import fmin, tpe, hp, Trials

from DEM_src.solver import DEMSolver
from DEM_src.problem import DEMProblem
from DEM_src.deep_energy_method import NNParameters


def hyperopt_main_generator(
    rho: npt.NDArray,
    problem: DEMProblem,
    datafile: io.TextIOWrapper,
    iteration_data: dict[str, int | float],
):
    def hyperopt_main(x_var: dict[str, int | float | str]):
        nn_parameters = NNParameters(
            layer_count=int(x_var["layer_count"]),
            neuron_count=int(x_var["neuron_count"]),
            learning_rate=float(x_var["learning_rate"]),
            weight_deviation=float(x_var["weight_deviation"]),
            fourier_deviation=float(x_var["fourier_deviation"]),
            iteration_count=int(x_var["iteration_count"]),
            activation_function=str(x_var["activation_function"]),
            convergence_tolerance=5e-5,
        )

        i = iteration_data["iteration"]
        N = iteration_data["iteration_count"]
        min_loss = iteration_data["min_loss"]
        text = f"hyperopt iteration {i+1}/{N} ({int((i+1)/N*100)}%). Min loss: {min_loss:.6g}"

        try:
            terminal_width = os.get_terminal_size().columns
        except OSError:
            terminal_width = 80
        else:
            if i != 0:
                # we don't want to spam the terminal - move cursor up 3 rows
                # to overwrite previous line (why 3 and not 2?)
                print("\033[3A")

        left_width = int((terminal_width - (len(text) + 2)) / 2)
        right_width = (terminal_width - (len(text) + 2)) - left_width
        print(f"{'─'*left_width} {text} {'─'*right_width}", flush=True)

        problem.dem.set_nn_parameters(nn_parameters)
        problem.dem.train_model(rho, problem.mesh)
        loss = problem.dem.get_loss(rho, problem.mesh)

        datafile.write(f"{x_var} - loss: {loss:.6g}\n")
        iteration_data["iteration"] += 1
        iteration_data["min_loss"] = min(loss, iteration_data["min_loss"])

        return loss

    return hyperopt_main


def optimize_hyperparameters(
    rho,
    problem: DEMProblem,
    datafile_path: str,
):
    os.makedirs(os.path.dirname(datafile_path), exist_ok=True)
    with open(datafile_path, "w", encoding="utf-8") as datafile:
        iteration_data = {
            "iteration": 0,
            "iteration_count": 100,
            "min_loss": float("Infinity"),
        }
        hyperopt_main = hyperopt_main_generator(rho, problem, datafile, iteration_data)

        activation_functions = ["tanh", "relu", "rrelu", "sigmoid"]
        space = {
            "layer_count": hp.quniform("layer_count", 3, 5, 1),
            "neuron_count": 2 * hp.quniform("neuron_count", 10, 60, 1),
            "learning_rate": hp.loguniform("learning_rate", 0, 2),
            "weight_deviation": hp.uniform("weight_deviation", 0, 1),
            "fourier_deviation": hp.uniform("fourier_deviation", 0, 1),
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
            verbose=False,
        )
        assert best is not None

        best["activation_function"] = activation_functions[best["activation_function"]]
        print("--- Optimal parameters ---")
        print(best)
        datafile.write(f"\n--- Optimal parameters ---\n{best}")


def run(design_path: str, output_path: str = "output"):
    try:
        os.get_terminal_size()
    except OSError:
        # if terminal has no size, \r might not work,
        # so we don't want the solver to be verbose.
        solver = DEMSolver(40, design_path, verbose=False)
    else:
        solver = DEMSolver(40, design_path, verbose=True)

    solver.problem.set_penalization(solver.parameters.penalties[0])

    design = os.path.splitext(os.path.basename(design_path))[0]
    datafile_path = f"{output_path}/hyperopt/{design}_hyperopt.txt"
    optimize_hyperparameters(solver.rho, solver.problem, datafile_path)
