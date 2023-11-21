import argparse

from DEM_src.solver import Solver
from DEM_src.optimize_hyperparameters import run

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "design_file",
        type=argparse.FileType("r"),
        help="path to a json file where your problem is defined. See readme for more information",
    )
    parser.add_argument(
        "-d",
        "--data_path",
        required=False,
        default="output",
        help="the folder where the data output is stored (default: 'output')",
    )
    parser.add_argument(
        "-o",
        "--optimize_hyperparameters",
        action="store_true",
        help="with this flag, the program wil optimize the hyperparameters for the given design",
    )

    args = parser.parse_args()
    optimize_hyperparameters = args.optimize_hyperparameters
    design_filename = args.design_file.name
    data_path = args.data_path
    args.design_file.close()

    if optimize_hyperparameters:
        run(design_filename, data_path)
    else:
        solver = Solver(design_filename)
        solver.solve()
