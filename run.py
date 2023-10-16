import sys
import json
import argparse
from src.solver import Solver
from src.fluid_problem import FluidProblem
from src.elasisity_problem import ElasticityProblem


def get_problem(design_file):
    design = json.load(design_file)

    if design.get("problem"):
        if design["problem"] == "elasticity":
            return ElasticityProblem()
        if design["problem"] == "fluid":
            return FluidProblem()

        sys.exit(f"Got design with unknown problem: '{design['problem']}'")
    else:
        sys.exit("Got design without problem")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "design_file",
        type=argparse.FileType("r"),
        help="path to a json file where your problem is defined. See readme for more information",
    )
    parser.add_argument(
        "N",
        metavar="element_count",
        type=int,
        help="the number of finite elements in a unit length",
    )
    parser.add_argument(
        "-d",
        "--data_path",
        required=False,
        default="output",
        help="the folder where the data output is stored (default: 'output')",
    )
    parser.add_argument(
        "-k",
        "--skip_multiple",
        type=int,
        required=False,
        default=1,
        help="only stores data if the iteration is a multiple of skip_multiple. "
        + "Last iteration is always stored. (default: 1)",
    )

    args = parser.parse_args()

    N = args.N
    data_path = args.data_path
    skip_multiple = args.skip_multiple
    design_filename = args.design_file.name

    problem = get_problem(args.design_file)
    args.design_file.close()

    solver = Solver(
        N,
        design_filename,
        problem,
        data_path,
        skip_multiple,
    )
    solver.solve()
