import os
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
        elif design["problem"] == "fluid":
            return FluidProblem()
        else:
            print(f"Got design with unknown problem: '{design['problem']}'")
            exit()
    else:
        print(f"Got design without problem!")
        exit()


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
        "--skip_frequency",
        type=int,
        required=False,
        default=1,
        help="how many iterations are skipped before data is stored. "
        + "Last iteration is always stored. (default: 0)",
    )
    parser.add_argument(
        "-s",
        "--data_size",
        type=str,
        required=False,
        choices=["small", "medium", "large"],
        default="small",
        help="How many points are sampled when producing output data. "
        + "'small' gives one point per cell for most data, and 9 points per cell for the final data. "
        + "'medium' gives 9 points for most data, and 25 points form the final data. "
        + "'large' gives 25 points for most data, and 49 points from the final data. "
        + "(default: 'small')",
    )

    args = parser.parse_args()

    N = args.N
    data_path = args.data_path
    skip_frequency = args.skip_frequency
    design_filename = args.design_file.name

    if args.data_size == "smal":
        data_multiple = 1
        final_data_multiple = 3
    elif args.data_size == "medium":
        data_multiple = 3
        final_data_multiple = 5
    elif args.data_size == "large":
        data_multiple = 5
        final_data_multiple = 9

    problem = get_problem(args.design_file)
    args.design_file.close()

    solver = Solver(
        N,
        design_filename,
        problem,
        data_path,
        data_multiple,
        skip_frequency,
        final_data_multiple,
    )
    solver.solve()
