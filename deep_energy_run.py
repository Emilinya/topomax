import argparse

from DEM_src.solver import Solver

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "design_file",
        type=argparse.FileType("r"),
        help="path to a json file where your problem is defined. See readme for more information",
    )

    args = parser.parse_args()
    design_filename = args.design_file.name
    args.design_file.close()

    solver = Solver(design_filename)
    solver.solve()
