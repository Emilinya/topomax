import os
import sys
import json
from src.solver import Solver
from src.fluid_problem import FluidProblem
from src.elasisity_problem import ElasticityProblem


def get_problem(design_file):
    if not os.path.isfile(design_file):
        print("Got a design path that is not a file!")
        return None

    with open(design_file, "r") as data:
        try:
            design = json.load(data)
        except:
            print("Got a domain size that is not an integer!")
            return None

    if design.get("problem"):
        if design["problem"] == "elasticity":
            return ElasticityProblem()
        elif design["problem"] == "fluid":
            return FluidProblem()
        else:
            print(f"Got design with unknown problem: '{design['problem']}'")
            return None
    else:
        print(f"Got design without problem!")
        return None


if __name__ == "__main__":
    if len(sys.argv) == 3:
        got_error = False

        design_file = sys.argv[1]
        problem = get_problem(design_file)
        if not problem:
            got_error = True

        N = sys.argv[2]
        try:
            N = int(N)
        except:
            print("Got a domain size that is not an integer!")
            got_error = True

        if got_error:
            print("This program is used as follows:")
            print(f"  python3 {sys.argv[0]} <design file> <domain size (N)>")
        else:
            solver = Solver(design_file, N, problem)
            solver.solve()
    else:
        print("Got an invalid number of arguments. This program is used as follows:")
        print(f"  python3 {sys.argv[0]} <design file> <domain size (N)>")
