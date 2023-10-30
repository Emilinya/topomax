import argparse

from DeepEnergy.main import DeepEnergySolver

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "example",
        type=int,
        choices=[1, 2],
        help="the example you want to run",
    )

    args = parser.parse_args()

    solver = DeepEnergySolver(args.example)
    solver.solve()
