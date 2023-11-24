import argparse


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
        "-n",
        "--use_neural_network_solver",
        action="store_true",
        help="with this flag, the program wil solve the state equation using the "
        + "DEM instead of the FEM. The DEM is worse in every way, don't use it.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="with this flag, the program wil print the loss as the DEM trains, "
        + "if you are using the DEM solver",
    )
    parser.add_argument(
        "-o",
        "--optimize_hyperparameters",
        action="store_true",
        help="with this flag, the program wil optimize the hyperparameters for the given design",
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
    design_filename = args.design_file.name
    args.design_file.close()

    if args.optimize_hyperparameters:
        from DEM_src.optimize_hyperparameters import run

        run(design_filename, args.data_path)
    else:
        if args.use_neural_network_solver:
            from DEM_src.solver import DEMSolver

            solver = DEMSolver(args.N, design_filename, args.data_path, args.verbose)
            solver.solve()
        else:
            from FEM_src.solver import FEMSolver

            solver = FEMSolver(
                args.N,
                design_filename,
                args.data_path,
                args.skip_multiple,
            )
            solver.solve()
