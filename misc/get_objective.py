import os
import pickle
import argparse

import dolfin as df
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from FEM_src.solver import FEMSolver
from FEM_src.utils import sample_function
from plot import create_cmap


class DEMDesign(df.UserExpression):
    def __init__(
        self,
        domain_size: tuple[float, float],
        data: npt.NDArray[np.float64],
        N: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.width, self.height = domain_size
        self.data = data
        self.N = N
        self.Nx = int(N * self.width)
        self.Ny = int(N * self.height)

    def eval(self, values, pos):
        xi, yi = int(pos[0] * self.N), int(pos[1] * self.N)

        if pos[0] == self.width:
            xi -= 1
        if pos[1] == self.height:
            yi -= 1

        values[0] = self.data[yi, xi]

    def value_shape(self):
        return ()


def get_design(N: int, design_file: str):
    solver = "DEM"
    design = os.path.splitext(os.path.basename(design_file))[0]
    data_folder = os.path.join("output", solver, design, "data")

    design_path = ""
    max_k = 0

    if not os.path.isdir(data_folder):
        raise ValueError(f"{data_folder} is not a directory!")

    for data_file in os.listdir(data_folder):
        N_str, _p_str, *rest = data_file[:-4].split("_")
        if len(rest) > 1 or rest[0] == "result":
            continue

        k_str = rest[0]

        data_N = int(N_str.split("=")[1])
        data_k = int(k_str.split("=")[1])

        if data_N == N and data_k > max_k:
            design_path = os.path.join(data_folder, data_file)
            max_k = data_k

    with open(design_path, "rb") as datafile:
        data_obj = pickle.load(datafile)

    objective = data_obj["objective"]
    w, h = data_obj["domain_size"]

    rho_path = os.path.splitext(design_path)[0] + "_rho.npy"
    data: npt.NDArray[np.float64] = np.load(rho_path)

    return design, objective, DEMDesign((w, h), data, N)


def plot_projection(design, rho, big_N, domain_size):
    cmap = create_cmap((1, 1, 1), (1, 0, 0), (0, 0, 0), "highlight")
    (x_ray, y_ray), rho_grid = sample_function(rho, big_N, "center")
    plt.pcolormesh(x_ray, y_ray, rho_grid[:, :, 0], vmin=0, vmax=1, cmap=cmap)
    plt.colorbar()

    plt.xlim(0, domain_size[0])
    plt.ylim(0, domain_size[1])
    plt.gca().set_aspect("equal", "box")

    out_path = f"misc/output/projections/{design}.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.savefig(out_path, dpi=200, bbox_inches="tight")


def main(N: int, design_file: str):
    design, dem_objective, dem_design = get_design(N, design_file)

    big_N = N * 4

    solver = FEMSolver(big_N, design_file)

    solver.problem.set_penalization(solver.parameters.penalties[0])
    rho = df.project(dem_design, solver.control_space)
    fem_objective = solver.problem.calculate_objective(rho)

    plot_projection(design, rho, big_N, (dem_design.width, dem_design.height))

    print(f"{design_file}, {N=}")
    print(
        f"DEM got an objective of {dem_objective:.6g}, but FEM gives {fem_objective:.6g}"
    )


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

    args = parser.parse_args()
    design_filename = args.design_file.name
    args.design_file.close()

    main(args.N, design_filename)
