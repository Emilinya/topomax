import os
import pickle

import dolfin as df
import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

from FEM_src.solver import FEMSolver
from FEM_src.utils import load_function, sample_function
from plot import create_cmap


class DEMRhoExpression(df.UserExpression):
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
        self.N = N / min(self.width, self.height)
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


def plot_interpolation(
    method: str,
    design: str,
    N: int,
    rho: df.Function,
    big_N: int,
    domain_size: tuple[float, float],
    old_objective: float,
    new_objective: float,
):
    w, h = domain_size
    fig = plt.figure(figsize=(6.4 * w / h, 4.8))
    plt.rcParams.update({"font.size": 10 * np.sqrt(w / h)})

    cmap = create_cmap((1, 1, 1), (1, 0, 0), (0, 0, 0), "highlight")
    (x_ray, y_ray), rho_grid = sample_function(rho, big_N, "center")
    plt.pcolormesh(x_ray, y_ray, rho_grid[:, :, 0], vmin=0, vmax=1, cmap=cmap)
    plt.colorbar()

    plt.xlim(0, domain_size[0])
    plt.ylim(0, domain_size[1])
    plt.gca().set_aspect("equal", "box")

    plt.title(f"Old: {old_objective:.5g}, New: {new_objective:.5g}")

    out_path = f"misc/output/interpolations/{method}_{N=}_{design}.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def get_data(method: str, design_file: str, N: int):
    design = os.path.splitext(os.path.basename(design_file))[0]
    data_folder = os.path.join("output", method, design, "data")

    if not os.path.isdir(data_folder):
        raise ValueError(f"{data_folder} is not a directory!")

    result_p = -float("Infinity")
    result_file = None

    for data_file in os.listdir(data_folder):
        if data_file[-10:-4] == "result":
            file_N = int(data_file.split("_")[0].split("=")[1])
            if N == file_N:
                p = float(data_file.split("_")[1].split("=")[1])
                if p > result_p:
                    result_file = data_file

    if result_file is None:
        raise ValueError(f"No result found with {N=} found in {data_folder}!")

    result_path = os.path.join(data_folder, result_file)
    with open(result_path, "rb") as datafile:
        result_obj = pickle.load(datafile)

    design_path = f"{result_path[:-11]}_k={result_obj.min_index}.dat"
    with open(design_path, "rb") as datafile:
        data_obj = pickle.load(datafile)

    objective = data_obj.objective
    rho_path = design_path[:-4] + "_rho"

    return design, objective, rho_path


def main(method: str, design_file: str, N: int):
    design, old_objective, rho_path = get_data(method, design_file, N)

    big_N = 320

    solver = FEMSolver(big_N, design_file)
    solver.problem.set_penalization(max(solver.parameters.penalties))

    if method == "DEM":
        data: npt.NDArray[np.float64] = np.load(rho_path + ".npy")
        rho_expression = DEMRhoExpression(solver.problem.domain_size, data, N)
        rho = df.interpolate(rho_expression, solver.control_space)
    else:
        small_rho, *_ = load_function(rho_path + ".dat")
        rho = df.interpolate(small_rho, solver.control_space)

    new_objective = solver.problem.calculate_objective(rho)

    plot_interpolation(
        method,
        design,
        N,
        rho,
        big_N,
        solver.problem.domain_size,
        old_objective,
        new_objective,
    )

    print(
        f"{method} {design} with {N=}: Old objective is {old_objective:.6g}, "
        + f"new objective is {new_objective:.6g}"
    )


def multimain():
    elasticity_designs = ["cantilever", "short_cantilever", "bridge"]
    fluid_designs = ["diffuser", "pipe_bend", "twin_pipe"]

    for method in ["DEM", "FEM"]:
        for design in elasticity_designs + fluid_designs:
            for N in [40, 80, 160, 320]:
                main(method, f"designs/{design}.json", N)


if __name__ == "__main__":
    multimain()
