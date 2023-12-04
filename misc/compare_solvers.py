import os
import argparse

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from designs.design_parser import parse_design
from DEM_src.solver import DEMSolver
from DEM_src.utils import unflatten
from FEM_src.solver import FEMSolver
from FEM_src.utils import sample_function


def FEM(design_path: str, N: int, ax: Axes):
    solver = FEMSolver(N, design_path)

    solver.problem.set_penalization(solver.parameters.penalties[0])
    u = solver.problem.forward(solver.rho)

    (x_ray, y_ray), u_grid = sample_function(u, N, "edges")

    redux = 3
    small_ux = u_grid[::redux, ::redux, 0]
    small_uy = u_grid[::redux, ::redux, 1]
    small_x = x_ray[::redux]
    small_y = y_ray[::redux]

    colormesh = ax.pcolormesh(small_x, small_y, np.sqrt(small_ux**2 + small_uy**2))
    ax.quiver(small_x, small_y, small_ux, small_uy)
    plt.colorbar(colormesh)
    ax.set_title("FEM")

    return x_ray, y_ray, u_grid[:, :, 0], u_grid[:, :, 1]


def DEM(design_path: str, N: int, ax: Axes):
    solver = DEMSolver(N, design_path, verbose=True)

    solver.problem.set_penalization(solver.parameters.penalties[0])
    solver.problem.calculate_objective(solver.rho)
    u = solver.problem.dem.get_u(mesh=solver.mesh).detach().numpy()
    ux, uy = unflatten(u, solver.mesh.shape)

    redux = 3
    small_ux = ux[::redux, ::redux]
    small_uy = uy[::redux, ::redux]
    small_x = solver.mesh.x_grid[::redux, ::redux]
    small_y = solver.mesh.y_grid[::redux, ::redux]

    colormesh = ax.pcolormesh(small_x, small_y, np.sqrt(small_ux**2 + small_uy**2))
    ax.quiver(small_x, small_y, small_ux, small_uy)
    plt.colorbar(colormesh)
    ax.set_title("DEM")

    return solver.mesh.x_grid[0, :], solver.mesh.y_grid[:, 0], ux, uy


def compare(design_path: str, N: int):
    domain = parse_design(design_path)[0]
    w, h = domain.width, domain.height

    design = os.path.splitext(os.path.basename(design_path))[0]
    nice_design = " ".join(design.capitalize().split("_"))

    figure_width = 2.6 + 3 * (w / h) + 0.6 * (w / h) ** 2
    wspace = 0.2 + 0.05 * (w / h) - 0.05 * (w / h) ** 2

    fig = plt.figure(figsize=(figure_width, 4.8))
    gs = fig.add_gridspec(nrows=2, ncols=2, wspace=wspace, hspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    for ax in [ax1, ax2, ax3, ax4]:
        ax.set_aspect(1)
    ax2.get_yaxis().set_visible(False)
    ax4.get_yaxis().set_visible(False)

    x_ray, y_ray, fem_ux_grid, fem_uy_grid = FEM(design_path, N, ax1)
    dem_x_ray, dem_y_ray, dem_ux_grid, dem_uy_grid = DEM(design_path, N, ax2)

    if not np.all(x_ray == dem_x_ray) and np.all(y_ray == dem_y_ray):
        raise ValueError("FEM and DEM does not return the same x and y arrays!")

    inside_dem = np.array([dem_ux_grid[1:-1, 1:-1], dem_uy_grid[1:-1, 1:-1]])
    inside_fem = np.array([fem_ux_grid[1:-1, 1:-1], fem_uy_grid[1:-1, 1:-1]])

    dem_norm = np.sqrt(np.sum(inside_dem * inside_dem, axis=0))
    fem_norm = np.sqrt(np.sum(inside_fem * inside_fem, axis=0))

    dir_error = np.log10(
        abs((1 - np.sum(inside_dem * inside_fem, axis=0) / (dem_norm * fem_norm)) / 2)
        + 1e-14
    )
    norm_errors = np.log10(np.abs(dem_norm - fem_norm) / fem_norm + 1e-14)

    dir_colormesh = ax3.pcolormesh(x_ray[1:-1], y_ray[1:-1], dir_error)
    plt.colorbar(
        dir_colormesh,
        label="log$_{10}\\left(\\frac{1}{2} - \\frac{u_{FEM} \\cdot u_{DEM}}"
        + "{2|u_{FEM}||u_{DEM}|}\\right)$",
    )
    ax3.set_title("Direction errors")

    mag_colormesh = ax4.pcolormesh(x_ray[1:-1], y_ray[1:-1], norm_errors)
    plt.colorbar(
        mag_colormesh,
        label="log$_{10}\\left(\\frac{||u_{FEM}| - |u_{DEM}||}{|u_{FEM}|}\\right)$",
    )
    ax4.set_title("Norm errors")

    plt.suptitle(f"{nice_design} FEM DEM comparison")

    out_path = f"misc/output/comp/comp_{design}.png"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.savefig(out_path, dpi=200, bbox_inches="tight")


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

    compare(design_filename, args.N)
