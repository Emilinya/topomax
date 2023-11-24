import os
import sys
import pickle
from typing import Literal, Sequence

from tqdm import tqdm
import numpy as np
import numpy.typing as npt
from matplotlib import colors
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.collections import QuadMesh

from designs.definitions import ProblemType
from designs.design_parser import parse_design
from FEM_src.utils import load_function, sample_function


def create_cmap(start, middle, end, name):
    cdict: dict[
        Literal["red", "green", "blue", "alpha"], Sequence[tuple[float, ...]]
    ] = {
        "red": [
            tuple([0.0] + [start[0]] * 2),
            tuple([0.5] + [middle[0]] * 2),
            tuple([1.0] + [end[0]] * 2),
        ],
        "green": [
            tuple([0.0] + [start[1]] * 2),
            tuple([0.5] + [middle[1]] * 2),
            tuple([1.0] + [end[1]] * 2),
        ],
        "blue": [
            tuple([0.0] + [start[2]] * 2),
            tuple([0.5] + [middle[2]] * 2),
            tuple([1.0] + [end[2]] * 2),
        ],
    }
    return colors.LinearSegmentedColormap(name, segmentdata=cdict)


def get_design_data(
    solver: str, data_path: str
) -> tuple[npt.NDArray[np.float64], float, float, float]:
    with open(data_path, "rb") as datafile:
        data_obj = pickle.load(datafile)
    objective = data_obj["objective"]
    w, h = data_obj["domain_size"]

    data_root, _ = os.path.splitext(data_path)

    if solver == "FEM":
        rho_path = data_root + "_rho.dat"
        rho, *_ = load_function(rho_path)
        _, data = sample_function(rho, 200 / min(w, h), "center")
        data = data[:, :, 0]
    else:
        rho_path = data_root + "_rho.npy"
        data = np.load(rho_path)

    return data, objective, w, h


def create_design_figure(
    ax: Axes, data: np.ndarray, w: float, h: float, N: int, cmap: colors.Colormap
):
    multiple = int(np.sqrt(data.size / (w * h * N**2)))
    Nx, Ny = int(N * w * multiple), int(N * h * multiple)

    X, Y = np.meshgrid(np.linspace(0, w, Nx), np.linspace(0, h, Ny))
    return ax.pcolormesh(X, Y, data, cmap=cmap, vmin=0, vmax=1)


def plot_design(
    solver: str,
    design: str,
    data_path: str,
    N: int,
    p: float,
    k: int,
    cmap: colors.Colormap,
):
    data, objective, w, h = get_design_data(solver, data_path)

    plt.figure(figsize=(6.4 * w / h, 4.8))
    plt.rcParams.update({"font.size": 10 * np.sqrt(w / h)})

    mappable = create_design_figure(plt.gca(), data, w, h, N, cmap)
    plt.colorbar(mappable, label=r"$\rho(x, y)$ []")
    plt.xlim(0, w)
    plt.ylim(0, h)
    plt.gca().set_aspect("equal", "box")

    plt.xlabel("$x$ []")
    plt.ylabel("$y$ []")
    plt.title(f"{N=}, p={p:.5g}, k={k:.5g}, objective={objective:.3g}")

    output_file = (
        os.path.join("output", solver, design, "figures", f"{N=}_{p=}_{k=}") + ".png"
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close()


def multiplot(
    solver: str,
    design: str,
    N: int,
    p: float,
    vals: list[tuple[str, int]],
    cmap: colors.Colormap,
):
    *_, w, h = get_design_data(solver, vals[0][0])
    fig, axss = plt.subplots(
        2,
        3,
        figsize=(6.4 * w / h, 4.8),
        sharex=True,
        sharey=True,
        gridspec_kw={"wspace": 0.04, "hspace": 0.05, "width_ratios": [1, 1, 1.25]},
    )
    plt.rcParams.update({"font.size": 9 + 3 * w / h})

    if len(vals) < 6:
        return
    if len(vals) > 6:
        vals[5] = vals[-1]

    pcolormesh: QuadMesh | None = None
    for ax, (data_path, k) in zip(axss.flat, vals):
        assert isinstance(ax, Axes)

        ax.axis("off")
        ax.set_aspect("equal", "box")
        data, _, w, h = get_design_data(solver, data_path)
        pcolormesh = create_design_figure(ax, data, w, h, N, cmap)
        ax.set_title(f"${k=}$")
    assert pcolormesh is not None

    fig.colorbar(pcolormesh, ax=axss[:, -1], shrink=0.8, label=r"$\rho(x, y)$ []")
    output_file = os.path.join(
        "output", solver, design, "figures", f"{N=}_{p=}_multi.png"
    )
    plt.savefig(output_file, dpi=200, bbox_inches="tight")


def reduce_length(long_list: list, desired_length: int):
    if len(long_list) <= desired_length:
        return long_list

    space = int(np.ceil(len(long_list) / (desired_length - 1)))
    spaces = np.full(desired_length - 1, space, dtype=int)
    extra = space * (desired_length - 1) - (len(long_list) - 1)

    # if extra > desired_length - 1, remove from all spaces equally
    spaces -= int(extra / (desired_length - 1))
    extra %= desired_length - 1

    # if extra < desired_length - 1, remove from first spaces
    spaces[:extra] -= 1

    idx = 0
    new_list = [long_list[idx]]
    for space in spaces:
        idx += space
        new_list.append(long_list[idx])

    return new_list


def get_designs(solver: str, selected_designs: list[str] | None):
    # designs = {design: {N: {p: [(data_path, k)]}}}
    designs: dict[str, dict[int, dict[float, list[tuple[str, int]]]]] = {}

    for design in os.listdir(os.path.join("output", solver)):
        if selected_designs is not None and design not in selected_designs:
            continue

        data_folder = os.path.join("output", solver, design, "data")
        if not os.path.isdir(data_folder):
            continue

        designs[design] = designs.get(design, {})

        for data in os.listdir(data_folder):
            data_path = os.path.join(data_folder, data)
            if not os.path.isfile(data_path):
                continue

            N_str, p_str, k_str, *other = data[:-4].split("_")
            if other != []:
                continue

            N = int(N_str.split("=")[1])
            p = float(p_str.split("=")[1])
            k = int(k_str.split("=")[1])

            designs[design][N] = designs[design].get(N, {})
            designs[design][N][p] = designs[design][N].get(p, [])
            designs[design][N][p].append((data_path, k))

    return designs


def plot_designs(
    solver: str,
    designs: dict[str, dict[int, dict[float, list[tuple[str, int]]]]],
    fluid_cmap: colors.Colormap,
    elasticity_cmap: colors.Colormap,
):
    plot_count = 0
    for design_dict in designs.values():
        for ps in design_dict.values():
            for ks in ps.values():
                ks.sort(key=lambda v: v[1])
                ks[:] = reduce_length(ks, 6)

                plot_count += len(ks) + 1

    # we have no designs :(
    if plot_count == 0:
        return

    with tqdm(total=plot_count) as pbar:
        for design, design_dict in designs.items():
            parameters, *_ = parse_design(os.path.join("designs", design) + ".json")
            if parameters.problem == ProblemType.FLUID:
                cmap = fluid_cmap
            else:
                cmap = elasticity_cmap

            for N, ps in design_dict.items():
                for p, ks in ps.items():
                    for data_path, k in ks:
                        plot_design(solver, design, data_path, N, p, k, cmap)
                        pbar.update(1)
                    multiplot(solver, design, N, p, ks, cmap)
                    pbar.update(1)


def main():
    # colormap with the colors of the trans flag, from red to white to blue
    traa_blue = [91 / 255, 206 / 255, 250 / 255]
    traa_red = [245 / 255, 169 / 255, 184 / 255]
    traa_cmap = create_cmap(traa_red, (1, 1, 1), traa_blue, "traa")

    # colormap with between-values highlighted, from white to red to black
    highlight_cmap = create_cmap((1, 1, 1), (1, 0, 0), (0, 0, 0), "highlight")

    # colormap from white to black
    boring_cmap = create_cmap((1, 1, 1), (0.5, 0.5, 0.5), (0, 0, 0), "boring")

    selected_designs = None
    if len(sys.argv) > 1:
        selected_designs = sys.argv[1:]

    all_solvers = ["FEM", "DEM"]
    solvers = []
    if selected_designs is not None:
        for solver in all_solvers:
            if solver in selected_designs and solver not in solvers:
                solvers.append(solver)
    if len(solvers) == 0:
        solvers = all_solvers

    for solver in solvers:
        designs = get_designs(solver, selected_designs)
        plot_designs(solver, designs, traa_cmap, highlight_cmap)


if __name__ == "__main__":
    main()
