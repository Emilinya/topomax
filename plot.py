import os
import argparse
from dataclasses import dataclass
from typing import Literal, Sequence

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
from matplotlib.collections import QuadMesh
from matplotlib.axes import Axes
from matplotlib import colors
from tqdm import tqdm

from designs.definitions import ProblemType
from designs.design_parser import parse_design
from FEM_src.utils import load_function, sample_function
from src.utils import SolverResult, IterationData, get_solver_data


@dataclass
class DesignData:
    design: str
    data: dict[int, dict[str, tuple[list[IterationData], SolverResult]]]


def create_cmap(
    start: tuple[float, float, float],
    middle: tuple[float, float, float],
    end: tuple[float, float, float],
    name: str,
):
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


def get_rho(
    method: str, data_folder, data: IterationData, points: int
) -> tuple[npt.NDArray[np.float64], float, float, float]:
    objective = data.objective
    w, h = data.domain_size

    rho_path = os.path.join(data_folder, data.rho_file)

    if method == "FEM":
        rho, *_ = load_function(rho_path)
        _, design_data = sample_function(rho, int(points / min(w, h)), "center")
        design_data = design_data[:, :, 0]
    else:
        design_data = np.load(rho_path)

    return design_data, objective, w, h


def create_design_figure(
    ax: Axes, data: np.ndarray, w: float, h: float, N: int, cmap: colors.Colormap
):
    N = int(N / min(w, h))
    multiple = int(np.sqrt(data.size / (w * h * N**2)))
    Nx, Ny = int(N * w * multiple), int(N * h * multiple)

    X, Y = np.meshgrid(np.linspace(0, w, Nx), np.linspace(0, h, Ny))
    return ax.pcolormesh(X, Y, data, cmap=cmap, vmin=0, vmax=1)


def plot_design(
    N: int,
    p: str,
    k: int,
    data: IterationData,
    cmap: colors.Colormap,
    method: str,
    design: str,
    simple: bool,
    output_path: str,
):
    data_folder = f"{output_path}/{method}/{design}/data"
    rho, objective, w, h = get_rho(method, data_folder, data, 200)

    fig = plt.figure(figsize=(6.4 * w / h, 4.8))
    plt.rcParams.update({"font.size": 10 * np.sqrt(w / h)})

    mappable = create_design_figure(plt.gca(), rho, w, h, N, cmap)
    if simple:
        plt.xticks([], [])
        plt.yticks([], [])
    else:
        plt.colorbar(mappable, label=r"$\rho(x, y)$ []")
        plt.xlabel("$x$ []")
        plt.ylabel("$y$ []")
        plt.title(f"{N=}, p={float(p):.5g}, k={k:.5g}, objective={objective:.3g}")

    plt.xlim(0, w)
    plt.ylim(0, h)
    plt.gca().set_aspect("equal", "box")

    output_file = (
        os.path.join(output_path, method, design, "figures", f"{N=}_p={p}_{k=}")
        + ".png"
    )
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)


def multiplot(
    N: int,
    p: str,
    cmap: colors.Colormap,
    vals: list[IterationData],
    method: str,
    design: str,
    output_path: str,
):
    data_folder = f"{output_path}/{method}/{design}/data"
    *_, w, h = get_rho(method, data_folder, vals[0], 1)
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
    for ax, data in zip(axss.flat, vals):
        assert isinstance(ax, Axes)

        ax.axis("off")
        ax.set_aspect("equal", "box")
        rho, _, w, h = get_rho(method, data_folder, data, 100)
        pcolormesh = create_design_figure(ax, rho, w, h, N, cmap)
        ax.set_title(f"$k={data.iteration}$")
    assert pcolormesh is not None

    fig.colorbar(pcolormesh, ax=axss[:, -1], shrink=0.8, label=r"$\rho(x, y)$ []")
    output_file = os.path.join(
        output_path, method, design, "figures", f"{N=}_p={p}_multi.png"
    )
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close(fig)


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


def get_design_data_dict(
    methods: list[str] | None,
    designs: list[str] | None,
    Ns: list[int] | None,
    output_path: str,
):
    design_data_dict: dict[str, list[DesignData]] = {}

    for method in os.listdir(output_path):
        if methods is not None and method not in methods:
            continue

        design_data_dict[method] = []

        for design in os.listdir(os.path.join(output_path, method)):
            if designs is not None and design not in designs:
                continue

            data_folder = os.path.join(output_path, method, design, "data")
            if not os.path.isdir(data_folder):
                continue

            results, data_list = get_solver_data(method, design, output_path)

            design_data_dict[method].append(DesignData(design, {}))
            design_data = design_data_dict[method][-1]

            for N, p, _, data in data_list:
                if Ns is not None and N not in Ns:
                    continue

                if not design_data.data.get(N):
                    design_data.data[N] = {}

                p_data = design_data.data[N]
                if not p_data.get(p):
                    for r_N, r_p, result in results:
                        if r_N == N and r_p == p:
                            p_data[p] = ([], result)
                            break

                p_data[p][0].append(data)

    return design_data_dict


def plot_designs(
    design_dict: dict[str, list[DesignData]],
    fluid_cmap: colors.Colormap,
    elasticity_cmap: colors.Colormap,
    simple: bool,
    output_path: str,
):
    plot_count = 0
    for designs in design_dict.values():
        for design_data in designs:
            for p_data in design_data.data.values():
                for p, (data_list, result) in p_data.items():
                    data_list.sort(key=lambda v: v.iteration)

                    min_index = result.min_index
                    if min_index != (len(data_list) - 1):
                        data_list = data_list[: min_index + 1]

                    p_data[p] = (reduce_length(data_list, 6), result)

                    plot_count += len(p_data[p][0]) + 1

    # we have no designs :(
    if plot_count == 0:
        return

    with tqdm(total=plot_count) as pbar:
        for method, designs in design_dict.items():
            for design_data in designs:
                design = design_data.design

                parameters, *_ = parse_design(os.path.join("designs", design) + ".json")
                if parameters.problem == ProblemType.FLUID:
                    cmap = fluid_cmap
                else:
                    cmap = elasticity_cmap

                for N, p_data in design_data.data.items():
                    for p, (data_list, _) in p_data.items():
                        for data in data_list:
                            plot_design(
                                N,
                                p,
                                data.iteration,
                                data,
                                cmap,
                                method,
                                design,
                                simple,
                                output_path,
                            )
                            pbar.update(1)
                        multiplot(
                            N,
                            p,
                            cmap,
                            data_list,
                            method,
                            design_data.design,
                            output_path,
                        )
                        pbar.update(1)


def main(
    methods: list[str] | None,
    designs: list[str] | None,
    Ns: list[int] | None,
    simple: bool,
    output_path: str,
):
    # colormap with the colors of the trans flag, from red to white to blue
    traa_blue = (91 / 255, 206 / 255, 250 / 255)
    traa_red = (245 / 255, 169 / 255, 184 / 255)
    traa_cmap = create_cmap(traa_red, (1, 1, 1), traa_blue, "traa")

    # colormap with between-values highlighted, from white to red to black
    highlight_cmap = create_cmap((1, 1, 1), (1, 0, 0), (0, 0, 0), "highlight")

    design_data = get_design_data_dict(methods, designs, Ns, output_path)
    plot_designs(design_data, traa_cmap, highlight_cmap, simple, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-m",
        "--method",
        nargs="+",
        required=False,
        help="The methods you want to plot the designs form. Allows multiple values, "
        + "but they must be either 'FEM' or 'DEM'. If not set, both methods will be used.",
    )
    parser.add_argument(
        "-d",
        "--design",
        nargs="+",
        required=False,
        help="The designs you want to plot. If not set, all designs will be plotted.",
    )
    parser.add_argument(
        "-N",
        "--elements",
        type=int,
        nargs="+",
        required=False,
        help="The discritization parameters you want to plot. "
        + "If not set, all N values will be plotted.",
    )
    parser.add_argument(
        "-s",
        "--simple",
        action="store_true",
        help="With this flag, the figures created will show just the design. "
        + "They will have have no axis, no title and no colorbar.",
    )
    parser.add_argument(
        "-o",
        "--output_path",
        required=False,
        default="output",
        help="the folder where the data you want to plot is stored (default: 'output')",
    )

    args = parser.parse_args()
    main(args.method, args.design, args.elements, args.simple, args.output_path)
