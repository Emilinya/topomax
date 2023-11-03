import os
import sys
import pickle
import numpy as np
from tqdm import tqdm
from matplotlib import colors
import matplotlib.pyplot as plt

from designs.definitions import ProblemType
from designs.design_parser import parse_design


def create_cmap(start, middle, end, name):
    cdict = {
        "red": [
            [0.0] + [start[0]] * 2,
            [0.5] + [middle[0]] * 2,
            [1.0] + [end[0]] * 2,
        ],
        "green": [
            [0.0] + [start[1]] * 2,
            [0.5] + [middle[1]] * 2,
            [1.0] + [end[1]] * 2,
        ],
        "blue": [
            [0.0] + [start[2]] * 2,
            [0.5] + [middle[2]] * 2,
            [1.0] + [end[2]] * 2,
        ],
    }
    return colors.LinearSegmentedColormap(name, segmentdata=cdict)


def get_design_data(data_path: str):
    with open(data_path, "rb") as datafile:
        data_obj = pickle.load(datafile)
    objective = data_obj["objective"]
    w, h = data_obj["domain_size"]
    k = data_obj["iteration"]

    data_root, _ = os.path.split(data_path)
    rho_path = os.path.join(data_root, f"{k=}_rho.npy")
    data = np.load(rho_path)

    return data, objective, w, h


def create_design_figure(ax, data: np.ndarray, w: float, h: float, cmap):
    Ny, Nx = data.shape
    X, Y = np.meshgrid(np.linspace(0, w, Nx), np.linspace(0, h, Ny))
    return ax.pcolormesh(X, Y, data, cmap=cmap, vmin=0, vmax=1)


def plot_design(design, data_path: str, k: int, cmap):
    data, objective, w, h = get_design_data(data_path)

    plt.figure(figsize=(6.4 * w / h, 4.8))
    plt.rcParams.update({"font.size": 10 * np.sqrt(w / h)})

    mappable = create_design_figure(plt, data, w, h, cmap)
    plt.colorbar(mappable, label=r"$\rho(x, y)$ []")
    plt.xlim(0, w)
    plt.ylim(0, h)
    plt.gca().set_aspect("equal", "box")

    plt.xlabel("$x$ []")
    plt.ylabel("$y$ []")
    plt.title(f"k={k:.5g}, objective={objective:.3g}")

    output_file = os.path.join("output", "DEM", design, "figures", f"{k=}") + ".png"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close()


def multiplot(design: str, vals: list[tuple[str, int]], cmap):
    *_, w, h = get_design_data(vals[0][0])
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

    for ax, (data_path, k) in zip(axss.flat, vals):
        ax.axis("off")
        ax.set_aspect("equal", "box")
        data, _, w, h = get_design_data(data_path)
        pcolormesh = create_design_figure(ax, data, w, h, cmap)
        ax.set_title(f"${k=}$")
    fig.colorbar(pcolormesh, ax=axss[:, -1], shrink=0.8, label=r"$\rho(x, y)$ []")
    output_file = os.path.join("output", "DEM", design, "figures", f"multi.png")
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

    # designs = {design: [(data_path, k)]}
    designs: dict[str, list[tuple[str, int]]] = {}

    for design in os.listdir(os.path.join("output", "DEM")):
        if selected_designs is not None and design not in selected_designs:
            continue

        data_folder = os.path.join("output", "DEM", design, "data")
        if not os.path.isdir(data_folder):
            continue

        designs[design] = designs.get(design, [])

        for data in os.listdir(data_folder):
            data_path = os.path.join(data_folder, data)
            if not os.path.isfile(data_path):
                continue

            k_str, *other = data[:-4].split("_")
            if len(other) > 0:
                continue

            k = int(k_str.split("=")[1])
            designs[design].append((data_path, k))

    plot_count = 0
    for ks in designs.values():
        ks.sort(key=lambda v: v[1])
        ks[:] = reduce_length(ks, 6)

        plot_count += len(ks) + 1

    # we have no designs :(
    if plot_count == 0:
        sys.exit()

    with tqdm(total=plot_count) as pbar:
        for design, ks in designs.items():
            parameters, *_ = parse_design(os.path.join("designs", design) + ".json")
            if parameters.problem == ProblemType.FLUID:
                cmap = traa_cmap
            else:
                cmap = highlight_cmap

            for data_path, k in ks:
                plot_design(design, data_path, k, cmap)
                pbar.update(1)
            multiplot(design, ks, cmap)
            pbar.update(1)


if __name__ == "__main__":
    main()
