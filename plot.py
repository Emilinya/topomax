from designs.design_parser import parse_design
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from scipy import io
import numpy as np
import sys
import os

try:
    from tqdm import tqdm
except ModuleNotFoundError:

    def tqdm(values):
        N = len(values)
        if N == 0:
            return []
        if N == 1:
            return [values[0]]

        terminal_size = os.get_terminal_size().columns - 1
        for i, v in enumerate(values):
            progress_str = f"{i+1}/{N}"
            marker_space = terminal_size - (5 + len(progress_str))

            progress = i / (N - 1)
            markers = int(progress * marker_space)

            print(
                f"\r├{'█'*markers}{' '*(marker_space - markers)}┤ ({progress_str})",
                end="",
            )

            yield v
        print()

# create a colormap from black to white to light blue
traa_blue = [91 / 255, 206 / 255, 250 / 255]
cdict = {
    "red": [
        [0.0] + [traa_blue[0]] * 2,
        [0.5] + [1.0] * 2,
        [1.0] + [0.0] * 2,
    ],
    "green": [
        [0.0] + [traa_blue[1]] * 2,
        [0.5] + [1.0] * 2,
        [1.0] + [0.0] * 2,
    ],
    "blue": [
        [0.0] + [traa_blue[2]] * 2,
        [0.5] + [1.0] * 2,
        [1.0] + [0.0] * 2,
    ],
}
black2blue = colors.LinearSegmentedColormap("testCmap", segmentdata=cdict)


def plot_design(design, data_path, N, k):
    mat = io.loadmat(data_path)
    data = mat["data"]
    objective = mat["objective"][0, 0]

    parameters, *_ = parse_design(os.path.join("designs", design) + ".json")
    w, h = parameters.width, parameters.height

    Nx, Ny = int(N * w), int(N * h)

    plt.figure(figsize=(6.4 * w / h, 4.8))
    plt.rcParams.update({"font.size": 10 * np.sqrt(w / h)})

    X, Y = np.meshgrid(np.linspace(0, w, Nx), np.linspace(0, h, Ny))
    plt.pcolormesh(X, Y, data, cmap=black2blue, vmin=0, vmax=1)
    plt.colorbar(label=r"$\rho(x, y)$ []")
    plt.xlim(0, w)
    plt.ylim(0, h)
    plt.gca().set_aspect("equal", "box")

    plt.xlabel("$x$ []")
    plt.ylabel("$y$ []")
    plt.title(f"{N=}, k={k:.5g}, objective={objective:.3g}")

    output_file = os.path.join("output", design, "figures", f"{N=}_{k=}") + ".png"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    plt.savefig(output_file, dpi=200, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    selected_designs = None
    if len(sys.argv) > 1:
        selected_designs = sys.argv[1:]

    designs = []
    for design in os.listdir("output"):
        if selected_designs and design not in selected_designs:
            continue

        data_folder = os.path.join("output", design, "data")
        if not os.path.isdir(data_folder):
            continue

        for data in os.listdir(data_folder):
            data_path = os.path.join(data_folder, data)
            if not os.path.isfile(data_path):
                continue

            N_str, k_str = data[:-4].split("_")
            N = int(N_str.split("=")[1])
            k = float(k_str.split("=")[1])

            designs.append((design, data_path, N, k))

    if len(designs) > 0:
        for design in tqdm(designs):
            plot_design(*design)
