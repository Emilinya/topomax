from typing import Callable

import numpy as np

PLOT_ERRORS = False

if PLOT_ERRORS:
    import matplotlib.pyplot as plt


def get_convergance(Ns: list[int], error_func: Callable[[int], float]) -> float:
    errors = []
    for N in Ns:
        errors.append(error_func(N))

    poly = np.polynomial.Polynomial.fit(np.log(Ns), np.log(errors), 1)

    if PLOT_ERRORS:
        plt.rcParams.update({"font.size": 14})
        plt.plot(np.log(Ns), np.log(errors), "go--")
        plt.xlabel(r"$\ln(N)$")
        plt.ylabel(r"$\ln(\text{error})$")
        plt.text(
            0.55,
            0.55,
            r"$\frac{\Delta y}{\Delta x}=" + f"{poly.coef[1]:.3f}" + "$",
            fontsize=20,
            transform=plt.gca().transAxes,
        )
        plt.savefig("temp.png", dpi=200, bbox_inches="tight")
        plt.clf()

    return poly.coef[1]


def get_average(Ns: list[int], error_func: Callable[[int], float]) -> float:
    errors = []
    for N in Ns:
        errors.append(error_func(N))

    return float(np.mean(errors))
