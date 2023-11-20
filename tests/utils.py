from typing import Callable

import numpy as np


def get_convergance(Ns: list[int], error_func: Callable[[int], float]) -> float:
    errors = []
    for N in Ns:
        errors.append(error_func(N))

    poly = np.polynomial.Polynomial.fit(np.log(Ns), np.log(errors), 1)
    return poly.coef[1]
