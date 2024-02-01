from __future__ import annotations
from typing import Callable
import time

from scipy import optimize


class Timer:
    """
    A timer that makes timing things nicer
    """

    def __init__(self, task: str | None = None):
        self.task = task
        if not self.task is None:
            self.task = self.task.capitalize()

        self.start_time = time.time()

    def __enter__(self):
        return self

    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        if self.task is None:
            print(self.get_time_string())
        else:
            print(f"{self.task} took {self.get_time_string()}")

    def restart(self):
        self.start_time = time.time()

    def get_time_seconds(self):
        return time.time() - self.start_time

    def get_time_string(self):
        seconds = self.get_time_seconds()
        return Timer.prettify_seconds(seconds)

    @classmethod
    def prettify_seconds(cls, seconds):
        whole_seconds = int(seconds)
        milliseconds = (seconds - whole_seconds) * 1000

        if whole_seconds == 0:
            return f"{milliseconds:.3g}ms"
        if whole_seconds < 60:
            return f"{whole_seconds:d}s {milliseconds:.0f}ms"
        if seconds < 60**2:
            whole_minutes = int(whole_seconds / 60)
            whole_seconds = whole_seconds % 60
            return f"{whole_minutes:d}m {whole_seconds:d}s {milliseconds:.0f}ms"

        # more than one hour
        whole_hours = int(whole_seconds / (60**2))
        whole_seconds = whole_seconds % (60**2)

        whole_minutes = int(whole_seconds / 60)
        whole_seconds = whole_seconds % 60

        return (
            f"{whole_hours:d}h {whole_minutes:d}m {whole_seconds:d}s {milliseconds:d}ms"
        )


def typeify_optimize(optimize_output):
    """
    The functions in scipy.optimize can return a lot of different values depending
    on their inputs. This is annoying from a typing perspective, so this utility
    function asserts that the output is a simple `tuple[float, RootResults]`
    """

    assert isinstance(optimize_output, tuple) and len(optimize_output) == 2
    assert isinstance(optimize_output[1], optimize.RootResults)
    assert isinstance(optimize_output[0], float)

    return optimize_output[0], optimize_output[1]


def smart_brentq(f: Callable[[float], float], initial_radius: float, max_radius: float):
    """
    Function that uses optimize.brentq to find a root of f. It will
    try a range [-r, r], where r starts as initial_radius. If f does
    not change sign in that range, r will be doubled. This repeats until
    either f changes sign in the range, or r > max_radius. In the latter case,
    a ValueError will be thrown.
    """

    r = initial_radius
    while True:
        if r > max_radius:
            raise ValueError(
                "f(-max_radius) and f(max_radius) must have different signs!"
            )

        try:
            return typeify_optimize(optimize.brentq(f, -r, r, full_output=True))
        except ValueError:
            r *= 2


def constrain(value: str | int | float, space: int):
    """
    Constrain a value so it fits within a given number of characters.

    Examples
    --------
    >>> constrain(np.pi, 5)
    '3.142'
    >>> constrain(-1 / 173, 8)
    '-5.8e-03'
    >>> constrain(5, 4)
    ' 5  '
    >>> constrain('hi', 6)
    '  hi  '
    >>> constrain(np.pi, 1)
    '3'
    >>> constrain(np.pi, 2)
    '3 '
    >>> constrain(np.pi, 3)
    '3.1'
    >>> constrain(-1 / 173, 5)
    '-0.00'
    >>> constrain(5.0, 4)
    '5.00'
    >>> constrain(5555555555, 6)
    ' 6e+09'
    >>> constrain(5555555555, 4)
    'Biggg'
    >>> constrain('hi', 6)
    '  hi  '
    >>> constrain('funny', 2)
    'fu'
    """

    if isinstance(value, str):
        if len(value) > space:
            return value[:space]
        return f"{value:^{space}}"

    if isinstance(value, int):
        if len(str(value)) <= space:
            return f"{value:^{space}}"
        value = float(value)

    wasted_space = str(value).find(".") + 1

    use_exp = False
    if space - wasted_space < -1 or abs(value) < 1e-2:
        use_exp = True
        wasted_space = 6 + (value < 0)

        if space - wasted_space < -1:
            # we can't have a negative amount of decimals

            if abs(value) < 1e-2:
                # It's too small
                str_val = str(value)
                if len(str_val) > space:
                    return str_val[:space]
                return str_val + "0" * (len(str_val) - space)

            # It's too big
            if space > 3:
                return "Big" + "g" * (space - 3)
            return "Big"[:space]

    padding = ""
    if space - wasted_space == 0:
        padding = " "

    if space - wasted_space == -1:
        # we don't want decimals -> no point
        wasted_space -= 1

    if use_exp:
        return padding + f"{value:.{space - wasted_space}e}"

    return f"{value:.{space - wasted_space}f}" + padding
