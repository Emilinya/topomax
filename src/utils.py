from __future__ import annotations
import time

import numpy as np


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

    def __exit__(self, exc_type, exc_val, exc_tb):
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


def get_print_spacings(strings: list[str]):
    lengths = [len(s) for s in strings]
    max_length = max(lengths)

    # make max_length even
    max_length += max_length % 2

    return [max_length - (l % 2) for l in lengths]


def print_title(titles: list[str], spacings: list[int]):
    print(" │ ".join([f"{t:^{s}}" for t, s in zip(titles, spacings)]))
    print("─┼─".join(["─" * s for s in spacings]))


def print_values(values: list[int | float], spacings: list[int]):
    print(
        " │ ".join([constrain(v, s) for v, s in zip(values, spacings)]),
        flush=True,
    )


def constrain(value: str | int | float, space: int):
    """
    Constrain a value so it fits within a given value of characters.

    Examples
    --------
    >>> constrain(np.pi, 1)
    '3'
    >>> constrain(np.pi, 2)
    ' 3'
    >>> constrain(np.pi, 3)
    '3.1'
    >>> constrain(-1 / 173, 6)
    '-6e-03'
    >>> constrain(-1 / 173, 5)
    '-0.00'
    >>> constrain(5.0, 4)
    '5.00'
    >>> constrain(5, 4)
    ' 5  '
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

            if abs(value) > 1:
                # It's too big
                if space > 3:
                    return "Big" + "g" * (space - 3)
                return "Big"[:space]

            # It's too small
            str_val = str(value)
            if len(str_val) > space:
                return str_val[:space]
            return str_val + "0" * (len(str_val) - space)

    padding = ""
    if space - wasted_space == 0:
        padding = " "

    if space - wasted_space == -1:
        # we don't want decimals -> no point
        wasted_space -= 1

    if use_exp:
        return padding + f"{value:.{space - wasted_space}e}"

    return padding + f"{value:.{space - wasted_space}f}"
