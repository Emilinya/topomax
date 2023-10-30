import os
import time

import numpy as np
import matplotlib.pyplot as plt


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


def smart_savefig(filename, **kwargs):
    """
    This is a wrapper around plt.savefig that creates the
    directory the file is getting saved in if it does not exist.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    plt.savefig(filename, **kwargs)


def constrain(number: int | float, space: int):
    """
    Constrain a number so it fits within a given number of characters. \n
    Ex: constrain(np.pi, 5) = 3.142, constrain(-1/173, 6) = -5.8e-3.
    """
    try:
        if number == 0:
            return f"{number:.{space - 2}f}"

        is_negative = number < 0
        obj_digits = int(np.log10(abs(number))) + 1
        if obj_digits <= 0:
            return f"{number:.{space - 6 - is_negative}e}"

        return f"{number:.{space - obj_digits - is_negative - 1}f}"
    except Exception as e:
        # something has gone wrong, but we don't want to raise an excepton
        return "?" * space
