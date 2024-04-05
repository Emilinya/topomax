from __future__ import annotations

import os
from enum import Enum

from src.utils import constrain, prettify_seconds


class ColumnType(Enum):
    TIME = "     Time     "
    ITERATION = "Iteration"
    OBJECTIVE = "Objective"
    DELRA_RHO = "    Δρ    "
    TOLERANCE = "Tolerance"
    TOTAL_TIME = "  Total time  "
    DELTA_OBJECTIVE = "ΔObjective"


class Printer:
    def __init__(self, columns: list[ColumnType]):
        for column_type in ColumnType:
            if not isinstance(column_type, ColumnType):
                raise ValueError(
                    f"Printer got column that is not a ColumnType: '{column_type}'"
                )

        self.columns = columns
        self.titles = [c.value for c in self.columns]
        self.spacings = [len(t) for t in self.titles]

        self.value_map: dict[ColumnType, float | int | str] = {}
        self.total_time = 0.0

    def set_iteration(self, iteration: int):
        self.value_map[ColumnType.ITERATION] = iteration

    def set_objective(self, objective: float):
        previous = self.value_map.get(ColumnType.OBJECTIVE)

        if previous is not None:
            assert not isinstance(previous, str)
            self.value_map[ColumnType.DELTA_OBJECTIVE] = previous - objective

        self.value_map[ColumnType.OBJECTIVE] = objective

    def set_delta_rho(self, delta_rho: float):
        self.value_map[ColumnType.DELRA_RHO] = delta_rho

    def set_tolerance(self, tolerance: float):
        self.value_map[ColumnType.TOLERANCE] = tolerance

    def set_time(self, seconds: float):
        self.total_time += seconds
        self.value_map[ColumnType.TOTAL_TIME] = self.short_time(
            self.total_time, len(ColumnType.TOTAL_TIME.value)
        )
        self.value_map[ColumnType.TIME] = self.short_time(
            seconds, len(ColumnType.TIME.value)
        )

    def set(self, tolerance: float, objective: float, iteration: int, seconds: float):
        self.set_tolerance(tolerance)
        self.set_objective(objective)
        self.set_iteration(iteration)
        self.set_time(seconds)

    def short_time(self, seconds: float, limit: int):
        if limit < 0:
            raise ValueError(f"You can't limit time string to {limit} characters!")

        time = prettify_seconds(seconds)
        while len(time) > limit:
            time = " ".join(time.split(" ")[:-1])
        return time

    def get_reduced_spacings(self):
        try:
            terminal_width = os.get_terminal_size().columns
        except OSError:
            terminal_width = 999

        length = -1
        reduced_spacings: list[int] = []
        for l in self.spacings:
            length += l + 3
            if length > terminal_width:
                break

            reduced_spacings.append(l)

        return reduced_spacings

    def title_length(self):
        reduced_spacings = self.get_reduced_spacings()

        return sum(reduced_spacings) + 3 * (len(reduced_spacings) - 1)

    def print_title(self):
        reduced_spacings = self.get_reduced_spacings()

        print(" │ ".join([f"{t:^{s}}" for t, s in zip(self.titles, reduced_spacings)]))
        print("─┼─".join(["─" * s for s in reduced_spacings]))

    def print_values(self):
        reduced_spacings = self.get_reduced_spacings()
        values = [self.value_map.get(c, "") for c in self.columns]

        print(
            " │ ".join([constrain(v, s) for v, s in zip(values, reduced_spacings)]),
            flush=True,
        )

    def exit(self, exit_condition: str):
        self.print_values()
        print(f"EXIT: {exit_condition}!")
