from __future__ import annotations
from abc import ABC, abstractmethod

import dolfin as df


class Filter(ABC):
    """A class that filters a given design function to remove high frequency elements."""

    @abstractmethod
    def apply(self, input_function):
        ...


class HelmholtzFilter(Filter):
    """
    A filter what uses the equation -ε²Δξ + ξ = ρ,
    where ρ is the input and ξ is the output, to filter a function
    """

    def __init__(self, epsilon: float | None = None, N: int | None = None):
        """
        Create a HelmholtzFilter with a given epsilon value. If epsilon is None,
        epsilon is instead set to 4 / N. If N is also None, a ValueError is raised.
        """

        if epsilon is not None:
            self.epsilon = epsilon
        elif N is not None:
            self.epsilon = 4 / N
        else:
            raise ValueError(
                "Tried to create HelmholtzFilter with neither epsilon nor N!"
            )

    def apply(self, input_function, function_space=None):
        """
        solve -ε²Δξ + ξ = ρ, ∇ξ·n = 0 on ∂Ω,
        where ρ is the input and ξ is the output,
        using the weak form (ϵ²∇ξ, ∇v) + (ξ, v) = (ρ, v) ∀v
        """

        if function_space is None:
            function_space = input_function.function_space()
        trial = df.TrialFunction(function_space)
        test = df.TestFunction(function_space)

        lhs = (
            self.epsilon**2 * df.inner(df.grad(trial), df.grad(test))
            + df.inner(trial, test)
        ) * df.dx
        rhs = df.inner(input_function, test) * df.dx

        filtered_function = df.Function(function_space)
        df.solve(lhs == rhs, filtered_function)

        return filtered_function
