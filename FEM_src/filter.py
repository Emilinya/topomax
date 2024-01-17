from __future__ import annotations

import dolfin as df

from FEM_src.pde_solver import SmartMumpsSolver


class HelmholtzFilter:
    """
    A filter that computes the Helmholtz filter equation `-ε²Δξ + ξ = ρ on Ω,
    ∇ξ·n = 0 on ∂Ω`, where ρ is the input and ξ is the output. This PDE smooths
    the input, removing small features.
    """

    def __init__(self, epsilon: float, function_space: df.FunctionSpace):
        """
        This init precomputes values to save time when the filter is used
        repeatedly. Updating epsilon or the function space therefore requires
        creating a new filter.
        """

        """
        The weak form of the filter PDE is (ϵ²∇ξ, ∇v) + (ξ, v) = (ρ, v),
        so a = (ϵ²∇ξ, ∇v) + (ξ, v), L = (ρ, v)
        """

        def a_func(trial, test, _):
            return (
                epsilon**2 * df.inner(df.grad(trial), df.grad(test))
                + df.inner(trial, test)
            ) * df.dx

        def L_func(test, input_function):
            return df.inner(input_function, test) * df.dx

        self.solver = SmartMumpsSolver(
            a_func, L_func, function_space=function_space, a_has_no_args=True
        )

    def apply(self, input_function: df.Function):
        return self.solver.solve(L_arg=input_function)
