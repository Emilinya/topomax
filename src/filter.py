import dolfin as df


class Filter:
    """A class that filters a given design function to remove high frequency elements."""

    def apply(self, input_rho):
        print("Filter can't be used directly, use one on the subclasses")
        exit(1)


class HelmholtzFilter(Filter):
    def __init__(self, N):
        self.epsilon = 0.02

    def apply(self, input_funcion, function_space=None):
        # solve -ε²Δξ + ξ = ρ, ∇ξ·n = 0 on ∂Ω
        # where ρ is the input and ξ is the output

        if function_space is None:
            function_space = input_funcion.function_space()
        trial_function = df.TrialFunction(function_space)
        test_function = df.TestFunction(function_space)

        lhs = (
            self.epsilon**2 * df.inner(df.grad(trial_function), df.grad(test_function))
            + df.inner(trial_function, test_function)
        ) * df.dx
        rhs = df.inner(input_funcion, test_function) * df.dx

        filtered_rho = df.Function(function_space)
        df.solve(lhs == rhs, filtered_rho)

        return filtered_rho
