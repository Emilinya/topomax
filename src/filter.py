import dolfin as df
import dolfin_adjoint as dfa


class Filter:
    """A class that filters a given design function to remove high frequency elements."""

    def apply(self, input_rho):
        print("Filter can't be used directly, use one on the subclasses")
        exit(1)


class HelmholtzFilter(Filter):
    def __init__(self, N):
        self.epsilon = 0.02

    def apply(self, input_rho):
        # solve -ε²Δξ + ξ = ρ, ∇ξ·n = 0 on ∂Ω
        # where ρ is the input and ξ is the output
        rho_space = input_rho.function_space()
        trial_rho = df.TrialFunction(rho_space)
        test_rho = df.TestFunction(rho_space)

        lhs = (
            self.epsilon**2 * df.inner(df.grad(trial_rho), df.grad(test_rho))
            + df.inner(trial_rho, test_rho)
        ) * df.dx
        rhs = df.inner(input_rho, test_rho) * df.dx

        filtered_rho = dfa.Function(rho_space)
        dfa.solve(lhs == rhs, filtered_rho)

        return filtered_rho
