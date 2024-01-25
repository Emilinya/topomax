from typing import Any, Callable
from abc import ABC, abstractmethod

import dolfin as df
from ufl.form import Form


class PDESolver(ABC):
    LeftCallable = Callable[[df.TrialFunction, df.TestFunction, Any | None], Form]
    RightCallable = Callable[[df.TestFunction, Any | None], Form]

    def __init__(
        self,
        a_func: LeftCallable,
        L_func: RightCallable,
        boundary_conditions: list[df.DirichletBC] | None = None,
        function_space: df.FunctionSpace | None = None,
    ):
        self.a_func = a_func
        self.L_func = L_func
        self.function_space = function_space

        if boundary_conditions is None:
            self.boundary_conditions = []
        else:
            self.boundary_conditions = boundary_conditions

    def set_function_space(self, function_space: df.FunctionSpace):
        self.function_space = function_space

    def _get_a_L(self, a_arg: Any | None = None, L_arg: Any | None = None):
        trial = df.TrialFunction(self.function_space)
        test = df.TestFunction(self.function_space)

        a = self.a_func(trial, test, a_arg)
        L = self.L_func(test, L_arg)

        return a, L

    @abstractmethod
    def solve(
        self, *, a_arg: Any | None = None, L_arg: Any | None = None
    ) -> df.Function:
        ...


class DefaultSolver(PDESolver):
    def solve(self, *, a_arg: Any | None = None, L_arg: Any | None = None):
        if self.function_space is None:
            raise ValueError("You must set function space before solving PDE!")

        a, L = self._get_a_L(a_arg, L_arg)
        solution = df.Function(self.function_space)
        df.solve(a == L, solution, bcs=self.boundary_conditions)

        return solution


class SimpleMUMPSSolver(PDESolver):
    def __init__(
        self,
        a_func: PDESolver.LeftCallable,
        L_func: PDESolver.RightCallable,
        boundary_conditions: list[df.DirichletBC] | None = None,
        function_space: df.FunctionSpace | None = None,
    ):
        super().__init__(a_func, L_func, boundary_conditions, function_space)

    def solve(self, *, a_arg: Any | None = None, L_arg: Any | None = None):
        if self.function_space is None:
            raise ValueError("You must set function space before solving PDE!")

        a, L = self._get_a_L(a_arg, L_arg)
        solution = df.Function(self.function_space)
        df.solve(
            a == L,
            solution,
            bcs=self.boundary_conditions,
            solver_parameters={"linear_solver": "mumps"},
        )

        return solution


class SmartMumpsSolver(PDESolver):
    def __init__(
        self,
        a_func: PDESolver.LeftCallable,
        L_func: PDESolver.RightCallable,
        boundary_conditions: list[df.DirichletBC] | None = None,
        function_space: df.FunctionSpace | None = None,
        a_has_no_args: bool = False,
        L_has_no_args: bool = False,
    ):
        super().__init__(a_func, L_func, boundary_conditions, function_space)

        self.A = None
        self.b = None

        if a_has_no_args or L_has_no_args:
            if self.function_space is None:
                raise ValueError(
                    "To precompute A or b, you must set the function space"
                )
            trial = df.TrialFunction(self.function_space)
            test = df.TestFunction(self.function_space)

            if a_has_no_args:
                a = self.a_func(trial, test, None)
                self.A = df.assemble(a)

            if L_has_no_args:
                L = self.L_func(test, None)
                self.b = df.assemble(L)

    def solve(self, *, a_arg: Any | None = None, L_arg: Any | None = None):
        if self.function_space is None:
            raise ValueError("You must set function space before solving PDE!")

        A = self.A
        b = self.b

        if self.A is None or self.b is None:
            trial = df.TrialFunction(self.function_space)
            test = df.TestFunction(self.function_space)

            if self.A is None:
                a = self.a_func(trial, test, a_arg)
                A = df.assemble(a)

            if self.b is None:
                L = self.L_func(test, L_arg)
                b = df.assemble(L)

        _ = [bc.apply(A, b) for bc in self.boundary_conditions]

        solution = df.Function(self.function_space)
        solution_vector = solution.vector()

        solver = df.LUSolver("mumps")
        solver.solve(A, solution_vector, b)

        return solution


class IterativeReuseSolver(PDESolver):
    def __init__(
        self,
        a_func: PDESolver.LeftCallable,
        L_func: PDESolver.RightCallable,
        boundary_conditions: list[df.DirichletBC] | None = None,
        function_space: df.FunctionSpace | None = None,
        reuse_previous_solution: bool = False,
    ):
        super().__init__(a_func, L_func, boundary_conditions, function_space)

        self.reuse_previous_solution = reuse_previous_solution
        self.previous_solution_vector = None

    def solve(self, *, a_arg: Any | None = None, L_arg: Any | None = None):
        if self.function_space is None:
            raise ValueError("You must set function space before solving PDE!")

        a, L = self._get_a_L(a_arg, L_arg)

        A = df.assemble(a)
        b = df.assemble(L)
        _ = [bc.apply(A, b) for bc in self.boundary_conditions]

        solution = df.Function(self.function_space)
        solution_vector = solution.vector()

        if self.reuse_previous_solution and self.previous_solution_vector is not None:
            solution_vector[:] = self.previous_solution_vector

        solver = df.KrylovSolver("cg", "ilu")
        # solver.parameters["relative_tolerance"] = 1e-14
        solver.parameters["nonzero_initial_guess"] = self.reuse_previous_solution
        solver.solve(A, solution_vector, b)

        if self.reuse_previous_solution:
            self.previous_solution_vector = solution_vector[:]

        return solution
