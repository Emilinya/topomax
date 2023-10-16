from __future__ import annotations
from abc import ABC, abstractmethod

import dolfin as df

from src.filter import Filter
from src.utils import MeshFunctionWrapper
from designs.design_parser import SolverParameters


class Problem(ABC):
    """
    Abstract base class for problems that define the state equation and the
    adjoint equation for topology optimization problems.
    """

    def __init__(self):
        self.is_initialized = False

        self.mesh = None
        self.data = None
        self.filter = None
        self.marker = None
        self.objective = None
        self.domain_size = None
        self.solution_space = None
        self.volume_fraction = None
        self.boundary_conditions = None

    def init(
        self,
        input_filter: Filter,
        mesh: df.Mesh,
        parameters: SolverParameters,
        extra_data,
    ):
        self.is_initialized = True

        self.mesh = mesh
        self.data = extra_data
        self.filter = input_filter
        self.objective = parameters.objective
        self.volume_fraction = parameters.fraction
        self.marker = MeshFunctionWrapper(self.mesh)
        self.domain_size = (parameters.width, parameters.height)

        self.solution_space = self.create_solution_space()
        self.boundary_conditions = self.create_boundary_conditions()

    @abstractmethod
    def calculate_objective_gradient(self) -> df.Function:
        ...

    @abstractmethod
    def calculate_objective(self, rho) -> float:
        ...

    @abstractmethod
    def forward(self, rho) -> df.Function:
        ...

    @abstractmethod
    def create_boundary_conditions(self) -> list:
        ...

    @abstractmethod
    def create_solution_space(self) -> df.FunctionSpace:
        ...
