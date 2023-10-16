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
        self.mesh = None
        self.data = None
        self.filter = None
        self.marker = None
        self.objective = None
        self.domain_size = None
        self.volume_fraction = None

    def init(
        self,
        input_filter: Filter,
        mesh: df.Mesh,
        parameters: SolverParameters,
        extra_data,
    ):
        self.mesh = mesh
        self.data = extra_data
        self.filter = input_filter
        self.objective = parameters.objective
        self.volume_fraction = parameters.fraction
        self.domain_size = (parameters.width, parameters.height)
        self.marker = MeshFunctionWrapper(self.mesh)

        self.create_function_spaces()
        self.create_boundary_conditions()

    @abstractmethod
    def calculate_objective_gradient(self):
        ...

    @abstractmethod
    def calculate_objective(self, rho) -> float:
        ...

    @abstractmethod
    def forward(self, rho):
        ...

    @abstractmethod
    def create_boundary_conditions(self):
        ...

    @abstractmethod
    def create_function_spaces(self):
        ...
