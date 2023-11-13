from __future__ import annotations
from abc import ABC, abstractmethod

import dolfin as df

from FEM_src.filter import Filter
from FEM_src.utils import MeshFunctionWrapper
from designs.definitions import DomainParameters
from src.penalizers import Penalizer


class Problem(ABC):
    """
    Abstract base class for problems that define the state equation and the
    adjoint equation for topology optimization problems.
    """

    def __init__(
        self,
        input_filter: Filter,
        mesh: df.Mesh,
        parameters: DomainParameters,
    ):
        self.mesh = mesh
        self.penalizer: Penalizer | None = None
        self.filter = input_filter
        self.volume_fraction = parameters.volume_fraction
        self.marker = MeshFunctionWrapper(self.mesh)
        self.domain_size = (parameters.width, parameters.height)

        self.solution_space = self.create_solution_space()
        self.boundary_conditions = self.create_boundary_conditions()

    def set_penalization(self, penalization: float):
        if self.penalizer is None:
            raise ValueError(
                "Classes deriving from Problem must "
                + "set a penalizer in their initializer"
            )

        self.penalizer.set_penalization(penalization)

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
