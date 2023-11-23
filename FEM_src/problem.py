from __future__ import annotations
from abc import abstractmethod

import dolfin as df

from FEM_src.utils import MeshFunctionWrapper
from designs.definitions import DomainParameters
from src.penalizers import Penalizer
from src.problem import Problem


class FEMProblem(Problem):
    def __init__(
        self,
        mesh: df.Mesh,
        parameters: DomainParameters,
    ):
        super().__init__()

        self.domain_size = (parameters.width, parameters.height)

        self.mesh = mesh
        self.marker = MeshFunctionWrapper(self.mesh)
        self.solution_space = self.create_solution_space()
        self.boundary_conditions = self.create_boundary_conditions()

        self.penalizer: Penalizer | None = None

    def set_penalization(self, penalization: float):
        if self.penalizer is None:
            raise ValueError(
                "Classes deriving from Problem must "
                + "set a penalizer in their initializer"
            )

        self.penalizer.set_penalization(penalization)

    @abstractmethod
    def create_boundary_conditions(self) -> list:
        ...

    @abstractmethod
    def create_solution_space(self) -> df.FunctionSpace:
        ...
