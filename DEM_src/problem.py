from __future__ import annotations
from abc import abstractmethod

import torch

from src.problem import Problem
from DEM_src.utils import Mesh
from DEM_src.DeepEnergyMethod import DeepEnergyMethod
from DEM_src.dirichlet_enforcer import DirichletEnforcer
from DEM_src.ObjectiveCalculator import ObjectiveCalculator


class DEMProblem(Problem):
    def __init__(
        self,
        mesh: Mesh,
        device: torch.device,
        verbose: bool,
    ):
        super().__init__()
        self.verbose = verbose
        self.mesh = mesh
        self.device = device

        dirichlet_enforcer, objective_calculator = self.create_dem_parameters()
        self.dem = self.create_dem(dirichlet_enforcer, objective_calculator)

    def set_penalization(self, penalization: float):
        self.dem.objective_calculator.set_penalization(penalization)

    @abstractmethod
    def create_dem_parameters(self) -> tuple[DirichletEnforcer, ObjectiveCalculator]:
        ...

    @abstractmethod
    def create_dem(
        self,
        dirichlet_enforcer: DirichletEnforcer,
        objective_calculator: ObjectiveCalculator,
    ) -> DeepEnergyMethod:
        ...
