from __future__ import annotations
from abc import ABC, abstractmethod


class Problem(ABC):
    """
    Abstract base class for problems that define the state equation and the
    adjoint equation for topology optimization problems.
    """

    @abstractmethod
    def set_penalization(self, penalization: float):
        ...

    @abstractmethod
    def calculate_objective_gradient(self):
        ...

    @abstractmethod
    def calculate_objective(self, rho) -> float:
        ...

    @abstractmethod
    def forward(self, rho):
        ...
