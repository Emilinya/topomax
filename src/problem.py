from __future__ import annotations

from typing import Any
from abc import ABC, abstractmethod


class Problem(ABC):
    """
    Abstract base class for problems that define the state equation and the
    adjoint equation for topology optimization problems.
    """

    @abstractmethod
    def set_penalization(self, penalization: float): ...

    @abstractmethod
    def calculate_objective_gradient(self) -> Any: ...

    @abstractmethod
    def calculate_objective(self, rho: Any) -> float: ...

    @abstractmethod
    def forward(self, rho: Any) -> Any: ...
