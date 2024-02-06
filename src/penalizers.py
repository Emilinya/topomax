from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any


class Penalizer(ABC):
    def __init__(self):
        self.penalization: float | None = None

    def set_penalization(self, penalization: float):
        self.penalization = penalization

    def assert_has_penalization(self):
        if self.penalization is None:
            raise ValueError("You must set penalization before calling penalizer")

        # I must return here because the type checker does not understand that
        # self.penalization must be not None after this function call otherwise
        return self.penalization

    @abstractmethod
    def __call__(self, rho: Any) -> Any: ...

    @abstractmethod
    def derivative(self, rho: Any) -> Any: ...


class ElasticPenalizer(Penalizer):
    """Solid isotropic material penalization (SIMP)."""

    def __init__(self):
        super().__init__()
        self.minimum = 1e-6

    def __call__(self, rho):
        self.penalization = self.assert_has_penalization()
        p, m = self.penalization, self.minimum

        return m + rho**p * (1 - m)

    def derivative(self, rho):
        self.penalization = self.assert_has_penalization()
        p, m = self.penalization, self.minimum

        return p * rho ** (p - 1) * (1 - m)


class FluidPenalizer(Penalizer):
    """What is this called?"""

    def __init__(self):
        super().__init__()
        self.minimum = 2.5 / 100**2
        self.maximum = 2.5 / 0.01**2

    def __call__(self, rho):
        self.penalization = self.assert_has_penalization()
        q, mini, maxi = self.penalization, self.minimum, self.maximum

        return maxi + (mini - maxi) * rho * (1 + q) / (rho + q)

    def derivative(self, rho):
        self.penalization = self.assert_has_penalization()
        q, mini, maxi = self.penalization, self.minimum, self.maximum

        return (mini - maxi) * q * (1 + q) / (rho + q) ** 2
