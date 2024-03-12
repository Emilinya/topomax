import torch
import numpy as np

from DEM_src.utils import Mesh, flatten
from DEM_src.objective_calculator import ObjectiveCalculator
from src.penalizers import ElasticPenalizer
from tests.utils import get_convergance

class DummyObjective(ObjectiveCalculator):
    def value(self, u, grad_u):
        return [torch.sum(u**2, 0) + torch.sum(grad_u**2, [0, 1])]

    def calculate_energy(
        self, u: torch.Tensor, shape: tuple[int, int], density: torch.Tensor
    ):
        (value,) = self.evaluate(u, shape, self.value)

        return torch.sum(value * self.detJ)

    def calculate_objective_and_gradient(
        self, u: torch.Tensor, shape: tuple[int, int], density: torch.Tensor
    ):
        pass


def linear(x_grid, y_grid):
    ux = x_grid + y_grid
    uy = x_grid - y_grid

    return [ux, uy]


def linear_analytic(mesh: Mesh):
    w, h = mesh.width, mesh.height

    return 2 * w * h * (2 + (w**2 + h**2) / 3)


def trig(x_grid, y_grid):
    ux = np.cos(2 * y_grid) * np.sin(x_grid)
    uy = np.cos(y_grid) * np.sin(2 * x_grid)

    return [ux, uy]


def trig_analytic(mesh: Mesh):
    w, h = mesh.width, mesh.height

    t1 = 24 * w * h - 4 * h * np.sin(2 * w) + h * np.sin(4 * w) + 4 * w * np.sin(2 * h)
    t2 = (np.sin(2 * w) - w) * np.sin(4 * h) + np.sin(4 * w) * np.sin(2 * h)

    return (t1 + t2) / 8


def compare(f, f_analytic, mesh: Mesh, objective: ObjectiveCalculator):
    u = torch.from_numpy(flatten(f(mesh.x_grid, mesh.y_grid))).float()

    numeric = float(
        objective.calculate_energy(u, mesh.shape, torch.ones_like(u))
    )
    analytic = f_analytic(mesh)

    return abs(analytic - numeric)


def test_evaluate():
    def linear_errfunc(N: int):
        mesh = Mesh(2 * N, 3 * N, 5, 2)
        objective = DummyObjective(mesh, ElasticPenalizer())
        return compare(linear, linear_analytic, mesh, objective)

    def trig_errfunc(N: int):
        mesh = Mesh(2 * N, 3 * N, 5, 2)
        objective = DummyObjective(mesh, ElasticPenalizer())
        return compare(trig, trig_analytic, mesh, objective)

    Ns = list(range(5, 100, 5))
    assert get_convergance(Ns, linear_errfunc) < -2
    assert get_convergance(Ns, trig_errfunc) < -2
