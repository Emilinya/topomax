from dataclasses import dataclass
import numpy.typing as npt
import numpy as np


class Domain:
    def __init__(
        self,
        Nx: int,
        Ny: int,
        x_min: float,
        y_min: float,
        length: float,
        height: float,
    ):
        self.Nx = Nx
        self.Ny = Ny
        self.x_min = x_min
        self.y_min = y_min
        self.length = length
        self.height = height

        # create points
        self.x_ray = np.linspace(self.x_min, self.length, self.Nx)
        self.y_ray = np.linspace(self.y_min, self.height, self.Ny)

        # create an array containing nodal coordinates
        self.coordinates = np.array(
            [
                np.repeat(self.x_ray, len(self.y_ray)),
                np.tile(self.y_ray, len(self.x_ray)),
            ]
        ).T

        self.shape = (self.Ny, self.Nx)
        self.extent = (self.x_min, self.length, self.y_min, self.height)
        self.dxdy = (self.length / (self.Nx - 1), self.height / (self.Ny - 1))


@dataclass
class TopOptParameters:
    E: float
    nu: float
    verbose: bool
    filter_radius: float
    output_folder: str
    max_iterations: int
    volume_fraction: float
    convergence_tolerances: npt.NDArray[np.float64]


@dataclass
class NNParameters:
    input_size: int
    output_size: int
    layer_count: int
    neuron_count: int
    learning_rate: float
    CNN_deviation: float
    rff_deviation: float
    iteration_count: int
    activation_function: str
