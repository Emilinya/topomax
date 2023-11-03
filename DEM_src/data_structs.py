from dataclasses import dataclass
import numpy as np


class Domain:
    def __init__(
        self,
        Nx: int,
        Ny: int,
        length: float,
        height: float,
    ):
        self.Nx = Nx
        self.Ny = Ny
        self.length = length
        self.height = height

        # create points
        self.x_ray = np.linspace(0.0, self.length, self.Nx)
        self.y_ray = np.linspace(0.0, self.height, self.Ny)

        # create an array containing nodal coordinates
        self.coordinates = np.array(
            [
                np.repeat(self.x_ray, len(self.y_ray)),
                np.tile(self.y_ray, len(self.x_ray)),
            ]
        ).T

        self.shape = (self.Ny, self.Nx)
        self.dxdy = (self.length / (self.Nx - 1), self.height / (self.Ny - 1))


@dataclass
class NNParameters:
    verbose: bool
    input_size: int
    output_size: int
    layer_count: int
    neuron_count: int
    learning_rate: float
    CNN_deviation: float
    rff_deviation: float
    iteration_count: int
    activation_function: str
    convergence_tolerance: float
