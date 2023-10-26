from dataclasses import dataclass
from numpy import ndarray


@dataclass
class Domain:
    Nx: int
    Ny: int
    x_min: float
    y_min: float
    length: float
    height: float

    @property
    def shape(self):
        return (self.Nx, self.Ny)

    @property
    def dxdy(self):
        return (self.length / (self.Nx - 1), self.height / (self.Ny - 1))

    @property
    def extent(self):
        return (self.x_min, self.length, self.y_min, self.height)


@dataclass
class TopOptParameters:
    E: float
    nu: float
    verbose: bool
    filter_radius: float
    output_folder: str
    max_iterations: int
    volume_fraction: float
    convergence_tolerances: ndarray[float]


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
