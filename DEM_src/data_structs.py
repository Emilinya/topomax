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
        x_ray = np.linspace(0.0, self.length, self.Nx + 1)
        y_ray = np.linspace(0.0, self.height, self.Ny + 1)

        self.x_grid, self.y_grid = np.meshgrid(x_ray, y_ray)

        self.intervals = (self.Ny, self.Nx)
        self.shape = (self.Ny + 1, self.Nx + 1)
        self.dxdy = (self.length / self.Nx, self.height / self.Ny)
