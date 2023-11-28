import numpy as np

from DEM_src.utils import Mesh
from designs.definitions import Side


class SideDomain:
    def __init__(self, mesh: Mesh, side: Side, center: float, length: float):
        left = center - length / 2
        right = center + length / 2

        flat_x = mesh.x_grid.T.flatten()
        flat_y = mesh.y_grid.T.flatten()

        if side == Side.LEFT:
            side_condition = flat_x == 0
            side_points = flat_y
        elif side == Side.RIGHT:
            side_condition = flat_x == mesh.length
            side_points = flat_y
        elif side == Side.TOP:
            side_condition = flat_y == mesh.height
            side_points = flat_x
        elif side == Side.BOTTOM:
            side_condition = flat_y == 0
            side_points = flat_x
        else:
            raise ValueError(f"Unknown side: '{side}'")

        if side in (Side.LEFT, Side.RIGHT):
            self.side_index = 1
            self.stride = 1
            self.width = mesh.Ny + 1
        elif side in (Side.TOP, Side.BOTTOM):
            self.side_index = 0
            self.stride = mesh.Ny + 1
            self.width = mesh.Nx + 1
        self.side = side

        left_condition = side_points >= left
        right_condition = side_points <= right
        (load_indices,) = np.where(side_condition & left_condition & right_condition)

        load_points = np.array([flat_x[load_indices], flat_y[load_indices]]).T

        self.indices = load_indices
        self.left_error = load_points[0, self.side_index] - left
        self.right_error = right - load_points[-1, self.side_index]
