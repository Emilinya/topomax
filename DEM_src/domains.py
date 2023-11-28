import warnings

import numpy as np
import numpy.typing as npt

from DEM_src.utils import Mesh, distance
from designs.definitions import Side, CircularRegion


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


class CircleDomain:
    def __init__(self, mesh: Mesh, circle: CircularRegion):
        intersections = self.find_intersections(mesh, circle)
        self.indices, self.areas = self.find_areas(mesh, circle, intersections)

        analytial_area = self.get_analytical_area(circle)
        self.area_error = abs(np.sum(self.areas) - analytial_area) / analytial_area
        if abs(np.sum(self.areas) - analytial_area) / analytial_area > 0.1:
            warnings.warn(
                f"CircleDomain has an error of {self.area_error*100:.2g} %, "
                + "you should probably use a finer mesh"
            )

    def get_analytical_area(self, circle: CircularRegion):
        return np.pi * circle.radius**2

    def find_intersections(self, mesh: Mesh, circle: CircularRegion):
        """Find every point where the circle intersects the mesh grid"""
        r2 = circle.radius**2
        cx, cy = circle.center

        intersections: list[tuple[float, float]] = []

        # find all intersections
        for y in mesh.y_grid[:, 0]:
            h2 = (y - cy) ** 2
            if h2 > r2:
                continue

            l = np.sqrt(r2 - h2)
            intersections.append((cx - l, y))
            intersections.append((cx + l, y))

        for x in mesh.x_grid[0, :]:
            h2 = (x - cx) ** 2
            if h2 > r2:
                continue

            l = np.sqrt(r2 - h2)
            intersections.append((x, cy - l))
            intersections.append((x, cy + l))

        if len(intersections) == 0:
            raise ValueError(
                "Tried to precompute circle areas, but circle does not intersect grid!"
            )

        intersection_array: npt.NDArray[np.float64] = np.array(intersections)

        # remove duplicates
        duplicate_idxs: list[int] = []
        for i, p0 in enumerate(intersection_array):
            if i in duplicate_idxs:
                continue

            for j in range(i + 1, len(intersection_array)):
                p1 = intersection_array[j]

                if distance(p0, p1) < 1e-6 * circle.radius:
                    duplicate_idxs.append(j)

        keep_idxs = np.setdiff1d(np.arange(len(intersection_array)), duplicate_idxs)
        intersection_array = intersection_array[keep_idxs]

        return intersection_array

    def to_grid_position(self, mesh: Mesh, point: tuple[float, float] | npt.NDArray):
        px, py = point
        dx, dy = mesh.dxdy

        return (round(px / dx, 12), round(py / dy, 12))

    def to_grid_idx(self, mesh: Mesh, point: tuple[float, float] | npt.NDArray):
        x, y = self.to_grid_position(mesh, point)
        return (int(np.floor(x)), int(np.floor(y)))

    def get_surounding(self, mesh: Mesh, point: tuple[float, float]):
        dx, dy = mesh.dxdy
        fx, fy = self.to_grid_position(mesh, point)

        # We use a set to avoid duplicates. Is this neccesary?
        surounding_set: set[tuple[int, int]] = set()

        surounding_set.add((int(np.ceil(fx)), int(np.ceil(fy))))
        surounding_set.add((int(np.ceil(fx)), int(np.floor(fy))))
        surounding_set.add((int(np.floor(fx)), int(np.ceil(fy))))
        surounding_set.add((int(np.floor(fx)), int(np.floor(fy))))

        surounding_list: list[tuple[float, float]] = []
        for ix, iy in list(surounding_set):
            surounding_list.append((ix * dx, iy * dy))

        return surounding_list

    def find_areas(
        self,
        mesh: Mesh,
        circle: CircularRegion,
        intersection_array: npt.NDArray[np.float64],
    ):
        dx, dy = mesh.dxdy

        circle_center = np.array(circle.center)
        circle_areas = np.full(mesh.intervals, dx * dy)
        circle_idxs: list[tuple[int, int]] = []

        # find index of points with area > 0:
        grid_diagonal = np.sqrt(dx**2 + dy**2)
        center_x = (mesh.x_grid[0, :-1] + mesh.x_grid[0, 1:]) / 2
        center_y = (mesh.y_grid[:-1, 0] + mesh.y_grid[1:, 0]) / 2
        for y in center_y:
            for x in center_x:
                to_center = distance(np.array([x, y]), circle_center)
                if to_center > circle.radius + (grid_diagonal / 2):
                    continue

                surounding = np.array(self.get_surounding(mesh, (x, y)))
                min_to_circle = np.min(distance(surounding, circle_center))
                if min_to_circle - circle.radius < -1e-6 * circle.radius:
                    ix, iy = self.to_grid_idx(mesh, (x, y))
                    circle_idxs.append((iy, ix))

        circle_idx_array = np.array(circle_idxs)

        # sort intersections by angle
        cx, cy = circle_center
        angles = np.angle(
            (intersection_array[:, 0] - cx) + 1j * (intersection_array[:, 1] - cy)
        )
        intersection_array = intersection_array[np.argsort(-angles)]

        # find all areas
        for i, p0 in enumerate(intersection_array):
            p1 = intersection_array[(i + 1) % len(intersection_array)]

            p0ix, p0iy = self.to_grid_position(mesh, p0)
            p1ix, p1iy = self.to_grid_position(mesh, p1)

            area = 0
            center = (p0 + p1) / 2
            index = self.to_grid_idx(mesh, center)
            surounding = np.array(self.get_surounding(mesh, center))

            # we don't want surounding to include p0 or p1
            surounding = surounding[
                np.where(
                    (distance(surounding, p0) > 1e-6 * grid_diagonal)
                    & (distance(surounding, p1) > 1e-6 * grid_diagonal)
                )
            ]

            if (p0ix == int(p0ix) and p1iy == int(p1iy)) or (
                p0iy == int(p0iy) and p1ix == int(p1ix)
            ):
                # we are of type 1 or 2
                corner = surounding[np.argmin(distance(surounding, center))]

                p0c = distance(p0, corner)
                p1c = distance(p1, corner)

                if distance(corner, circle_center) <= circle.radius:
                    # we are of type 1, add triangular area
                    area = 0.5 * p0c * p1c
                else:
                    # we are of type 2, subtract triangular area
                    area = dx * dy - 0.5 * p0c * p1c
            else:
                # we are of type 3, add trapazoidal area
                in_surounding = surounding[
                    np.where(distance(surounding, circle_center) <= circle.radius)
                ]

                c0 = in_surounding[np.argmin(distance(in_surounding, p0))]
                c1 = in_surounding[np.argmin(distance(in_surounding, p1))]

                c0c1 = distance(c0, c1)
                p0c0 = distance(p0, c0)
                p1c1 = distance(p1, c1)

                area = 0.5 * c0c1 * (p0c0 + p1c1)

            circle_areas[index[1], index[0]] = area

        return circle_idx_array, circle_areas[tuple(circle_idx_array.T)]
