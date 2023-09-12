import dolfin as df
import dolfin_adjoint as dfa

import numpy as np
from designs.design_parser import Region


class SidesDomain(df.SubDomain):
    def __init__(self, domain_size: tuple[float, float], sides: list[str]):
        super().__init__()
        self.domain_size = domain_size
        self.sides = sides

    def inside(self, pos, on_boundary):
        if not on_boundary:
            return False

        for side in self.sides:
            if side == "left" and df.near(pos[0], 0.0):
                return True
            elif side == "right" and df.near(pos[0], self.domain_size[0]):
                return True
            elif side == "top" and df.near(pos[1], self.domain_size[1]):
                return True
            elif side == "bottom" and df.near(pos[1], 0.0):
                return True
        return False


class RegionDomain(df.SubDomain):
    def __init__(self, region: Region):
        super().__init__()
        cx, cy = region.center
        w, h = region.size
        self.x_region = (cx - w / 2, cx + w / 2)
        self.y_region = (cy - h / 2, cy + h / 2)

    def inside(self, pos, _):
        return df.between(pos[0], self.x_region) and df.between(pos[1], self.y_region)


class PointDomain(df.SubDomain):
    def __init__(self, point: tuple[float, float]):
        super().__init__()
        self.point = point

    def inside(self, pos, _):
        return df.near(pos[0], self.point[0]) and df.near(pos[1], self.point[1])


class MeshFunctionWrapper:
    """
    A wrapper around 'df.cpp.mesh.MeshFunctionSizet' that handles
    domain indexes for you.
    """

    def __init__(self, mesh: dfa.Mesh):
        self.mesh_function = df.cpp.mesh.MeshFunctionSizet(mesh, 1)
        self.mesh_function.set_all(0)
        self.label_to_idx: dict[str, int] = {}
        self.idx = 1

    def add(self, sub_domain: df.SubDomain, label: str):
        sub_domain.mark(self.mesh_function, self.idx)
        self.label_to_idx[label] = self.idx
        self.idx += 1

    def get(self, label: str):
        return (self.mesh_function, self.label_to_idx[label])


def constrain(number, space):
    """Constrain a number so it fits within a given number of characters"""
    try:
        if number == 0:
            return f"{number:.{space - 1}f}"

        is_negative = number < 0
        obj_digits = int(np.log10(abs(number))) + 1
        if obj_digits <= 0:
            return f"{number:.{space - 6 - is_negative}e}"

        return f"{number:.{space - obj_digits - is_negative - 1}f}"
    except:
        # It would be stupid if the printing code crashed the optimizer,
        # wrap it in a try except just in case
        return "?" * space
