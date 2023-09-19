import dolfin as df

import numpy as np


def elastisity_alpha(rho):
    """Solid isotropic material penalization (SIMP)."""
    alpha_min = 1e-6
    return alpha_min + rho**3 * (1 - alpha_min)


def elastisity_alpha_derivative(rho):
    """SIMP derivative"""
    alpha_min = 1e-6
    return 3 * rho**2 * (1 - alpha_min)


def fluid_alpha(rho):
    """Does this have a name?"""
    q = 0.1
    alpha_min = 2.5 / 100**2
    alpha_max = 2.5 / 0.01**2

    return alpha_min + (alpha_max - alpha_min) * rho * (q + 1) / (q + rho)


def constrain(number, space):
    """Constrain a number so it fits within a given number of characters."""
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


class MeshFunctionWrapper:
    """
    A wrapper around 'df.cpp.mesh.MeshFunctionSizet' that handles
    domain indexes for you.
    """

    def __init__(self, mesh: df.Mesh):
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
