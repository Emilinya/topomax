from __future__ import annotations

import pickle
import numpy as np
import dolfin as df


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


def mesh_to_N(mesh: df.Mesh) -> int:
    # hmin is the length of the diagonal of a mesh grid square
    N = 1 / (mesh.hmin() / np.sqrt(2))
    if abs(N - int(round(N))) / N > 1e-14:
        print(
            f"save_function: got non-integer N: {N}, "
            + "this could result in wrong data getting saved"
        )
    return int(round(N))


def mesh_to_domain_size(mesh: df.Mesh) -> tuple[int, int]:
    mesh_coordinates = mesh.coordinates()
    domain_width = mesh_coordinates[:, 0].max()
    domain_height = mesh_coordinates[:, 1].max()
    return (domain_width, domain_height)


def save_function(
    f: df.Function,
    filename: str,
    problem: str,
    N: int | None = None,
    domain_size: tuple[int, int] | None = None,
):
    space = f.function_space()
    mesh = space.mesh()

    if N is None:
        N = mesh_to_N(mesh)

    if domain_size is None:
        domain_size = mesh_to_domain_size(mesh)

    with open(filename, "wb") as datafile:
        data = {
            "N": N,
            "domain_size": domain_size,
            "problem": problem,
            "vector": f.vector()[:],
        }
        pickle.dump(data, datafile)


def load_function(
    filename: str,
    mesh: df.Mesh | None = None,
    function_space: df.FunctionSpace | None = None,
) -> tuple[df.Function, df.Mesh, df.FunctionSpace]:
    with open(filename, "rb") as datafile:
        data = pickle.load(datafile)

    if mesh is None:
        w, h = data["domain_size"]
        mesh = df.Mesh(
            df.RectangleMesh(
                df.MPI.comm_world,
                df.Point(0.0, 0.0),
                df.Point(w, h),
                int(w * data["N"]),
                int(h * data["N"]),
            )
        )

    if function_space is None:
        if data["problem"] == "elasticity":
            displacement_element = df.VectorElement("CG", mesh.ufl_cell(), 2)
            function_space = df.FunctionSpace(mesh, displacement_element)
        elif data["problem"] == "fluid":
            velocity_space = df.VectorElement("CG", mesh.ufl_cell(), 2)
            pressure_space = df.FiniteElement("CG", mesh.ufl_cell(), 1)
            function_space = df.FunctionSpace(mesh, velocity_space * pressure_space)
        elif data["problem"] == "design":
            function_space = df.FunctionSpace(mesh, "CG", 1)
        else:
            raise ValueError(f"load_function got malformed problem: {data['problem']}")

    function = df.Function(function_space)
    function.vector()[:] = data["vector"]

    return (function, mesh, function_space)


def sample_function(
    f: df.Function,
    points: int,
    sample_type: str,
    N: int | None = None,
    domain_size: tuple[int, int] | None = None,
):
    if f.geometric_dimension() != 2:
        raise ValueError(
            f"Unhandled input size: {f.geometric_dimension()}. "
            + "Currently, only an input size of 2 is supported."
        )

    try:
        output_size = len(f(0, 0))
    except TypeError:
        output_size = 1

    if N is None:
        N = mesh_to_N(f.function_space().mesh())

    if domain_size is None:
        domain_size = mesh_to_domain_size(f.function_space().mesh())

    # we want to sample a multiple of N points, choose multiplier
    # so we always sample >= 'points' points
    multiplier = int(np.ceil(points / N))

    domain_samples = [int(s * N * multiplier) for s in domain_size]
    domain_rays = [np.linspace(0, s, Ns) for s, Ns in zip(domain_size, domain_samples)]
    output_grid = np.zeros(domain_samples[::-1] + [output_size])

    for yi in range(domain_samples[1]):
        for xi in range(domain_samples[0]):
            if sample_type == "center":
                x = (0.5 + xi) / (multiplier * N)
                y = (0.5 + yi) / (multiplier * N)
            elif sample_type == "edges":
                x = xi / (multiplier * N - 1 / domain_size[0])
                y = yi / (multiplier * N - 1 / domain_size[1])
            else:
                raise ValueError(
                    f"Unknown sample_type: {sample_type}. "
                    + "sample_type must be either 'center' or 'edges'"
                )
            output_grid[yi, xi, :] = f(x, y)

    return domain_rays, output_grid


def constrain(number: int | float, space: int):
    """
    Constrain a number so it fits within a given number of characters. \n
    Ex: constrain(np.pi, 5) = 3.142, constrain(-1/173, 6) = -5.8e-3.
    """
    try:
        if number == 0:
            return f"{number:.{space - 2}f}"

        is_negative = number < 0
        obj_digits = int(np.log10(abs(number))) + 1
        if obj_digits <= 0:
            return f"{number:.{space - 6 - is_negative}e}"

        return f"{number:.{space - obj_digits - is_negative - 1}f}"
    except Exception:
        # something has gone wrong, but we don't want to raise an excepton
        return "?" * space
