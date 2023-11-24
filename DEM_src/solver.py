import warnings

import torch
import numpy as np
import numpy.random as npr
import numpy.typing as npt
from sklearn.preprocessing import normalize
from scipy.sparse import coo_matrix, csr_matrix

from src.solver import Solver
from DEM_src.problem import DEMProblem
from DEM_src.data_structs import Domain
from DEM_src.integrator import integrate
from DEM_src.fluid_problem import FluidProblem
from DEM_src.elasisity_problem import ElasticityProblem
from designs.definitions import FluidDesign, ElasticityDesign


def create_density_filter(radius: float, domain: Domain) -> csr_matrix:
    # we can't use domain.x_grid as it has shape (Nx+1, Ny+1)
    x_ray = np.linspace(0, domain.length, domain.Nx)
    y_ray = np.linspace(0, domain.height, domain.Ny)
    x_grid, y_grid = np.meshgrid(x_ray, y_ray)
    X = x_grid.flatten()
    Y = y_grid.flatten()

    total = domain.Nx * domain.Ny

    wi, wj, wv = [], [], []
    for eid in range(total):
        my_X = X[eid]
        my_Y = Y[eid]

        dist = np.sqrt((X - my_X) ** 2 + (Y - my_Y) ** 2)
        neighbours = np.where(dist <= radius)[0]
        wi += [eid] * len(neighbours)
        wj += list(neighbours)
        wv += list(radius - dist[neighbours])

    W = normalize(
        coo_matrix((wv, (wi, wj)), shape=(total, total)), norm="l1", axis=1
    ).tocsr()  # Normalize row-wise

    return W


class DEMSolver(Solver):
    """Class that solves a given topology optimization problem using a magical algorithm."""

    def __init__(self, N: int, design_file: str, data_path="output", verbose=False):
        warnings.filterwarnings("ignore")
        npr.seed(2022)
        torch.manual_seed(2022)
        np.random.seed(2022)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
            print("CUDA is available, running on GPU")
        else:
            self.device = torch.device("cpu")
            print("CUDA not available, running on CPU")

        self.verbose = verbose
        super().__init__(N, design_file, data_path)

        # why is this neccesary? idk
        self.step_size_multiplier = 0.2

        # convince type checker that problem is indeed a DEM problem
        self.problem: DEMProblem = self.problem

    def get_name(self):
        return "DEM"

    def prepare_domain(self):
        if self.design_str == "bridge":
            Nx = 120
            Ny = 30
        elif self.design_str == "short_cantilever":
            Nx = 90
            Ny = 45
        else:
            Nx = int(self.width * self.N)
            Ny = int(self.height * self.N)

        self.domain = Domain(Nx, Ny, self.width, self.height)

    def create_rho(self, volume_fraction: float):
        return np.ones(self.domain.intervals) * volume_fraction

    def create_problem(self, design: FluidDesign | ElasticityDesign):
        if isinstance(design, FluidDesign):
            return FluidProblem(
                self.domain,
                self.device,
                self.verbose,
                design,
            )
        if isinstance(design, ElasticityDesign):
            control_filter = create_density_filter(0.25, self.domain)
            return ElasticityProblem(
                self.domain,
                self.device,
                self.verbose,
                control_filter,
                design,
            )

        raise ValueError(
            f"Got unknown problem '{self.parameters.problem}' "
            + f"with design of type '{type(design)}'"
        )

    def to_array(self, rho: npt.NDArray):
        return rho

    def set_from_array(self, rho: npt.NDArray, values: npt.NDArray):
        rho[:, :] = values

    def integrate(self, values: npt.NDArray):
        return integrate(values, self.domain)

    def save_rho(self, rho: npt.NDArray, file_root: str):
        np.save(
            f"{file_root}_rho.npy",
            rho.reshape(self.domain.intervals),
        )
