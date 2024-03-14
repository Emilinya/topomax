import os

import torch
import numpy as np
import numpy.typing as npt

from src.solver import Solver
from DEM_src.utils import Mesh
from DEM_src.problem import DEMProblem
from DEM_src.integrator import integrate
from DEM_src.fluid_problem import FluidProblem
from DEM_src.elasisity_problem import ElasticityProblem
from designs.definitions import FluidParameters, ElasticityParameters


class DEMSolver(Solver):
    """Class that solves a given topology optimization problem using a magical algorithm."""

    def __init__(
        self,
        N: int,
        design_file: str,
        data_path="output",
        skip_multiple: int = 1,
        verbose=False,
    ):
        np.random.seed(2022)
        torch.manual_seed(2022)

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            torch.set_default_tensor_type("torch.cuda.FloatTensor")
            print("CUDA is available, running on GPU")
        else:
            self.device = torch.device("cpu")
            print("CUDA not available, running on CPU")

        self.verbose = verbose
        super().__init__(N, design_file, data_path, skip_multiple)

        # convince type checker that problem is indeed a DEM problem
        self.problem: DEMProblem = self.problem

    def get_name(self):
        return "DEM"

    def get_step_size(self):
        return self.parameters.dem_step_size

    def prepare_domain(self):
        Nx = int(self.width * self.N)
        Ny = int(self.height * self.N)

        self.mesh = Mesh(Nx, Ny, self.width, self.height)

    def create_rho(self, volume_fraction: float):
        return np.ones(self.mesh.intervals) * volume_fraction

    def create_problem(
        self, problem_parameters: FluidParameters | ElasticityParameters
    ):
        if isinstance(problem_parameters, FluidParameters):
            return FluidProblem(
                self.mesh,
                self.device,
                self.verbose,
                problem_parameters,
            )
        if isinstance(problem_parameters, ElasticityParameters):
            return ElasticityProblem(
                self.mesh,
                self.device,
                self.verbose,
                problem_parameters,
            )

        raise ValueError(
            f"Got unknown problem '{self.parameters.problem}' "
            + f"with problem parameters of type '{type(problem_parameters)}'"
        )

    def to_array(self, rho: npt.NDArray):
        return rho

    def set_from_array(self, rho: npt.NDArray, values: npt.NDArray):
        rho[:, :] = values

    def integrate(self, values: npt.NDArray):
        return integrate(values, self.mesh)

    def save_rho(self, rho: npt.NDArray, file_root: str):
        rho_file = f"{file_root}_rho.npy"
        np.save(rho_file, rho.reshape(self.mesh.intervals))
        return os.path.basename(rho_file)
