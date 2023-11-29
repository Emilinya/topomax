import dolfin as df
import numpy.typing as npt

from src.solver import Solver
from FEM_src.problem import FEMProblem
from FEM_src.utils import save_function
from FEM_src.filter import HelmholtzFilter
from FEM_src.fluid_problem import FluidProblem
from FEM_src.elasisity_problem import ElasticityProblem
from designs.definitions import FluidDesign, ElasticityDesign


class FEMSolver(Solver):
    def __init__(
        self,
        N: int,
        design_file: str,
        data_path: str = "output",
        skip_multiple: int = 1,
    ):
        df.set_log_level(df.LogLevel.WARNING)
        # turn off redundant output in parallel
        df.parameters["std_out_all_processes"] = False

        super().__init__(N, design_file, data_path, skip_multiple)

        # convince type checker that problem is indeed a FEM problem
        self.problem: FEMProblem = self.problem

    def get_name(self):
        return "FEM"

    def get_step_size(self):
        return self.parameters.fem_step_size

    def prepare_domain(self):
        self.mesh = df.Mesh(
            df.RectangleMesh(
                df.MPI.comm_world,
                df.Point(0.0, 0.0),
                df.Point(self.width, self.height),
                int(self.width * self.N),
                int(self.height * self.N),
            )
        )

        self.control_space = df.FunctionSpace(self.mesh, "CG", 1)

    def create_rho(self, volume_fraction: float):
        rho = df.Function(self.control_space)
        rho.vector()[:] = volume_fraction

        return rho

    def create_problem(self, design: FluidDesign | ElasticityDesign):
        if isinstance(design, FluidDesign):
            return FluidProblem(self.mesh, design, self.parameters)
        if isinstance(design, ElasticityDesign):
            control_filter = HelmholtzFilter(epsilon=0.02)
            return ElasticityProblem(self.mesh, design, self.parameters, control_filter)

        raise ValueError(
            f"Got unknown problem '{self.parameters.problem}' "
            + f"with design of type '{type(design)}'"
        )

    def to_array(self, rho: df.Function) -> npt.NDArray:
        return rho.vector()[:]

    def set_from_array(self, rho: df.Function, values: npt.NDArray):
        rho.vector()[:] = values

    def integrate(self, values: npt.NDArray):
        integral_func = df.Function(self.control_space)
        integral_func.vector()[:] = values
        return float(df.assemble(integral_func * df.dx))

    def save_rho(self, rho: df.Function, file_root: str):
        rho_file = file_root + "_rho.dat"
        save_function(rho, rho_file, "design")
