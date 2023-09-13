import dolfin_adjoint as dfa

from src.filter import Filter
from designs.design_parser import SolverParameters
from src.utils import MeshFunctionWrapper


class Problem:
    def init(
        self, control_filter: Filter, mesh: dfa.Mesh, parameters: SolverParameters
    ):
        self.volume_fraction = parameters.fraction
        self.control_filter = control_filter
        self.domain_size = (parameters.width, parameters.height)
        self.marker = MeshFunctionWrapper(mesh)

        self.create_function_spaces(mesh)
        self.create_boundary_conditions()
        self.create_rho()

        return self.get_rho(), self.create_objective()

    def get_rho(self):
        print("Problem can't be used directly, use one on the subclasses")
        exit(1)

    def create_objective(self):
        print("Problem can't be used directly, use one on the subclasses")
        exit(1)

    def create_function_spaces(self, mesh: dfa.Mesh):
        print("Problem can't be used directly, use one on the subclasses")
        exit(1)

    def create_boundary_conditions(self):
        print("Problem can't be used directly, use one on the subclasses")
        exit(1)

    def create_rho(self):
        print("Problem can't be used directly, use one on the subclasses")
        exit(1)
