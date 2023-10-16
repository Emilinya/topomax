import dolfin as df

from src.domains import SidesDomain
from src.utils import MeshFunctionWrapper
from src.penalizers import FluidPenalizer
from src.fluid_problem import BoundaryFlows
from designs.design_parser import Flow, Side


def contains_same_elements(list1: list, list2: list):
    no_extra = set(list1).difference(list2) == set()
    no_missing = set(list2).difference(list1) == set()
    return no_extra and no_missing


def solve_equation(function_space, boundary_conditions):
    v, q = df.TestFunctions(function_space)
    u, p = df.TrialFunctions(function_space)
    equation = (
        FluidPenalizer.eval(1.0) * df.inner(u, v)
        + df.inner(df.grad(u), df.grad(v))
        + df.inner(df.grad(p), v)
        + df.inner(df.div(u), q)
    ) * df.dx

    uh = df.Function(function_space)
    df.solve(df.lhs(equation) == df.rhs(equation), uh, bcs=boundary_conditions)
    return uh


def simple_bc(function_space, boundary_flows: BoundaryFlows):
    return [df.DirichletBC(function_space.sub(0), boundary_flows, "on_boundary")]


def marker_bc(marker, function_space, boundary_flows: BoundaryFlows):
    flow_sides = [flow.side for flow in boundary_flows.flows]
    assert contains_same_elements(flow_sides, [Side.LEFT, Side.RIGHT])

    all_sides = [Side.LEFT, Side.RIGHT, Side.TOP, Side.BOTTOM]
    no_slip_sides = list(set(all_sides).difference(flow_sides))
    assert contains_same_elements(no_slip_sides, [Side.TOP, Side.BOTTOM])

    marker.add(SidesDomain(boundary_flows.domain_size, flow_sides), "flow")
    marker.add(SidesDomain(boundary_flows.domain_size, no_slip_sides), "no_slip")

    return [
        df.DirichletBC(function_space.sub(0), boundary_flows, *marker.get("flow")),
        df.DirichletBC(
            function_space.sub(0), df.Constant((0.0, 0.0)), *marker.get("no_slip")
        ),
    ]


def test_marker():
    # create mesh
    N = 20
    domain_size = (1.5, 1.0)
    mesh = df.Mesh(
        df.RectangleMesh(
            df.MPI.comm_world,
            df.Point(0.0, 0.0),
            df.Point(domain_size[0], domain_size[1]),
            int(domain_size[0] * N),
            int(domain_size[1] * N),
        )
    )

    # create function spaces
    velocity_space = df.VectorElement("CG", mesh.ufl_cell(), 2)
    pressure_space = df.FiniteElement("CG", mesh.ufl_cell(), 1)
    function_space = df.FunctionSpace(mesh, velocity_space * pressure_space)

    # create boundary flows
    flows = [
        Flow(Side.LEFT, 0.25, 1 / 6, 1),
        Flow(Side.LEFT, 0.75, 1 / 6, 1),
        Flow(Side.RIGHT, 0.25, 1 / 6, -1),
        Flow(Side.RIGHT, 0.75, 1 / 6, -1),
    ]
    boundary_flows = BoundaryFlows(domain_size, flows)

    # simple boundary condition without using marker
    solution_simple = solve_equation(
        function_space, simple_bc(function_space, boundary_flows)
    )
    (u_simple, _) = df.split(solution_simple)

    # the same boundary condition, now using marker
    marker = MeshFunctionWrapper(mesh)
    solution_marker = solve_equation(
        function_space, marker_bc(marker, function_space, boundary_flows)
    )
    (u_marker, _) = df.split(solution_marker)

    assert df.assemble((u_simple - u_marker) ** 2 * df.dx) < 1e-14
