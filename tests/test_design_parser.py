from designs.definitions import Side, FluidParameters, ElasticityParameters
from designs.design_parser import parse_design
import pytest


def test_parse_design():
    # test elasticity

    domain_parameters, problem_parameters = parse_design(
        "tests/test_designs/elasticity_design.json"
    )

    assert isinstance(problem_parameters, ElasticityParameters)

    assert domain_parameters.width == 3
    assert domain_parameters.height == 1
    assert domain_parameters.penalties == [3]

    assert problem_parameters.body_force is not None
    assert problem_parameters.body_force.region.radius == 0.05
    assert problem_parameters.body_force.region.center == (2.9, 0.5)
    assert problem_parameters.body_force.value == (0, -1)

    assert len(problem_parameters.fixed_sides) == 1
    assert problem_parameters.fixed_sides[0] == Side.LEFT

    assert problem_parameters.tractions is None

    # test fluids

    domain_parameters, problem_parameters = parse_design(
        "tests/test_designs/fluid_design.json"
    )

    assert isinstance(problem_parameters, FluidParameters)

    assert domain_parameters.width == 1.5
    assert domain_parameters.height == 1
    assert domain_parameters.penalties == [0.01, 0.1]
    assert domain_parameters.volume_fraction == 1 / 3

    assert len(problem_parameters.flows) == 4
    assert problem_parameters.flows[0].center == 0.25
    assert problem_parameters.flows[1].side == Side.LEFT
    assert problem_parameters.flows[2].length == 1 / 6
    assert problem_parameters.flows[3].rate == -1

    with pytest.raises(ValueError):
        parse_design("tests/test_designs/broken_design.json")
