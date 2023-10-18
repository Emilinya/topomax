from designs.definitions import (
    Side,
    ElasticityDesign,
    ElasticityObjective,
    FluidDesign,
    FluidObjective,
)
from designs.design_parser import parse_design
import pytest


def test_parse_design():
    # test elasticity

    parameters, design = parse_design("tests/test_designs/elasticity_design.json")

    assert isinstance(design, ElasticityDesign)

    assert parameters.width == 3
    assert parameters.height == 1
    assert parameters.penalties == [3]
    assert design.objective == ElasticityObjective.MINIMIZE_COMPLIANCE

    assert design.parameters.body_force is not None
    assert design.parameters.body_force.region.radius == 0.05
    assert design.parameters.body_force.region.center == (2.9, 0.5)
    assert design.parameters.body_force.value == (0, -1)

    assert len(design.parameters.fixed_sides) == 1
    assert design.parameters.fixed_sides[0] == Side.LEFT

    assert design.parameters.tractions is None

    # test fluids

    parameters, design = parse_design("tests/test_designs/fluid_design.json")

    assert isinstance(design, FluidDesign)

    assert parameters.width == 1.5
    assert parameters.height == 1
    assert parameters.penalties == [0.01, 0.1]
    assert parameters.volume_fraction == 1 / 3
    assert design.objective == FluidObjective.MINIMIZE_POWER

    assert len(design.parameters.flows) == 4
    assert design.parameters.flows[0].center == 0.25
    assert design.parameters.flows[1].side == Side.LEFT
    assert design.parameters.flows[2].length == 1 / 6
    assert design.parameters.flows[3].rate == -1

    assert design.parameters.no_slip is None
    assert design.parameters.zero_pressure is None
    assert design.parameters.max_region is None

    with pytest.raises(ValueError):
        parse_design("tests/test_designs/broken_design.json")
