from designs.design_parser import Side, parse_design
import pytest

def test_parse_design():
    # test elasticity

    parameters, force_region, fixed_sides, traction = parse_design(
        "tests/test_designs/elasticity_design.json"
    )
    assert parameters.width == 3
    assert parameters.height == 1
    assert parameters.objective == "minimize_compliance"

    assert force_region.radius == 0.05
    assert force_region.center == (2.9, 0.5)
    assert force_region.value == (0, -1)

    assert len(fixed_sides) == 1
    assert fixed_sides[0] == Side.LEFT

    assert traction == (0, 0)

    # test fluids

    parameters, flows, no_slip, zero_pressure, max_region = parse_design(
        "tests/test_designs/fluid_design.json"
    )

    assert parameters.width == 1.5
    assert parameters.height == 1
    assert abs(parameters.fraction - 1 / 3) < 1e-12
    assert parameters.objective == "minimize_power"

    assert len(flows) == 4
    assert flows[0].center == 0.25
    assert flows[1].side == Side.LEFT
    assert (flows[2].length - 1 / 6) < 1e-12
    assert flows[3].rate == -1

    assert no_slip == None
    assert zero_pressure == None
    assert max_region == None

    with pytest.raises(ValueError):
        parse_design("tests/test_designs/broken_design.json")
