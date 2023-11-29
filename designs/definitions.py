from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto


# I want this to work with n-tuples, but how
# do I then make the type hinting happy?
def to_2_tuple(ray: list[float]):
    """
    turn a list of 2 values into a tuple of 2 values,
    raises ValueError if length is not 2.
    """
    if len(ray) != 2:
        raise ValueError(
            "Got array that should have had 2 elements, "
            + f"but had {len(ray)} instead: '{ray}'"
        )

    return (ray[0], ray[1])


class Side(Enum):
    LEFT = auto()
    RIGHT = auto()
    TOP = auto()
    BOTTOM = auto()

    @classmethod
    def from_string(cls, string: str):
        if string == "Left":
            return cls.LEFT
        if string == "Right":
            return cls.RIGHT
        if string == "Top":
            return cls.TOP
        if string == "Bottom":
            return cls.BOTTOM

        raise ValueError(
            f"Malformed side: '{string}'\nLegal sides "
            + "are: 'Left', ''Right', 'Top' or 'Bottom'"
        )


class ElasticityObjective(Enum):
    MINIMIZE_COMPLIANCE = auto()

    @classmethod
    def from_string(cls, objective: str):
        if objective == "MinimizeCompliance":
            return cls.MINIMIZE_COMPLIANCE

        raise ValueError(
            f"Malformed objective: '{objective}'\nLegal objective "
            + "is: 'MinimizeCompliance'"
        )


class FluidObjective(Enum):
    MINIMIZE_POWER = auto()

    @classmethod
    def from_string(cls, objective: str):
        if objective == "MinimizePower":
            return cls.MINIMIZE_POWER

        raise ValueError(
            f"Malformed objective: '{objective}'\nLegal objective "
            + "is: 'MinimizePower'"
        )


class ProblemType(Enum):
    FLUID = auto()
    ELASTICITY = auto()

    @classmethod
    def from_string(cls, problem: str):
        if problem == "Fluid":
            return cls.FLUID
        if problem == "Elasticity":
            return cls.ELASTICITY

        raise ValueError(
            f"Malformed problem: '{problem}'\nLegal problems "
            + "are: 'Fluid' and 'Elasticity'"
        )


@dataclass
class SquareRegion:
    center: tuple[float, float]
    size: tuple[float, float]

    @classmethod
    def from_dict(cls, region_dict: dict):
        return cls(to_2_tuple(region_dict["center"]), to_2_tuple(region_dict["size"]))


@dataclass
class CircularRegion:
    center: tuple[float, float]
    radius: float

    @classmethod
    def from_dict(cls, region_dict: dict):
        return cls(to_2_tuple(region_dict["center"]), region_dict["radius"])


@dataclass
class Flow:
    side: Side
    center: float
    length: float
    rate: float

    @classmethod
    def from_dict(cls, flow_dict: dict):
        return cls(
            Side.from_string(flow_dict["side"]),
            flow_dict["center"],
            flow_dict["length"],
            flow_dict["rate"],
        )

    def to_tuple(self):
        return (self.side, self.center, self.length, self.rate)


@dataclass
class FluidParameters:
    flows: list[Flow]
    no_slip: list[Side] | None
    zero_pressure: list[Side] | None
    max_region: SquareRegion | None
    viscosity: float

    @classmethod
    def from_dict(cls, parameter_dict: dict):
        flows = [Flow.from_dict(flow_dict) for flow_dict in parameter_dict["flows"]]

        no_slip = None
        if parameter_dict.get("no_slip") is not None:
            no_slip = [Side.from_string(side) for side in parameter_dict["no_slip"]]

        zero_pressure = None
        if parameter_dict.get("zero_pressure") is not None:
            zero_pressure = [
                Side.from_string(side) for side in parameter_dict["zero_pressure"]
            ]

        max_region = None
        if parameter_dict.get("max_region") is not None:
            max_region = SquareRegion.from_dict(parameter_dict["max_region"])

        return cls(
            flows, no_slip, zero_pressure, max_region, parameter_dict["viscosity"]
        )


@dataclass
class Force:
    region: CircularRegion
    value: tuple[float, float]

    @classmethod
    def from_dict(cls, force_dict: dict):
        return cls(
            CircularRegion.from_dict(force_dict["region"]),
            to_2_tuple(force_dict["value"]),
        )


@dataclass
class Traction:
    side: Side
    center: float
    length: float
    value: tuple[float, float]

    @classmethod
    def from_dict(cls, traction_dict: dict):
        return cls(
            Side.from_string(traction_dict["side"]),
            traction_dict["center"],
            traction_dict["length"],
            to_2_tuple(traction_dict["value"]),
        )

    def to_tuple(self):
        return (self.side, self.center, self.length, self.value)


@dataclass
class ElasticityParameters:
    fixed_sides: list[Side]
    body_force: Force | None
    tractions: list[Traction] | None
    young_modulus: float
    poisson_ratio: float

    @classmethod
    def from_dict(cls, parameter_dict: dict):
        fixed_sides = [Side.from_string(side) for side in parameter_dict["fixed_sides"]]

        body_force = None
        if parameter_dict.get("body_force") is not None:
            body_force = Force.from_dict(parameter_dict["body_force"])

        tractions = None
        if parameter_dict.get("tractions") is not None:
            tractions = [
                Traction.from_dict(traction_dict)
                for traction_dict in parameter_dict["tractions"]
            ]

        return cls(
            fixed_sides,
            body_force,
            tractions,
            parameter_dict["young_modulus"],
            parameter_dict["poisson_ratio"],
        )


@dataclass
class DomainParameters:
    width: float
    height: float
    problem: ProblemType
    fem_step_size: float
    dem_step_size: float
    penalties: list[float]
    volume_fraction: float

    @classmethod
    def from_dict(cls, problem: str, parameter_dict: dict):
        return cls(
            parameter_dict["width"],
            parameter_dict["height"],
            ProblemType.from_string(problem),
            parameter_dict["fem_step_size"],
            parameter_dict["dem_step_size"],
            parameter_dict["penalties"],
            parameter_dict["volume_fraction"],
        )


@dataclass
class FluidDesign:
    objective: FluidObjective
    parameters: FluidParameters

    @classmethod
    def from_dict(cls, objective: str, parameter_dict: dict):
        return cls(
            FluidObjective.from_string(objective),
            FluidParameters.from_dict(parameter_dict),
        )


@dataclass
class ElasticityDesign:
    objective: ElasticityObjective
    parameters: ElasticityParameters

    @classmethod
    def from_dict(cls, objective: str, parameter_dict: dict):
        return cls(
            ElasticityObjective.from_string(objective),
            ElasticityParameters.from_dict(parameter_dict),
        )
