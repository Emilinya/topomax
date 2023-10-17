from __future__ import annotations
import json

from designs.definitions import (
    DomainParameters,
    ProblemType,
    FluidDesign,
    ElasticityDesign,
)


def create_design(problem: ProblemType, objective: str, problem_parameters: dict):
    if problem == ProblemType.FLUID:
        return FluidDesign.from_dict(objective, problem_parameters)
    if problem == ProblemType.ELASTICITY:
        return ElasticityDesign.from_dict(objective, problem_parameters)

    raise ValueError(f"Unknown problem: {problem}")


def parse_design(filename: str):
    with open(filename, "rb") as design_file:
        design_dict = json.load(design_file)

    key = list(design_dict.keys())
    if len(key) > 1:
        raise ValueError(f"Malformed design: got more than one key: {key}")
    key = key[0]

    parameters = DomainParameters.from_dict(key, design_dict[key]["domain_parameters"])
    design = create_design(
        parameters.problem,
        design_dict[key]["objective"],
        design_dict[key]["problem_parameters"],
    )

    return parameters, design


if __name__ == "__main__":
    parameters, design = parse_design("designs/pipe_bend.json")
    print(parameters)
    print(design)
