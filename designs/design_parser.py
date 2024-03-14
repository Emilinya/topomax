from __future__ import annotations
import json

from designs.definitions import (
    DomainParameters,
    ProblemType,
    FluidParameters,
    ElasticityParameters,
)


def parse_design(filename: str):
    with open(filename, "rb") as design_file:
        design_dict = json.load(design_file)

    key = list(design_dict.keys())
    if len(key) > 1:
        raise ValueError(f"Malformed design: got more than one key: {key}")
    key = key[0]

    domain_parameters = DomainParameters.from_dict(
        key, design_dict[key]["domain_parameters"]
    )
    problem = domain_parameters.problem
    parameters = design_dict[key]["problem_parameters"]

    if problem == ProblemType.FLUID:
        problem_parameters = FluidParameters.from_dict(parameters)
    elif problem == ProblemType.ELASTICITY:
        problem_parameters = ElasticityParameters.from_dict(parameters)
    else:
        raise ValueError(f"Unknown problem: {problem}")

    return domain_parameters, problem_parameters


if __name__ == "__main__":

    def main():
        domain_parameters, problem_parameters = parse_design("designs/pipe_bend.json")
        print(domain_parameters)
        print(problem_parameters)

    main()
