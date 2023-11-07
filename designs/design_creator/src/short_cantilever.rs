use crate::definitions::*;

pub fn short_cantilever() -> Design {
    return Design::Elasticity(ProblemDesign {
        objective: ElasticityObjective::MinimizeCompliance,
        domain_parameters: DomainParameters {
            width: 10.0,
            height: 5.0,
            penalties: vec![3.0],
            volume_fraction: 0.4,
        },
        problem_parameters: ElasticityParameters {
            fixed_sides: vec![Side::Left],
            body_force: None,
            tractions: Some(vec![Traction {
                side: Side::Right,
                center: 2.5,
                length: 1.0 / 9.0,
                value: (0.0, -2000.0),
            }]),
            young_modulus: 2e5,
            poisson_ratio: 0.3,
        },
    });
}
