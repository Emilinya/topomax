use crate::definitions::*;

pub fn fem_short_cantilever() -> Design {
    return Design::Elasticity(ProblemDesign {
        objective: ElasticityObjective::MinimizeCompliance,
        domain_parameters: DomainParameters {
            width: 2.0,
            height: 1.0,
            fem_step_size: 5.0,
            dem_step_size: 2000.0,
            penalties: vec![3.0],
            volume_fraction: 0.4,
        },
        problem_parameters: ElasticityParameters {
            fixed_sides: vec![Side::Left],
            body_force: None,
            tractions: Some(vec![Traction {
                side: Side::Right,
                center: 0.5,
                length: 1.0 / 45.0,
                value: (0.0, -1.0),
            }]),
            young_modulus: 1.0,
            poisson_ratio: 0.3,
        },
    });
}
