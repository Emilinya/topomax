use crate::definitions::*;

pub fn bridge() -> Design {
    return Design::Elasticity(ProblemDesign {
        objective: ElasticityObjective::MinimizeCompliance,
        domain_parameters: DomainParameters {
            width: 6.0,
            height: 1.0,
            fem_step_size: 0.02,
            dem_step_size: 80.0,
            penalties: vec![3.0],
            volume_fraction: 0.4,
        },
        problem_parameters: ElasticityParameters {
            fixed_sides: vec![Side::Left, Side::Right],
            body_force: None,
            tractions: Some(vec![Traction {
                side: Side::Top,
                center: 3.0,
                length: 0.25,
                value: (0.0, -1.0),
            }]),
            young_modulus: 1.0,
            poisson_ratio: 0.3,
        },
    });
}
