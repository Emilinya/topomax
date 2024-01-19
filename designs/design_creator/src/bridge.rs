use crate::definitions::*;

pub fn bridge() -> Design {
    return Design::Elasticity(ProblemDesign {
        objective: ElasticityObjective::MinimizeCompliance,
        domain_parameters: DomainParameters {
            width: 12.0,
            height: 2.0,
            fem_step_size: 0.001,
            dem_step_size: 0.4,
            penalties: vec![3.0],
            volume_fraction: 0.4,
        },
        problem_parameters: ElasticityParameters {
            fixed_sides: vec![Side::Left, Side::Right],
            body_force: None,
            tractions: Some(vec![Traction {
                side: Side::Top,
                center: 6.0,
                length: 0.5,
                value: (0.0, -2000.0),
            }]),
            fem_filter_radius: 0.074,
            dem_filter_radius: 0.25,
            young_modulus: 2e5,
            poisson_ratio: 0.3,
        },
    });
}
