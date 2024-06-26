use crate::definitions::*;

pub fn bridge() -> Design {
    Design::Elasticity(ProblemDesign {
        domain_parameters: DomainParameters {
            width: 12.0,
            height: 2.0,
            fem_step_size: 0.001,
            dem_step_size: 0.2,
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
            filter_radius: 0.25 / (2.0 * 3f64.sqrt()),
            young_modulus: 2e5,
            poisson_ratio: 0.3,
        },
    })
}
