use crate::definitions::*;

pub fn cantilever() -> Design {
    Design::Elasticity(ProblemDesign {
        domain_parameters: DomainParameters {
            width: 3.0,
            height: 1.0,
            fem_step_size: 25.0,
            dem_step_size: 37_500.0,
            penalties: vec![3.0],
            volume_fraction: 0.5,
        },
        problem_parameters: ElasticityParameters {
            fixed_sides: vec![Side::Left],
            body_force: Some(Force {
                region: CircularRegion {
                    center: (2.9, 0.5),
                    radius: 0.05,
                },
                value: (0.0, -1.0),
            }),
            tractions: None,
            filter_radius: 0.02,
            young_modulus: 5.0 / 2.0,
            poisson_ratio: 1.0 / 4.0,
        },
    })
}
