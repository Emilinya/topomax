use crate::definitions::*;

pub fn dem_cantilever() -> Design {
    Design::Elasticity(ProblemDesign {
        objective: ElasticityObjective::MinimizeCompliance,
        domain_parameters: DomainParameters {
            width: 15.0,
            height: 5.0,
            fem_step_size: 0.01,
            dem_step_size: 2.5,
            penalties: vec![3.0],
            volume_fraction: 0.5,
        },
        problem_parameters: ElasticityParameters {
            fixed_sides: vec![Side::Left],
            body_force: Some(Force {
                region: CircularRegion {
                    center: (14.5, 2.5),
                    radius: 0.25,
                },
                value: (0.0, -2000.0),
            }),
            tractions: None,
            fem_filter_radius: 0.006,
            dem_filter_radius: 0.02,
            young_modulus: 2e5,
            poisson_ratio: 0.3,
        },
    })
}
