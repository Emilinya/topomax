use crate::definitions::*;

pub fn triangle() -> Design {
    Design::Elasticity(ProblemDesign {
        objective: ElasticityObjective::MinimizeCompliance,
        domain_parameters: DomainParameters {
            width: 1.0,
            height: 1.0,
            fem_step_size: 25.0,
            dem_step_size: 25.0,
            penalties: vec![3.0],
            volume_fraction: 0.5,
        },
        problem_parameters: ElasticityParameters {
            fixed_sides: vec![Side::Bottom],
            body_force: Some(Force {
                region: CircularRegion {
                    center: (0.5, 0.5),
                    radius: 0.05,
                },
                value: (0.0, -10.0),
            }),
            tractions: None,
            young_modulus: 5.0 / 2.0,
            poisson_ratio: 1.0 / 4.0,
        },
    })
}
