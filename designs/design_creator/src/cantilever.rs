use crate::definitions::*;

pub fn cantilever() -> Design {
    Design::Elasticity(ProblemDesign {
        objective: ElasticityObjective::MinimizeCompliance,
        domain_parameters: DomainParameters {
            width: 3.0,
            height: 1.0,
            volume_fraction: 0.5,
        },
        problem_parameters: ElasticityParameters {
            fixed_sides: vec![Side::Left],
            force_region: Some(BodyForce {
                region: CircularRegion {
                    center: (2.9, 0.5),
                    radius: 0.05,
                },
                value: (0.0, -1.0),
            }),
            tractions: None,
        },
    })
}