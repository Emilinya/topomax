use crate::definitions::*;

pub fn diffuser() -> Design {
    Design::Fluid(ProblemDesign {
        objective: FluidObjective::MinimizePower,
        domain_parameters: DomainParameters {
            width: 1.0,
            height: 1.0,
            volume_fraction: 0.5,
        },
        problem_parameters: FluidParameters {
            flows: vec![
                Flow {
                    side: Side::Left,
                    center: 0.5,
                    length: 1.0,
                    rate: 1.0,
                },
                Flow {
                    side: Side::Right,
                    center: 0.5,
                    length: 1.0 / 3.0,
                    rate: -3.0,
                },
            ],
            no_slip: None,
            zero_pressure: None,
        },
    })
}
