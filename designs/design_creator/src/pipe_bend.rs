use crate::definitions::*;

pub fn pipe_bend() -> Design {
    Design::Fluid(ProblemDesign {
        objective: FluidObjective::MinimizePower,
        domain_parameters: DomainParameters {
            width: 1.0,
            height: 1.0,
            volume_fraction: 0.251,
        },
        problem_parameters: FluidParameters {
            flows: vec![
                Flow {
                    side: Side::Left,
                    center: 0.8,
                    length: 0.2,
                    rate: 1.0,
                },
                Flow {
                    side: Side::Bottom,
                    center: 0.8,
                    length: 0.2,
                    rate: -1.0,
                },
            ],
            no_slip: None,
            zero_pressure: None,
            max_region: None,
        },
    })
}
