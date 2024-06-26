use crate::definitions::*;

pub fn pipe_bend() -> Design {
    Design::Fluid(ProblemDesign {
        domain_parameters: DomainParameters {
            width: 1.0,
            height: 1.0,
            fem_step_size: 0.0015,
            dem_step_size: 0.0001,
            penalties: vec![0.1],
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
            viscosity: 1.0,
        },
    })
}
