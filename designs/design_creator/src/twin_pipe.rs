use crate::definitions::*;

pub fn twin_pipe() -> Design {
    Design::Fluid(ProblemDesign {
        objective: FluidObjective::MinimizePower,
        domain_parameters: DomainParameters {
            width: 1.5,
            height: 1.0,
            fem_step_size: 0.0015,
            dem_step_size: 0.00075,
            penalties: vec![0.01, 0.1],
            volume_fraction: 1.0 / 3.0,
        },
        problem_parameters: FluidParameters {
            flows: vec![
                Flow {
                    side: Side::Left,
                    center: 0.25,
                    length: 1.0 / 6.0,
                    rate: 1.0,
                },
                Flow {
                    side: Side::Left,
                    center: 0.75,
                    length: 1.0 / 6.0,
                    rate: 1.0,
                },
                Flow {
                    side: Side::Right,
                    center: 0.25,
                    length: 1.0 / 6.0,
                    rate: -1.0,
                },
                Flow {
                    side: Side::Right,
                    center: 0.75,
                    length: 1.0 / 6.0,
                    rate: -1.0,
                },
            ],
            no_slip: None,
            zero_pressure: None,
            viscosity: 1.0,
        },
    })
}
