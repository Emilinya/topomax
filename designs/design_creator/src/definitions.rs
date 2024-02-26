use serde::Serialize;

#[derive(Serialize)]
pub enum Side {
    Left,
    Right,
    Top,
    Bottom,
}

#[derive(Serialize)]
pub struct CircularRegion {
    pub center: (f64, f64),
    pub radius: f64,
}

#[derive(Serialize)]
pub enum ElasticityObjective {
    MinimizeCompliance,
}

#[derive(Serialize)]
pub enum FluidObjective {
    MinimizePower,
}

#[derive(Serialize)]
pub struct Flow {
    pub side: Side,
    pub center: f64,
    pub length: f64,
    pub rate: f64,
}

#[derive(Serialize)]
pub struct FluidParameters {
    pub flows: Vec<Flow>,
    pub no_slip: Option<Vec<Side>>,
    pub zero_pressure: Option<Vec<Side>>,
    pub viscosity: f64,
}

#[derive(Serialize)]
pub struct Force {
    pub region: CircularRegion,
    pub value: (f64, f64),
}

#[derive(Serialize)]
pub struct Traction {
    pub side: Side,
    pub center: f64,
    pub length: f64,
    pub value: (f64, f64),
}

#[derive(Serialize)]
pub struct ElasticityParameters {
    pub fixed_sides: Vec<Side>,
    pub body_force: Option<Force>,
    pub tractions: Option<Vec<Traction>>,
    pub filter_radius: f64,
    pub young_modulus: f64,
    pub poisson_ratio: f64,
}

#[derive(Serialize)]
pub struct DomainParameters {
    pub width: f64,
    pub height: f64,
    pub fem_step_size: f64,
    pub dem_step_size: f64,
    pub penalties: Vec<f32>,
    pub volume_fraction: f64,
}

#[derive(Serialize)]
pub struct ProblemDesign<T, U> {
    pub objective: T,
    pub domain_parameters: DomainParameters,
    pub problem_parameters: U,
}

#[derive(Serialize)]
pub enum Design {
    Fluid(ProblemDesign<FluidObjective, FluidParameters>),
    Elasticity(ProblemDesign<ElasticityObjective, ElasticityParameters>),
}
