use serde::Serialize;

#[derive(Serialize)]
pub enum Side {
    Left,
    Right,
    Top,
    Bottom,
}

#[derive(Serialize)]
pub struct SquareRegion {
    pub center: (f64, f64),
    pub size: (f64, f64),
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
    MaximizeFlow,
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
    pub max_region: Option<SquareRegion>,
}

#[derive(Serialize)]
pub struct BodyForce {
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
    pub force_region: Option<BodyForce>,
    pub tractions: Option<Vec<Traction>>,
}

#[derive(Serialize)]
pub struct DomainParameters {
    pub width: f64,
    pub height: f64,
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
