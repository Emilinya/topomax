[![GNU GPLv3 License](https://img.shields.io/github/license/Emilinya/topomax)](https://choosealicense.com/licenses/gpl-3.0/)

# Topomax

This is a topology optimization program based algorithm 8 from the paper *Proximal Galerkin: A structure-preserving finite element method for pointwise bound constraints* by Brendan Keith and Thomas M. Surowiec. It uses design files to define the boundary conditions and other parameters for your optimization problem in an easy-to-read way.

## Usage/Examples

### Dependencies
Topomax depends on FEniCS, which is not available on Windows. The [TopOpt](https://github.com/JohannesHaubner/TopOpt) repository includes a docker image which makes running the program on Windows easy, but as it is not my docker image, I can't guarantee that it will work forever. After [downloading docker](https://www.docker.com/products/docker-desktop/) you can simply run
```bash
docker pull ghcr.io/johanneshaubner/topopt:latest
docker run -it -v "$(pwd):/topomax" -w /topomax ghcr.io/johanneshaubner/topopt
pip install -r requirements.txt
```

Then, in the docker container, you can run the program as normal:
```bash
python3 run.py designs/twin_pipe.json 40
```

This also works if you are using Linux and Mac, but there you can also install FEniCS directly by following [their instalation instructions](https://fenicsproject.org/download/archive/). After installing FEniCS, you can install the rest of the dependencies using the requirements file.

### Running
The program is run using `run.py`. This program takes in two command line arguments; a design file and the number of finite elements in a unit length. The folder `designs` contains some design files, and you can easily make a custom design using those files as a template. If the design is `path/to/design.json`, the output of the program is saved to `output/design/data`. The produced data can be visualized with `plot.py`, which automatically reads all the data files, and produces corresponding figures in `output/design/figures`. `plot.py` can also take a list of designs as an argument to limit which designs it will plot. For instance,
```bash
python3 plot.py design1 design2
```
will only create figures from `output/design1/data` and `output/design2/data`.

## Design file format
The design files are written in json, but they are made by a Rust program. I did this so I could use Rust's rich type system to structure the data. I will therefore explain the design file format using the original Rust types. The root type is

```rust
pub enum Design {
    Fluid(ProblemDesign<FluidObjective, FluidParameters>),
    Elasticity(ProblemDesign<ElasticityObjective, ElasticityParameters>),
}
pub struct ProblemDesign<T, U> {
    pub objective: T,
    pub domain_parameters: DomainParameters,
    pub problem_parameters: U,
}
```
You can either have a fluid design with a fluid objective and fluid parameters, or you can have an elasticity design with an elasticity objective and elasticity parameters. All designs have domain parameters, which are defined as:

```rust
pub struct DomainParameters {
    pub width: f64,
    pub height: f64,
    pub penalties: Vec<f32>,
    pub volume_fraction: f64,
}
```
The width and height is the width and height of your domain, and the volume_fraction is the volume fraction for the volume constraint. The program solves the topology optimization problem
once for each penalty in the penalties list, and uses the result from the previous solution as the initial condition for the next one. This is sometimes necessary in cases where a large penalty is too restrictive.

### Fluid problem
For a fluid problem, the fluid objectives are
```rust
pub enum FluidObjective {
    MinimizePower,
}
```
where 'MinimizePower' means minimizing the total potential power of your fluid. The fluid parameters are:
```rust
pub struct FluidParameters {
    pub flows: Vec<Flow>,
    pub no_slip: Option<Vec<Side>>,
    pub zero_pressure: Option<Vec<Side>>,
}
```
where a side is simply
```rust
pub enum Side {
    Left,
    Right,
    Top,
    Bottom,
}
```
and a flow is defined as
```rust
pub struct Flow {
    pub side: Side,
    pub center: f64,
    pub length: f64,
    pub rate: f64,
}
```
A flow struct represents a parabolic flow out/in from a side, with a value of `rate` at its center. A positive rate represents inflow, and a negative rate represents outflow. Multiple flows can exist on the same side, and the total flow (sum of `length·rate` for all flows) must be 0. The parameter `no_slip` is a list of sides where the fluid velocity is 0. If it is None, it defaults to a list of all sides with no flow. The parameter `zero_pressure` is a list of sides where the fluid pressure is 0. This represents a side where fluid can flow out freely. If it is None, no sides will have zero pressure.

### Elasticity problem
For an elasticity problem, the elasticity objectives are
```rust
pub enum ElasticityObjective {
    MinimizeCompliance,
}
```
where 'MinimizeCompliance' means minimizing the elastic compliance of a material. The elasticity parameters are:
```rust
pub struct ElasticityParameters {
    pub fixed_sides: Vec<Side>,
    pub body_force: Force,
}
```
Where a force is
```rust
pub struct Force {
    pub region: CircularRegion,
    pub value: (f64, f64),
}
pub struct CircularRegion {
    pub center: (f64, f64),
    pub radius: f64,
}
```
and represents a force with a given value that acts on a circle with a given center and radius. The `fixed_sides` parameter represents a side where the material is fixed in place.

## Running Tests
To run tests, run the following command:
```bash
pytest
```

If you are using docker, pytest is not installed by default, so you first need to run
```bash
conda install pytest
```

## License

[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)
