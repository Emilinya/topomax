[![GNU GPLv3 License](https://img.shields.io/github/license/Emilinya/topomax)](https://choosealicense.com/licenses/gpl-3.0/)

# Topomax

This is a topology optimization program based on the entropic mirror descent algorithm described in the paper *Proximal Galerkin: A structure-preserving finite element method for pointwise bound constraints* by Brendan Keith and Thomas M. Surowiec. It uses design files to define the boundary conditions and other parameters for your optimization problem in an easy-to-read way. Topomax was made for the paper *Comparing Classical and Machine Learning Based Approaches to Topology Optimization* by Emilie Dørum, and as such, it implements both the finite element method and the deep energy method. The DEM implementation is based on the paper *Deep energy method in topology optimization applications* by Junyan He, Charul Chadha, Shashank Kushwaha, Seid Koric, Diab Abueidda and Iwona Jasiuk, where I have heavily rewritten the [source code they provided](https://github.com/Jasiuk-Research-Group/DeepEnergy-TopOpt), as well as fixed some of their bugs.

## Usage

### Dependencies
Topomax depends on FEniCS, which is not available on Windows. The [TopOpt](https://github.com/JohannesHaubner/TopOpt) repository includes a docker image which makes running the program on Windows easy, but as it is not my docker image, I can't guarantee that it will work forever. After [downloading docker](https://www.docker.com/products/docker-desktop/) you can simply run
```bash
docker pull ghcr.io/johanneshaubner/topopt:latest
docker run -it -v "$(pwd):/topomax" -w /topomax ghcr.io/johanneshaubner/topopt
```

Then, in the docker container, you can run the program as normal:
```bash
pip install -r requirements.txt
python3 run.py designs/diffuser.json 40
```

This also works if you are using Linux and Mac, but there you can also install FEniCS directly by following [their instalation instructions](https://fenicsproject.org/download/archive/). After installing FEniCS, you can install the rest of the dependencies using the requirements file.

### Running
The program is run using `run.py`. This program takes in two command line arguments; a design file and the number of finite elements along the shortest length. The folder `designs` contains some design files, and you can easily make a custom design using those files as a template. If the design is `path/to/design.json`, the output of the program is saved to `output/design/data`. The program also takes in some optional arguments, which you can see by running `run.py -h`. The produced data can be visualized with `plot.py`, which automatically reads all the data files, and produces corresponding figures in `output/design/figures`. You can limit what data is plotted by using some optional command line arguments which are listed when you run `plot.py -h`.

## Design file format
The design files are written in JSON, but they are made by a Rust program. I did this so I could use Rust's rich type system to structure the data. I will therefore explain the design file format using the original Rust types. The root type is

```rust
enum Design {
    Fluid(ProblemDesign<FluidParameters>),
    Elasticity(ProblemDesign<ElasticityParameters>),
}
struct ProblemDesign<T> {
    domain_parameters: DomainParameters,
    problem_parameters: T,
}
```
You can either have a fluid design with a fluid objective and fluid parameters, or you can have an elasticity design with an elasticity objective and elasticity parameters. All designs have domain parameters, which are defined as:

```rust
struct DomainParameters {
    width: f64,
    height: f64,
    fem_step_size: f64,
    dem_step_size: f64,
    penalties: Vec<f32>,
    volume_fraction: f64,
}
```
The width and height is the width and height of your domain, and the volume_fraction is the volume fraction for the volume constraint. The penalties are a list of either $p$-values for elasticity or $q$-values for fluids. The program will use the finished design for the previous penalty as the initial design for the next penalty, which allows for iterative refinement of a design as seen in the twin pipe example. The two step-size values must be found manually by testing several values and using the best one.

### Fluid problem
For a fluid problem, the fluid parameters are:
```rust
struct FluidParameters {
    flows: Vec<Flow>,
    viscosity: f64,
}
```
where a flow is defined as
```rust
struct Flow {
    side: Side,
    center: f64,
    length: f64,
    rate: f64,
}
```
and a side is simply
```rust
enum Side {
    Left,
    Right,
    Top,
    Bottom,
}
```
A flow struct represents a parabolic flow out/in from a side, with a value of `rate` at its center. A positive rate represents inflow, and a negative rate represents outflow. Multiple flows can exist on the same side, and the total flow (sum of `length·rate` for all flows) must be 0. For sides with no defined flow, the flow is assumed to be zero.

### Elasticity problem
For an elasticity problem, the elasticity parameters are:
```rust
struct ElasticityParameters {
    fixed_sides: Vec<Side>,
    body_force: Option<Force>,
    tractions: Option<Vec<Traction>>,
    filter_radius: f64,
    young_modulus: f64,
    poisson_ratio: f64,
}
```
where a force is
```rust
struct Force {
    region: CircularRegion,
    value: (f64, f64),
}
struct CircularRegion {
    center: (f64, f64),
    radius: f64,
}
```
and a traction is
```rust
struct Traction {
    side: Side,
    center: f64,
    length: f64,
    value: (f64, f64),
}
```
The force represents a body force with a given value that acts on a circle with a given center and radius, the traction represents a traction applied to a portion of one of the boundary sides, and the `fixed_sides` parameter represents a side where the material is fixed in place.
The filter radius is the radius for the Helmholtz filter.

## Running Tests
This program uses pytest for testing, so running them is simply done by running
```bash
pytest tests
```

The requirements file does not include pytest, so you first need to run
```bash
pip install pytest
```

## License

[GNU GPLv3](https://choosealicense.com/licenses/gpl-3.0/)
