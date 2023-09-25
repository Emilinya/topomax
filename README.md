[![GNU GPLv3 License](https://img.shields.io/github/license/Emilinya/topomax)](https://choosealicense.com/licenses/gpl-3.0/)

# Topomax

This is a topology optimization program based algorithm 8 from the paper *Proximal Galerkin: A structure-preserving finite element method for pointwise bound constraints* by Brendan Keith and Thomas M. Surowiec. It uses design files to define the boundary conditions and other parameters for your optimization problem in an easy-to-read way.

## Usage/Examples

### Running
The program is run using `run.py`. This program takes in two command line arguments; a design file and the number of finite elements in a unit length. The folder `designs` contains some design files, and you can easily make a custom design using those files as a template. If the design is `path/to/design.json`, the output of the program is saved to `output/design/data`. The produced data can be visualized with `plot.py`, which automatically reads all the data files, and produces corresponding figures in `output/design/figures`. `plot.py` can also take a list of designs as an argument to limit which designs it will plot. For instance,
```bash
python3 plot.py design1 design2
```
will only create figures from `output/design1/data` and `output/design2/data`.

### Docker
Topomax uses fenics-dolfin, which is not available on Windows. The [TopOpt](https://github.com/JohannesHaubner/TopOpt) repository includes a docker image which makes running the program on Windows easy, but as it is not my docker image, I can't guarantee that it will work forever. To use the docker image, simply run

```bash
docker pull ghcr.io/johanneshaubner/topopt:latest
docker run -it -v $(pwd):/topomax ghcr.io/johanneshaubner/topopt
```

Then, in the docker container, you can run the program as normal:
```bash
cd topomax
python3 run.py designs/twin_pipe.json 40
```

## Design file format
The design files are written in json, and the settings are:
1. objective \
    Allowed values: "minimize_compliance", "minimize_power" or "maximize_flow". \
    Description: Decides the objective function to be minimized.
2. width \
    Allowed values: `float` \
    Description: The width of the domain
3. height \
    Allowed values: `float` \
    Description: The height of the domain
4. fraction \
    Allowed values: `float` \
    Description: The fraction of the domain that is allowed to be empty, the volume fraction.

The options for elasticity are

5. fixed_sides \
    Allowed values: list with values "left", "right", "top", or "bottom". \
    Description: The sides of the domain that are fixed, that is, the sides with no displacement.

6. force_region \
    Allowed values: \
    - radius: `float`
    - center: (`float`, `float`)
    - value: (`float`, `float`)
    Description: The value and domain of the body force. It is zero everywhere else.

The options for fluids (currently unimplemented) are

8. flows \
    Allowed values: list of flows, where a flow has the following values:
    - side: "left", "right", "top", or "bottom"
    - center: `float`
    - length: `float`
    - rate: `float`

    Description: Defines the boundary conditions on the velocity field. Each flow describes a parabolic flow pattern. A positive rate indicates flow into the domain, and a negative flow pattern indicates flow out of the domain.
9. max_region \
    Allowed values: 
    - center: `(float, float)`
    - size: `(float, float)`

    Description: The region where you want to maximize flow. Mandatory for the "maximize_flow" objective, does nothing for the "minimize_power" objective. The desired flow direction is currently hard coded to (-1, 0).
10. no_slip (optional) \
    Allowed values: List containing values of "left", "right", "top", or "bottom" \
    Description: The sides where there is no flow (velocity is 0). Defaults to all sides with no defined flow.
11. zero_pressure (optional) \
    Allowed values: List containing values of "left", "right", "top", or "bottom" \
    Description: The sides where there is no pressure. If not set, pressure is 0 at (0, 0).

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
