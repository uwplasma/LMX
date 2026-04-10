# LMX Documentation

LMX is a research-facing, JAX-native code for inductionless liquid-metal MHD on
structured meshes. These pages are organized so that a new user can get started
quickly while an advanced user can trace every physical model and numerical
choice back to the implementation files.

## Start here

```{toctree}
:maxdepth: 2
:caption: User Guides

theory
input_reference
case_cookbook
benchmark_matrix
validation_report
external_benchmarks
research_directions
developer_guide
```

## What is implemented today

- fully developed laminar duct solvers
- Hartmann, Shercliff, and Hunt benchmark families
- layered conducting and insulating wall models
- CLI, TOML, restart, plotting, movie generation, and benchmark reporting

## What is planned next

- `extruded_inductionless` for laminar fringing-field benchmarks
- broader benchmark manifests for turbulence, heat transfer, and industrial
  blanket-style configurations
- more inverse and optimization workflows on the differentiable JAX core

## Documentation philosophy

The documentation distinguishes between:

- equations and assumptions
- user-facing inputs and outputs
- developer-facing architecture and file layout
- benchmark evidence and acceptance criteria

External executable comparisons are treated as secondary benchmark evidence, not
as the definition of the governing equations.
