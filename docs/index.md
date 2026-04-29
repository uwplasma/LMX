# LMX Documentation

LMX is a JAX-native code for inductionless liquid-metal magnetohydrodynamics on
structured meshes. The project is organized so that:

- a new user can install it quickly, run a benchmark case, and inspect output
- an advanced user can trace every equation and numerical choice back to the
  source files
- a researcher can reproduce figures, scaling studies, validation reports, and
  differentiable inverse-design workflows from the repository directly

## Highlights

- fully developed laminar duct solvers for Hartmann, Shercliff, and Hunt flows
- explicit multi-region conductivity treatment for layered duct walls
- restartable CLI and TOML workflows
- detailed runtime logging with initial/final residuals and conservation checks
- strong-scaling benchmark tooling for CPU and GPU kernels
- differentiable benchmark and inverse-design workflows in JAX
- geometry preview and postprocessing utilities
- 3D fringing-field workflows on rectangular ducts, layered ducts, and mapped
  pipes

## Read this first

```{toctree}
:maxdepth: 2
:caption: Core Guides

getting_started
theory
numerics
geometry
input_reference
case_cookbook
testing
```

## Validation, performance, and research workflows

```{toctree}
:maxdepth: 2
:caption: Validation and Research

benchmark_matrix
validation_report
media
closure_notes
performance
autodiff
fringing
external_benchmarks
executable_external_code_audit
research_directions
```

## Developer and maintenance notes

```{toctree}
:maxdepth: 2
:caption: Developer Material

developer_guide
```

## What is implemented today

- `fully_developed_inductionless`
  - the default duct solver family
- structured `rect_duct` and `layered_duct` cross-sections
- mapped `pipe_ogrid` geometry/preview tooling
- strong-scaling kernel benchmarks
- differentiable Hartmann sensitivity and inverse-design workflows
- fringing-field benchmark staging through stacked axial field bundles

## What is next

- the first true `extruded_inductionless` 3D pressure-velocity-potential slice
- deeper heavy-lane solver hardening on larger validation sets
- differentiable workflows on Hunt/fringing objectives
- broader variable-field and variable-geometry research studies driven from Python
