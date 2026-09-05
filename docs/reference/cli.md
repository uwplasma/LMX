# CLI and TOML reference

LMX accepts a TOML case directly or one of three commands:

```console
lmx CASE.toml
lmx run CASE [OPTIONS]
lmx validate CASE [OPTIONS]
lmx benchmark [OPTIONS]
```

`run` accepts `hartmann`, `shercliff`, `hunt`, `fringing_rect`,
`fringing_layered`, and `fringing_pipe`. Use `lmx run --help` for geometry,
field-transition, resolution, output, and logging controls.

```console
lmx run fringing_rect --ha 20 --nx-stations 21 --ny 24 --nz 24 \
  --length 6 --entry-center 1.5 --exit-center 4.5 --plots
```

`validate` solves a Hartmann, Shercliff, or Hunt case and writes profiles and
validation metrics. `benchmark` reports cold time, warm median, and warm
coefficient of variation for a bounded Hartmann case; it is a local performance
diagnostic, not a hardware-independent performance claim.

## TOML sections

The schema maps directly to the Python dataclasses:

| Section | Python object | Purpose |
|---|---|---|
| `[case]` | `CaseSpec` | name, forcing, initial state, reference gradient |
| `[geometry]` | `GeometrySpec` | kind, dimensions, mesh resolution, wall resolution |
| `[[regions]]` | `RegionSpec` | fluid/solid density, viscosity, conductivity |
| `[magnetic_field]` | `MagneticFieldSpec` | constant, analytic, or tabulated field |
| `[[boundary_conditions]]` | `BoundaryCondition` | velocity, pressure, current, and wall conditions |
| `[solver]` | `SolverConfig` | model, mode, coupling, SOLVAX selection |
| `[time_stepper]` | `TimeStepperConfig` | step sizes, iteration limits, physical tolerances |
| `[output]` | `OutputSpec` | NPZ, JSON, VTK, CSV, plots, and diagnostic-history stride |
| `[fringing]` | `FringingSpec` | axial entry/exit envelope |

Start from `examples/hartmann_case.toml`. Unknown keys, inconsistent geometry,
nonphysical material values, and unsupported solver combinations fail during
configuration rather than entering the numerical solve.

For extruded flow, `[solver].extruded_formulation` explicitly selects
`"stokes_projection"` (default), `"b1_finite_volume"` (ALEX pipe), or
`"b2_finite_volume"` (ALEX layered duct). Benchmark builders set their own
formulation. Names are labels and do not select equations. The B1/B2 options
retain the benchmark-specific boundary, material and numerical contracts;
selecting one is not certification for arbitrary research conditions.
Restart metadata must match the selected formulation; unspecified formulation
metadata is accepted only for `stokes_projection`.
