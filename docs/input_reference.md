# Input Reference

LMX runs from a single TOML file:

```bash
lmx examples/hartmann_case.toml
```

## Top-level blocks

- `[case]`
- `[geometry]`
- `[materials]`
- `[magnetic_field]`
- `[solver]`
- `[time_stepper]`
- `[output]`
- `[logging]`
- `[restart]`

## Minimal example

```toml
[case]
name = "hartmann_ha20"
kind = "hartmann"

[geometry]
kind = "rect_duct"
width = 2.0
height = 2.0
ny = 48
nz = 48
target_ha = 20.0

[materials]
fluid_density = 1.0
fluid_viscosity = 1.0
fluid_conductivity = 1.0

[magnetic_field]
kind = "constant"
value = [0.0, 0.0, 1.0]

[solver]
kind = "fully_developed_inductionless"
mode = "steady"
linear_solver = "auto"
preconditioner = "jacobi"
time_scheme = "implicit_euler"

[time_stepper]
dt = 1e-3
t_final = 0.1
max_steps = 500
steady_tolerance = 1e-8

[output]
directory = "artifacts/hartmann"
write_npz = true
write_vtk = true
write_plots = true
write_restart = true

[logging]
verbosity = "detailed"
write_log_file = true
```

## `[case]`

- `name`
  - case identifier used in output files
- `kind`
  - `hartmann`
  - `shercliff`
  - `hunt`

## `[geometry]`

- `kind`
  - `rect_duct`
  - `layered_duct`
- `width`, `height`
  - fluid duct dimensions
- `ny`, `nz`
  - cross-sectional resolution
- `target_ha`
  - Hartmann number used by the benchmark constructors
- layered-duct inputs
  - wall thicknesses
  - wall conductivity or conductance-ratio controls

## `[materials]`

- `fluid_density`
- `fluid_viscosity`
- `fluid_conductivity`
- layered-region properties when solids are present

## `[magnetic_field]`

- `kind`
  - `constant`
  - `analytic`
- `value`
  - constant field components
- `ramp_start`
- `ramp_duration`

## `[solver]`

- `kind`
  - `fully_developed_inductionless`
  - `legacy_reduced`
  - `extruded_inductionless` (reserved)
- `mode`
  - `steady`
  - `transient`
- `linear_solver`
  - `auto`
  - `cg`
  - `gmres`
  - `bicgstab`
- `preconditioner`
  - `none`
  - `jacobi`
  - `block_jacobi`
- `time_scheme`
  - `implicit_euler`
  - `crank_nicolson` reserved for future transient families

## `[time_stepper]`

- `dt`
- `t_final`
- `max_steps`
- `steady_tolerance`
- `steady_potential_tolerance`
- legacy-only controls
  - retained for `solver.kind = "legacy_reduced"`
  - intentionally not part of the default research path

## `[output]`

- `directory`
- `write_npz`
- `write_restart`
- `write_vtk`
- `write_plots`
- `copy_input_file`
- `field_stride`
- `probe_stride`
- `slice_stride`

## `[logging]`

- `verbosity`
  - `quiet`
  - `normal`
  - `detailed`
  - `debug`
- `write_log_file`

## `[restart]`

- `enabled`
- `input_npz`
- `append_histories`
- `write_restart_npz`

## Where each block is implemented

- parsing and validation: `lmx/config.py`
- data models: `lmx/specs.py`
- runtime logger: `lmx/runtime_logging.py`
- restart and output bundles: `lmx/io.py`
- solver dispatch: `lmx/solvers.py`
