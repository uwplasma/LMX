# Input Reference

LMX runs from a single TOML file:

```bash
lmx examples/hartmann_case.toml
```

## Top-level blocks

- `[case]`
- `[geometry]`
- `[magnetic_field]`
- `[solver]`
- `[time_stepper]`
- `[output]`
- `[logging]`
- `[restart]`
- `[fringing]`
- `[[regions]]`
- `[[boundary_conditions]]`

## Minimal complete example

```toml
[case]
name = "hartmann_ha20"
forcing = 1.0
initial_velocity = 0.0
reference_pressure_gradient = -1.0
reference_phi_cell = [0, 0]

[geometry]
kind = "rect_duct"
width = 2.0
height = 2.0
length = 1.0
nx = 1
ny = 48
nz = 48
wall_thickness = [0.0, 0.0, 0.0, 0.0]
wall_cells = [0, 0, 0, 0]
target_ha = 20.0

[magnetic_field]
kind = "constant"
value = [0.0, 0.0, 20.0]
ramp_start = 0.0
ramp_duration = 0.0

[solver]
kind = "fully_developed_inductionless"
mode = "steady"
linear_solver = "auto"
preconditioner = "jacobi"
time_scheme = "implicit_euler"
coupling_iterations = 12
coupling_tolerance = 1.0e-8

[time_stepper]
dt = 1.0e-3
t_final = 0.1
max_steps = 200
potential_iterations = 400
potential_tolerance = 1.0e-6
potential_relaxation = 1.0
potential_solver = "cg"
steady_tolerance = 1.0e-8
steady_potential_tolerance = 1.0e-6
checkpoint_stride = 1

[output]
directory = "artifacts/hartmann"
write_paraview = true
write_csv_profiles = true
write_npz = true
write_json_summary = true
write_plots = true
copy_input_file = true
write_stride = 1

[logging]
enabled = true
verbose = true
verbosity = "detailed"
banner = true
print_regions = true
print_boundaries = true
print_footer = true
flush = true
step_stride = 1

[restart]
enabled = false

[[regions]]
name = "fluid"
kind = "fluid"
conductivity = 1.0
density = 1.0
viscosity = 0.01

[[boundary_conditions]]
name = "y_min_wall"
kind = "no_slip"
axis = "y"
side = "min"

[[boundary_conditions]]
name = "y_max_wall"
kind = "no_slip"
axis = "y"
side = "max"

[[boundary_conditions]]
name = "z_min_wall"
kind = "insulating"
axis = "z"
side = "min"

[[boundary_conditions]]
name = "z_max_wall"
kind = "insulating"
axis = "z"
side = "max"
```

## `[case]`

- `name`
  - case identifier used in output files
- `forcing`
  - explicit streamwise forcing used by the reduced duct problems
- `initial_velocity`
  - initial streamwise velocity used for transient and steady starts
- `reference_pressure_gradient`
  - reference forcing scale used in benchmark constructors
- `reference_phi_cell`
  - gauge anchor index for `phi`
- `notes`
  - free-form case description copied into output metadata

## `[geometry]`

- `kind`
  - `rect_duct`
  - `layered_duct`
  - `pipe_ogrid`
- `width`, `height`, `length`
  - domain dimensions
- `nx`, `ny`, `nz`
  - grid counts
- `radius`, `nr`, `ntheta`
  - mapped-pipe controls for `pipe_ogrid`
- `wall_thickness`
  - four wall thickness values for layered ducts
- `wall_cells`
  - four wall-cell counts for layered ducts
- `target_ha`
  - benchmark Hartmann number used by the case constructors
- `target_side_layer`
  - optional side-layer tuning input for layered benchmarks

## `[magnetic_field]`

- `kind`
  - `constant`
  - `table`
  - analytic callables are Python-only, not TOML inputs
- `value`
  - constant field components
- `table_path`
  - optional table-backed field input path
- `ramp_start`
- `ramp_duration`

## `[solver]`

- `kind`
  - `fully_developed_inductionless`
  - `extruded_inductionless`
    - executable 3D fringing-field path for `rect_duct`, `layered_duct`, and
      `pipe_ogrid`
    - requires a `[fringing]` block in TOML inputs
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
- `coupling_iterations`
- `coupling_tolerance`

## `[time_stepper]`

- `dt`
- `t_final`
- `t_final` is a hard stop horizon; the solver does not round it up into an
  extra step when `t_final / dt` is fractional
- `max_steps`
- `max_steps` is a hard upper bound on executed steps
- `potential_iterations`
- `potential_tolerance`
- `potential_relaxation`
- `potential_solver`
- `steady_tolerance`
- `steady_potential_tolerance`
- `checkpoint_stride`
- reduced-solver-only controls:
  - `outer_iterations`
  - `current_reconstruction`
  - `post_update_potential_refresh`
  - `relaxation`
  - `velocity_update_limit`
  - `velocity_update_limiter`

## `[fringing]`

- `enabled`
  - required for `extruded_inductionless`
- `entry_center`
  - center of the entrance field transition along the axial direction
- `exit_center`
  - center of the exit field transition along the axial direction
- `transition_width`
  - smooth transition width used in the hyperbolic-tangent field profile
- `axis`
  - magnetic-field axis for the staged fringing profile
  - allowed values: `x`, `y`, `z`

## `[output]`

- `directory`
- `write_paraview`
- `write_csv_profiles`
- `write_npz`
  - for `extruded_inductionless`, also writes
    `system/<case>_extruded_manifest.json` plus `fields/stations/station_XXXX.npz`
    bundles for the selected axial stations
- `write_json_summary`
- `write_plots`
  - for `extruded_inductionless`, writes `overview.png` / `overview.pdf`
    summarizing station response, conservation histories, and peak-field
    cross-sections
- `copy_input_file`
- `write_stride`
  - for `extruded_inductionless`, controls the archived station stride used for
    `fields/stations/station_XXXX.npz`

## `[logging]`

- `enabled`
  - master on/off switch for the live runtime logger
- `verbose`
  - simple boolean alias for common usage
  - `true` enables live logging
  - `false` disables it
- `verbosity`
  - `quiet`
  - `normal`
  - `detailed`
  - `debug`
- `banner`
- `print_regions`
- `print_boundaries`
- `print_footer`
- `flush`
- `step_stride`
- at `detailed` and `debug`, the live log prints both initial and final
  residuals for the potential and velocity solves together with conservation
  diagnostics

Recommended usage:

- `verbose = false` for large parameter sweeps
- `verbosity = "normal"` for routine local runs
- `verbosity = "detailed"` for validation runs
- `verbosity = "debug"` for solver investigations

## `[restart]`

- `enabled`
- `path`
- `reset_histories`
- `write_restart`
- `restart_filename`

For `extruded_inductionless` runs, restart bundles now store the full
3D field bundle (`u`, `v`, `w`, `p`, `\phi`, current, Lorentz, and stationwise
conservation histories). The executable TOML path writes those bundles into
`restart/` and uses the structured 3D output layout:

- `system/`
- `fields/`
- `postProcessing/`
- `restart/`
- `logs/`

## `[[regions]]`

- `name`
- `kind`
  - `fluid`
  - solid-region tags used in layered ducts
- `conductivity`
- `density`
- `viscosity`
- `wall_thickness`

## `[[boundary_conditions]]`

- `name`
- `kind`
  - `no_slip`
  - `insulating`
  - `conducting_wall`
  - `inlet_velocity`
  - `inlet_flow_rate`
- `axis`
- `side`
- `region`
- `value`
  - scalar or 3-vector depending on the boundary type

## Python driver usage

In Python driver scripts, use the same runtime controls through
`lmx.config.LoggingSpec.from_user_controls(...)`. For example:

```python
from lmx.config import LoggingSpec
from lmx.runtime_logging import StreamingSolverLogger

logging = LoggingSpec.from_user_controls(verbose=True, verbosity="debug")
logger = StreamingSolverLogger(logging)
```

## Where each block is implemented

- parsing and validation: `lmx/config.py`
- data models: `lmx/specs.py`
- runtime logger: `lmx/runtime_logging.py`
- restart and output bundles: `lmx/io.py`
- solver dispatch: `lmx/solvers.py`
