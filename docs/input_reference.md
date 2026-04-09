# Input Reference

LMX can now run directly from a TOML input file:

```bash
lmx /Users/rogerio/local/tests/LMX/examples/hartmann_case.toml
```

That path is intended to be the main executable workflow for users who want an
OpenFOAM-like run experience with a text input file, live solver logs, and a
structured output directory.

## What a TOML run writes

When the input file enables the corresponding output options, LMX writes:

- `case_name.vtr` and `case_name.pvd` for ParaView
- `case_name_centerline.csv`
- `case_name_midplane_y.csv`
- `case_name_midplane_z.csv`
- `case_name_results.npz`
- `case_name_restart.npz` when restart output is enabled
- `case_name_summary.json`
- `case_name.log`
- overview and diagnostics figures when `write_plots = true`
- a copy of the input TOML when `copy_input_file = true`

## Input sections

An input file is organized as:

```toml
[case]
[geometry]
[magnetic_field]
[time_stepper]
[output]
[logging]
[restart]
[[regions]]
[[boundary_conditions]]
```

## `case`

- `name`: case name used for output file stems
- `solve_mode`: `"steady"` or `"transient"`
- `forcing`: explicit reduced-model streamwise forcing
- `initial_velocity`: initial fluid velocity
- `reference_pressure_gradient`: stored reference value for reduced-model comparisons
- `reference_phi_cell`: two integers fixing the electric-potential gauge
- `notes`: free-form string stored in metadata

## `geometry`

- `kind`: `"rect_duct"`, `"layered_duct"`, or `"pipe_ogrid"`
- `width`, `height`, `length`
- `nx`, `ny`, `nz`
- `radius`, `nr`, `ntheta` for mapped-pipe cases
- `wall_thickness`: four values `[left, right, bottom, top]`
- `wall_cells`: four integers `[left, right, bottom, top]`
- `target_ha`
- `target_side_layer`

## `magnetic_field`

- `kind`: currently practical TOML modes are `"constant"` and `"tabulated"`
- `value`: three-vector for constant fields
- `table_path`: path to a tabulated field file
- `ramp_start`
- `ramp_duration`

Custom analytic magnetic fields still require the Python API, because a callable
cannot be serialized cleanly into TOML.

## `time_stepper`

- `dt`
- `t_final`
- `max_steps`
- `outer_iterations`
- `potential_iterations`
- `potential_tolerance`
- `potential_relaxation`
- `potential_solver`
- `current_reconstruction`
- `post_update_potential_refresh`
- `steady_tolerance`
- `steady_potential_tolerance`
- `relaxation`
- `velocity_update_limit`
- `velocity_update_limiter`
- `checkpoint_stride`

The retained values in the shipped examples are the current known-good
baselines. They are not intended to be universal defaults for all future
geometries.

When `forcing = 0` and the case uses reduced mean-flow closure through
`inlet_flow_rate`, the solver now computes those cross-sectional means with
cell-area weighting on clustered meshes rather than raw cell counts.

## `output`

- `directory`
- `write_paraview`
- `write_csv_profiles`
- `write_npz`
- `write_json_summary`
- `write_plots`
- `copy_input_file`
- `write_stride`

## `logging`

- `enabled`
- `banner`
- `print_regions`
- `print_boundaries`
- `print_footer`
- `flush`
- `step_stride`

With `enabled = true`, the solver prints a live OpenFOAM-style stream and also
writes the same text to `case_name.log` in the output directory.

## `restart`

- `enabled`
- `path`
- `reset_histories`
- `write_restart`
- `restart_filename`

Restart is a runtime concern, so it lives in its own top-level table instead of
inside the physical case definition.

Current retained semantics:

- the restart source is a previously written LMX `.npz` solution dump
- the saved `u`, `phi`, `J`, `JxB`, `time`, and `residual` become the initial state
- transient runs continue from the saved `time` to the absolute `t_final` in the new TOML
- steady runs continue from the saved pseudo-time/state for up to the new `max_steps`
- `reset_histories = true` starts fresh diagnostic histories at the restart point
- `reset_histories = false` appends the new histories to the saved ones

Minimal example:

```toml
[restart]
enabled = true
path = "../artifacts/examples/toml_hartmann/hartmann_ha20_toml_results.npz"
reset_histories = false
write_restart = true
restart_filename = "hartmann_ha20_toml_restart.npz"
```

## `regions`

Each `[[regions]]` entry maps directly to `RegionSpec`:

- `name`
- `kind`: `"fluid"` or `"solid"`
- `conductivity`
- `density`
- `viscosity`
- `wall_thickness`

## `boundary_conditions`

Each `[[boundary_conditions]]` entry maps directly to `BoundaryCondition`:

- `name`
- `kind`
- `value`
- `region`
- `axis`
- `side`

Typical values:

- no-slip walls:
  `kind = "no_slip"`
- insulating walls:
  `kind = "insulating"`
- conducting walls via explicit solid regions:
  `kind = "conducting_wall"` with `region = "..."` and the appropriate `side`
- reduced-model flow-rate drive:
  `kind = "inlet_flow_rate"`
- startup-state metadata only:
  `kind = "inlet_velocity"`

## Shipped examples

- `examples/hartmann_case.toml`
- `examples/hartmann_restart_case.toml`
- `examples/shercliff_case.toml`
- `examples/hunt_case.toml`

The Hunt example is the most informative input file because it shows:

- multiple regions
- explicit insulating and conducting walls
- layered geometry controls
- the retained Hunt current/Lorentz reconstruction baseline

`examples/hartmann_restart_case.toml` is the retained restart/continue example.
Run `examples/hartmann_case.toml` first so the referenced NPZ exists, then run
the restart file to continue from that saved state.
