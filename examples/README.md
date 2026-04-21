# Examples

The examples are explicit templates for research workflows. They show how to:

- build cases directly in Python
- run from TOML input files
- generate plots and movies
- reproduce canonical Shercliff and Hunt benchmark figures
- save `.npz` state bundles
- resume runs from restart files
- benchmark strong scaling
- run autodiff sensitivity and inverse-design studies
- preview geometries and meshes before launching a solve
- stage fringing-field benchmark workflows for the next solver family
- prescribe variable magnetic fields directly from Python
- customize geometry objects before a solve and preview them in 3D
- preview bent-pipe geometries before solver support lands
- validate analytic cross-sectional magnetic fields before using them in runs

## Quick examples

```bash
python examples/hartmann_example.py --ha 20 --output ./artifacts/examples/hartmann
python examples/shercliff_example.py --ha 20 --output ./artifacts/examples/shercliff
python examples/hunt_example.py --ha 20 --output ./artifacts/examples/hunt
python examples/straight_duct_geometry_and_mesh.py
python examples/shercliff_showcase.py
python examples/hunt_showcase.py
python examples/straight_duct_profile_comparison.py
python examples/theory_meeting_demo.py --output ./artifacts/examples/theory_meeting_demo
python examples/strong_scaling_demo.py --output ./artifacts/examples/strong_scaling_cpu
python examples/autodiff_design_demo.py --output ./artifacts/examples/autodiff_design
python examples/autodiff_sensitivity_demo.py --output ./artifacts/examples/autodiff_sensitivity
python examples/autodiff_profile_design_demo.py --output ./artifacts/examples/autodiff_profile_design
python examples/autodiff_fringing_design_demo.py --output ./artifacts/examples/autodiff_fringing_design
python examples/autodiff_fringing_response_demo.py --output ./artifacts/examples/autodiff_fringing_response
python examples/autodiff_extruded_target_demo.py --output ./artifacts/examples/autodiff_extruded_target
python examples/autodiff_extruded_field_design_demo.py --output ./artifacts/examples/autodiff_extruded_field_design
python examples/autodiff_extruded_trajectory_demo.py --output ./artifacts/examples/autodiff_extruded_trajectory
python examples/plotting_api_demo.py --output ./artifacts/examples/plotting_api_demo
python examples/extruded_summary_figures.py --output ./artifacts/examples/extruded_summary_figures
python examples/readme_showcase_demo.py --output ./docs/_static/generated
# optional Hartmann alternative for wall-layer startup media
python examples/readme_showcase_demo.py --output ./docs/_static/generated --movie-case-kind hartmann
python examples/readme_showcase_demo.py --output ./docs/_static/generated --skip-geometry --movie-view 2d
python examples/readme_showcase_demo.py --output ./docs/_static/generated --skip-geometry --movie-view 3d
python examples/fringing_benchmark_demo.py --output ./artifacts/examples/fringing_benchmark
python examples/extruded_restart_demo.py --output ./artifacts/examples/extruded_restart_demo
python examples/extruded_validation_campaign.py --output ./artifacts/examples/extruded_validation_campaign
python examples/geometry_preview_demo.py --output ./artifacts/examples/geometry_preview
python examples/geometry_preview_demo.py --with-post-run --post-case hartmann --output ./artifacts/examples/geometry_preview_full
python examples/geometry_panel_demo.py --output ./artifacts/examples/geometry_panel
python examples/pipe_reference_comparison_demo.py --output ./artifacts/examples/pipe_reference_comparison
python examples/variable_field_geometry_demo.py --output ./artifacts/examples/variable_field_geometry
python examples/bent_pipe_preview.py
python examples/bent_pipe_inductionless_demo.py
python examples/variable_field_validation.py
python examples/variable_field_extruded_demo.py
python examples/variable_field_layered_demo.py
python examples/variable_field_bent_pipe_demo.py
python examples/variable_field_tabulated_demo.py
python examples/wham_mirror_pipe_demo.py
python examples/autodiff_wham_pressure_sensitivity.py
python examples/q2d_decay_validation.py
python examples/q2d_forced_validation.py
python examples/q2d_wall_bounded_validation.py
python examples/magnetic_obstacle_benchmark.py
python examples/magnetic_obstacle_regime_scan.py
python examples/magnetic_obstacle_baseline.py
```

## Input-file examples

```bash
lmx examples/hartmann_case.toml
lmx examples/hartmann_restart_case.toml
lmx examples/shercliff_case.toml
lmx examples/hunt_case.toml
lmx examples/fringing_rect_case.toml
lmx examples/fringing_tabulated_case.toml
lmx examples/fringing_layered_case.toml
lmx examples/fringing_layered_restart_case.toml
lmx examples/fringing_pipe_case.toml
lmx run hartmann --ha 20 --verbose
lmx run hunt --ha 20 --verbosity debug
lmx run fringing_rect --ha 20 --nx-stations 21 --output out/fringing_rect
lmx run fringing_layered --ha 20 --nx-stations 21 --wall-cells 1 --insulator-cells 1 --output out/fringing_layered
lmx run fringing_pipe --ha 20 --radius 0.5 --nr 24 --ntheta 48 --output out/fringing_pipe
```

The example TOML files expose both `verbose = true|false` and
`verbosity = "quiet" | "normal" | "detailed" | "debug"` in their `[logging]`
block, so they double as templates for both batch and interactive runs.
The fringing TOML files now also enable `write_plots = true`, so a plain
`lmx input.toml` run produces the station-history CSV, NPZ bundle, JSON
summary, and overview plots without any Python wrapper.
When run from the source tree, the validation and example workflows also pick
up the bundled closed-channel reference dataset automatically.

## Replot saved NPZ outputs

```bash
python examples/plot_npz_results.py --npz ./artifacts/examples/theory_meeting_demo/shercliff/shercliff_ha20_results.npz --output ./artifacts/examples/theory_meeting_demo/shercliff/replot
```

## Larger-study demos

Strong scaling:

```bash
python examples/strong_scaling_demo.py --output ./artifacts/examples/strong_scaling_cpu
python examples/strong_scaling_demo.py --remote-host <your_gpu_host> --output ./artifacts/examples/strong_scaling_full
```

Standard CLI runs can also select the execution backend directly:

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 lmx examples/hartmann_case.toml
XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 lmx examples/hartmann_case.toml
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 lmx examples/hunt_case.toml
ssh office 'cd /home/rjorge/tmp/lmx_scaling_repo && PYTHONPATH=/home/rjorge/tmp/lmx_scaling_repo CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda python3 -m lmx examples/hunt_case.toml'
```

Autodiff:

```bash
python examples/autodiff_design_demo.py --output ./artifacts/examples/autodiff_design
python examples/autodiff_sensitivity_demo.py --output ./artifacts/examples/autodiff_sensitivity
python examples/autodiff_profile_design_demo.py --output ./artifacts/examples/autodiff_profile_design
python examples/autodiff_fringing_design_demo.py --output ./artifacts/examples/autodiff_fringing_design
python examples/autodiff_fringing_response_demo.py --output ./artifacts/examples/autodiff_fringing_response
python examples/autodiff_extruded_target_demo.py --output ./artifacts/examples/autodiff_extruded_target
python examples/autodiff_extruded_field_design_demo.py --output ./artifacts/examples/autodiff_extruded_field_design
```

The autodiff examples cover:

- a Hartmann-number sensitivity scan of mean velocity
- a finite-difference cross-check of autodiff gradients
- inverse recovery of a synthetic forcing parameter from a target profile
- full-profile inverse design that recovers both forcing and Hartmann number
- fringing-history inverse design over axial field-profile parameters
- fringing multi-observable inverse design over axial field-profile parameters
  using both velocity and current-response targets
- inverse design against targets generated directly from
  `solve_extruded_inductionless(...)`, using a direct differentiable
  rectangular `extruded_inductionless` response model for the default
  rectangular target workflow
- field-level inverse design against selected `u`, `phi`, `jy`, and `p`
  slices from the extruded projection loop
- projection-trajectory inverse design against selected-station fields and
  charge-balance histories from the projection iterations
- `PNG`/`PDF` summary figures

Fringing-field scaffold:

```bash
python examples/fringing_benchmark_demo.py --geometry-kind rect_duct --ha-peak 20 --ny 12 --nz 12 --nx-stations 11 --max-steps 18 --coupling-iterations 10 --potential-iterations 60 --output ./artifacts/examples/fringing_benchmark
python examples/fringing_benchmark_demo.py --geometry-kind layered_duct --output ./artifacts/examples/fringing_benchmark_layered
python examples/fringing_benchmark_demo.py --geometry-kind pipe_ogrid --output ./artifacts/examples/fringing_benchmark_pipe
python examples/extruded_restart_demo.py --geometry-kind layered_duct --output ./artifacts/examples/extruded_restart_demo
python examples/extruded_validation_campaign.py --output ./artifacts/examples/extruded_validation_campaign
```

That example now writes a stacked axial field bundle with `u`, `v`, `w`, `p`,
`phi`, current, Lorentz, and charge-balance fields through the explicit
`solve_extruded_inductionless(...)` entry point. Rectangular ducts, layered
ducts, and mapped pipes all now use the same public path. The mapped-pipe lane
also has a dedicated external-profile comparison script.

The restart and validation campaign examples extend that same 3D lane:

- `extruded_restart_demo.py`
  - splits a run into base and resumed stages, then compares the resumed result
    against a direct run with the same total step count
- `extruded_validation_campaign.py`
  - runs the bounded larger-dataset fringing campaign on the
    `rect_duct,layered_duct,pipe_ogrid` conservation-validation set, writes JSON/CSV, and emits a
    figure
- `autodiff_extruded_trajectory_demo.py`
  - matches selected-station fields and conservation histories across the
    projection iterations themselves, not just the final extruded state
- `extruded_summary_figures.py`
  - writes extra 3D fringing figures and compact summary panels for the
    rectangular and layered datasets
- `readme_showcase_demo.py`
  - regenerates the README media bundle, including the geometry panel
    and bounded 2D/3D Hunt startup GIFs
  - supports split refreshes with `--movie-view 2d` or `--movie-view 3d`
    when only one GIF needs to be updated inside the five-minute local budget

Geometry and mesh preview:

```bash
python examples/geometry_preview_demo.py --output ./artifacts/examples/geometry_preview
python examples/geometry_preview_demo.py --with-post-run --post-case hartmann --output ./artifacts/examples/geometry_preview_full
```

The default path is intentionally preview-only so it stays fast. Add
`--with-post-run` to append a short steady solve and matching postprocessing
figures in the same output tree.

Variable field and custom geometry driver:

```bash
python examples/variable_field_geometry_demo.py --output ./artifacts/examples/variable_field_geometry
```

That example is the clearest Python-native template for users who want to:

- define a custom analytic magnetic field callable
- modify the benchmark constructors with `dataclasses.replace(...)`
- preview the resulting geometry before running
- run a short solve and emit the same plots and JSON summaries used elsewhere
- use LMX as a programmable research driver instead of only through TOML files

Bent-pipe preview:

```bash
python examples/bent_pipe_preview.py
```

That example is the preprocessing template for the curved-pipe lane:

- build a mapped constant-radius bend directly from Python
- write `PNG`/`PDF` previews of the curved centerline and cross-sections
- record the mesh parameters that should be reused when the bent-pipe solver lane is added

Bent-pipe inductionless baseline:

```bash
python examples/bent_pipe_inductionless_demo.py
```

That example is the current curved-pipe executable lane:

- build a bent-pipe inductionless problem and the matching straight-pipe reference
- solve both on the same fringing-field profile
- write a bent-pipe geometry-plus-solution panel
- record low-De equivalence metrics against the straight-pipe limit

Variable-field validation:

```bash
python examples/variable_field_validation.py
```

That example is the smallest field-QA workflow:

- build an analytic divergence-free cross-sectional magnetic field
- sample it on a research-sized grid
- write component and magnitude plots
- record finite-difference divergence metrics before using that field in a solve

Variable-field extruded solve:

```bash
python examples/variable_field_extruded_demo.py
```

That example extends the field-QA lane into an actual 3D solve:

- build an analytic divergence-free magnetic field
- run the rectangular `extruded_inductionless` solve with that field
- write the field preview and extruded overview plots
- record field-divergence and 3D conservation metrics in one summary

Variable-field layered duct and curved pipe:

```bash
python examples/variable_field_layered_demo.py
python examples/variable_field_bent_pipe_demo.py
python examples/variable_field_tabulated_demo.py
python examples/wham_mirror_pipe_demo.py
python examples/autodiff_wham_pressure_sensitivity.py
```

Those examples extend the same field API into:

- layered ducts with wall materials retained in the 3D solve
- curved pipes validated against the straight-pipe low-De limit under the same field
- reusable machine-readable summaries of field and conservation metrics
- a tabulated-field duct run that also matches the TOML/CLI path
- a tabulated WHAM-like mirror field written to disk and reused by a pipe solve
- a reduced differentiable pressure-drop sensitivity study with respect to
  coil-coil separation

Benchmark C / Q2D baseline:

```bash
python examples/q2d_decay_validation.py
```

That example is the first executable Benchmark C slice:

- solve a periodic quasi-2D Hartmann-friction decay problem
- compare the numerical final state against the analytic decay
- write a compact validation figure and summary JSON

Forced Benchmark C / Q2D duct:

```bash
python examples/q2d_forced_validation.py
```

That example upgrades the Q2D lane from decay-only to a forced duct baseline:

- solve a periodic Hartmann-friction mode driven toward a steady state
- compare the steady state and approach-to-steady-state against the analytic solution
- write a compact validation figure and summary JSON

Wall-bounded Benchmark C / Q2D duct:

```bash
python examples/q2d_wall_bounded_validation.py
```

That example moves the Q2D lane from periodic modes to a wall-bounded duct:

- solve a no-slip forced Q2D Hartmann-friction mode in a rectangular box
- compare the transient final state against the exact Dirichlet solution
- write a compact validation figure and summary JSON

First Benchmark D slice:

```bash
python examples/magnetic_obstacle_benchmark.py
python examples/magnetic_obstacle_baseline.py
python examples/wham_mirror_pipe_demo.py
python examples/autodiff_wham_pressure_sensitivity.py
```

Those examples are the current executable Benchmark D entry points:

- `magnetic_obstacle_benchmark.py`
  - solve a rectangular extruded duct with a localized analytic magnetic obstacle
  - compare it directly against a matched no-field reference
  - write both the generic extruded overview and a dedicated benchmark panel
  - record normalized velocity-deficit, pressure-excess, distortion, and
    conservation metrics
- `magnetic_obstacle_regime_scan.py`
  - sweep localized-field obstacle runs over `Bz` scale and forcing
  - write a compact response-map figure over velocity deficit, pressure excess,
    current response, and cross-cut distortion
  - stage the transition from the current baseline toward stronger-inertia Benchmark D cases
- `magnetic_obstacle_baseline.py`
  - solve a rectangular extruded duct with a localized analytic magnetic obstacle
  - write the full extruded overview panel
  - record obstacle-induced velocity deficit, current response, and conservation metrics
- `wham_mirror_pipe_demo.py`
  - write a tabulated WHAM-like mirror field
  - solve a straight pipe crossing that 3D field
  - export the field preview, a 3D WHAM overview figure, and the pipe-response overview
- `autodiff_wham_pressure_sensitivity.py`
  - treat the same mirror topology as a differentiable stationwise profile
  - compute pressure-drop sensitivity with respect to coil separation
  - write the three-panel sensitivity figure and summary JSON

Importable plotting API:

```bash
python examples/plotting_api_demo.py --output ./artifacts/examples/plotting_api_demo
```

That example is the minimal template for users who want to:

- call the plotting helpers directly from `import lmx`
- save geometry previews before running
- write steady-state overview plots after a solve
- generate transient GIFs from `solve_case_snapshots(...)`

## Teaching goal

The examples are intentionally written with explicit helper functions and
configuration blocks so that new users can adapt them into custom research
drivers.

## Straight-duct showcase

These four standalone scripts are the shortest path to the canonical
Shercliff/Hunt figures:

```bash
python examples/straight_duct_geometry_and_mesh.py
python examples/shercliff_showcase.py
python examples/hunt_showcase.py
python examples/straight_duct_profile_comparison.py
```

They are parameter-driven Python files rather than argparse front ends:

- edit the configuration block at the top of the file
- rerun the script
- inspect the output tree under `artifacts/examples/...`

The shared reusable logic lives in `lmx.showcase`, so the example scripts stay
teachable without duplicating the geometry setup, solve, and plotting code.
`straight_duct_profile_comparison.py` writes the normalized Shercliff/Hunt
midplane comparison figure used in the README and docs.
