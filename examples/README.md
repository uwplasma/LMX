# Examples

The examples are explicit templates for research workflows. They show how to:

- build cases directly in Python
- run from TOML input files
- generate plots and movies
- save `.npz` state bundles
- resume runs from restart files
- benchmark strong scaling
- run autodiff sensitivity and inverse-design studies
- preview geometries and meshes before launching a solve
- stage fringing-field benchmark workflows for the next solver family
- prescribe variable magnetic fields directly from Python
- customize geometry objects before a solve and preview them in 3D

## Quick examples

```bash
python examples/hartmann_example.py --ha 20 --output ./artifacts/examples/hartmann
python examples/shercliff_example.py --ha 20 --output ./artifacts/examples/shercliff
python examples/hunt_example.py --ha 20 --output ./artifacts/examples/hunt
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
python examples/extruded_summary_figures.py --output ./artifacts/examples/extruded_summary_figures
python examples/readme_showcase_demo.py --output ./docs/_static/generated
python examples/fringing_benchmark_demo.py --output ./artifacts/examples/fringing_benchmark
python examples/extruded_restart_demo.py --output ./artifacts/examples/extruded_restart_demo
python examples/extruded_validation_campaign.py --output ./artifacts/examples/extruded_validation_campaign
python examples/geometry_preview_demo.py --output ./artifacts/examples/geometry_preview
python examples/geometry_preview_demo.py --with-post-run --post-case hartmann --output ./artifacts/examples/geometry_preview_full
python examples/geometry_panel_demo.py --output ./artifacts/examples/geometry_panel
python examples/pipe_reference_comparison_demo.py --output ./artifacts/examples/pipe_reference_comparison
python examples/variable_field_geometry_demo.py --output ./artifacts/examples/variable_field_geometry
```

## Input-file examples

```bash
lmx examples/hartmann_case.toml
lmx examples/hartmann_restart_case.toml
lmx examples/shercliff_case.toml
lmx examples/hunt_case.toml
lmx examples/fringing_rect_case.toml
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

The shipped autodiff examples now cover:

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
  slices from the retained extruded projection loop
- projection-trajectory inverse design against selected-station fields and
  charge-balance histories from the retained projection iterations
- `PNG`/`PDF` summary figures

Fringing-field scaffold:

```bash
python examples/fringing_benchmark_demo.py --geometry-kind rect_duct --ha-peak 20 --ny 12 --nz 12 --nx-stations 11 --max-steps 18 --coupling-iterations 10 --potential-iterations 60 --output ./artifacts/examples/fringing_benchmark
python examples/fringing_benchmark_demo.py --geometry-kind layered_duct --output ./artifacts/examples/fringing_benchmark_layered
python examples/fringing_benchmark_demo.py --geometry-kind pipe_ogrid --output ./artifacts/examples/fringing_benchmark_pipe_exploratory
python examples/extruded_restart_demo.py --geometry-kind layered_duct --output ./artifacts/examples/extruded_restart_demo
python examples/extruded_validation_campaign.py --output ./artifacts/examples/extruded_validation_campaign
```

That example now writes a stacked axial field bundle with `u`, `v`, `w`, `p`,
`phi`, current, Lorentz, and charge-balance fields through the explicit
`solve_extruded_inductionless(...)` entry point. Rectangular ducts, layered
ducts, and mapped pipes all now use the same public path. The mapped-pipe lane
is still best treated as exploratory on the larger retained validation set,
even though the example surface is fully executable and now has a dedicated
qualitative external-profile comparison script.

The restart and validation campaign examples extend that same 3D lane:

- `extruded_restart_demo.py`
  - splits a run into base and resumed stages, then compares the resumed result
    against a direct run with the same total step count
- `extruded_validation_campaign.py`
  - runs the bounded larger-dataset fringing campaign on the retained
    `rect_duct,layered_duct,pipe_ogrid` conservation-validation set, writes JSON/CSV, and emits a
    figure
- `autodiff_extruded_trajectory_demo.py`
  - matches selected-station fields and conservation histories across the
    retained projection iterations themselves, not just the final extruded state
- `extruded_summary_figures.py`
  - writes extra 3D fringing figures and compact summary panels for the
    rectangular and layered datasets
- `readme_showcase_demo.py`
  - regenerates the retained README media bundle, including the geometry panel
    and bounded 2D/3D Hunt startup GIFs

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

## Teaching goal

The examples are intentionally written with explicit helper functions and
configuration blocks so that new users can adapt them into custom research
drivers.
