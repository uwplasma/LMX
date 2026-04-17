# LMX

[![CI](https://github.com/uwplasma/LMX/actions/workflows/ci.yml/badge.svg)](https://github.com/uwplasma/LMX/actions/workflows/ci.yml)
[![Docs](https://github.com/uwplasma/LMX/actions/workflows/docs.yml/badge.svg)](https://github.com/uwplasma/LMX/actions/workflows/docs.yml)
![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue)
![Coverage](https://img.shields.io/badge/coverage-96%25-brightgreen)
![License](https://img.shields.io/badge/license-MIT-green)

LMX is a JAX-native inductionless MHD code for structured meshes. It provides
fully developed duct solvers, a 3D `extruded_inductionless` fringing-field
solver lane, restartable CLI workflows, strong-scaling tooling, and
differentiable workflows for sensitivity analysis and inverse design.

## Why use LMX

- Fully developed Hartmann, Shercliff, and Hunt workflows
- Rectangular, layered, and mapped-pipe geometry support
- JAX-based CPU and GPU execution
- Explicit conservation diagnostics for charge closure and boundary-current audits
- Input-file and Python-driver workflows
- Built-in plots, movies, and validation reports
- Autodiff examples for inverse design and sensitivity analysis

## Installation

Minimal install:

```bash
git clone https://github.com/uwplasma/LMX
cd LMX
python -m pip install -e .
```

Full development install:

```bash
python -m pip install -e '.[dev,plotting,docs,extras]'
```

LMX supports Python `3.10+`, falls back to `tomli` on Python `3.10`, and works
with the installed JAX/JAXLIB pair rather than pinning a narrow runtime window.

## Quick start

CLI:

```bash
lmx examples/hartmann_case.toml
lmx examples/hunt_case.toml
lmx examples/fringing_rect_case.toml
lmx run fringing_layered --ha 20 --nx-stations 21 --wall-cells 1 --insulator-cells 1 --output out/fringing_layered
```

Python:

```python
from lmx.cases import make_hartmann_case
from lmx.solvers import solve_steady

case = make_hartmann_case(ha=20.0, ny=48, nz=48)
solution = solve_steady(case)
print(solution.diagnostics.residual_history[-1])
```

Post-processing from Python:

```python
from lmx import (
    make_hartmann_case,
    solve_steady,
    solve_case_snapshots,
    write_case_overview_plots,
    write_geometry_preview_plots,
    write_transient_movies,
)

case = make_hartmann_case(ha=20.0, ny=32, nz=32)
steady = solve_steady(case)
write_geometry_preview_plots(steady.mesh, "out/geometry", case_title=case.name)
write_case_overview_plots(steady, "out/steady", case_title=case.name)
frames = solve_case_snapshots(case, frame_count=40)
write_transient_movies(frames, "out/movies", case_title=case.name, output_stem="hartmann_startup")
```

Backend selection from the shell:

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 lmx examples/hartmann_case.toml
CUDA_VISIBLE_DEVICES=0 JAX_PLATFORMS=cuda lmx examples/hunt_case.toml
XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 lmx examples/hartmann_case.toml
```

For larger scaling studies, use:

```bash
python examples/strong_scaling_demo.py --output artifacts/examples/strong_scaling_cpu
python examples/strong_scaling_demo.py --remote-host office --output artifacts/examples/strong_scaling_full
```

## Showcase

### Geometry and flow states

The top panel shows the three geometry families and the actual mesh or
wall layout each solver sees:

- Hartmann flow in a rectangular duct
- Hunt flow in a layered duct with conducting Hartmann walls
- a mapped-pipe O-grid for fringing-field studies

LMX solves liquid-metal duct and pipe flows in imposed magnetic fields. No
plasma background is needed to read the figures:

- a `rect_duct` is a plain rectangular channel
- a `layered_duct` adds wall materials around that channel so conducting and
  insulating walls can be represented explicitly
- a `pipe_ogrid` is the same idea in a circular pipe with an O-grid mesh

The two most important benchmark ideas in the README are:

- `Hunt` flow:
  liquid metal in a rectangular duct with conducting Hartmann walls, insulating
  side walls, and a transverse magnetic field
- `fringing field`:
  a magnetic field that changes along the flow direction instead of staying
  uniform, which makes the problem fully 3D because the fluid accelerates and
  redistributes as it enters and leaves the magnetized region

![LMX geometry gallery](docs/_static/generated/geometry_gallery.png)

Those geometries are not shown in isolation in the rest of the README:

- the startup GIFs below use the layered Hunt geometry
- the fringing overview below uses the rectangular extruded 3D geometry
- the mapped-pipe workflow is exercised through the fringing CLI/TOML and the
  pipe comparison example in `examples/pipe_reference_comparison_demo.py`

### Straight-duct setup

The standalone straight-duct showcase scripts build the same Shercliff/Hunt
family from explicit geometry, material, and mesh parameters. The first figure
labels the wall-material layout used by the Shercliff and Hunt benchmarks, and
the second shows the clustered structured mesh used to resolve the Hartmann
layers.

<p align="center">
  <img src="docs/_static/generated/lm_duct_geometry_setup.png" alt="Straight-duct geometry setup" width="48%">
  <img src="docs/_static/generated/structured_mesh_ha20.png" alt="Structured straight-duct mesh" width="48%">
</p>

### 2D and 3D startup movies

These README assets are generated from `examples/readme_showcase_demo.py` and
show Hunt startup in 2D and 3D over `t = 0` to `t = 2 ms`. Hunt flow uses a
layered duct with conducting Hartmann walls and insulating side walls, so the
startup sequence develops thin Hartmann layers at the conducting walls and a
deformed core profile across the span. The run starts from a flat plug-flow
profile, time is shown in physical units, and all solved timesteps are written
to the GIF. The 2D panel carries the transient `y`- and `z`-centerline
diagnostics so the layer growth can be read directly from the movie, while the
3D panel renders the full streamwise-velocity field as a stack of `y-z` slices
inside the duct volume. The README regeneration path uses a bounded `49 × 49`
cross-section with `dt = 1e-5 s`, `t_final = 2e-3 s`, `coupling_iterations = 6`,
and `potential_iterations = 48`.

<p align="center">
  <img src="docs/_static/generated/readme_hunt_startup_2d.gif" alt="LMX 2D Hunt startup movie" width="48%">
  <img src="docs/_static/generated/readme_hunt_startup_3d.gif" alt="LMX 3D Hunt startup movie" width="48%">
</p>

### Fringing-field response

The 3D fringing overview shows the imposed magnetic-field ramp, the response of
the streamwise velocity, the pressure span, and the charge/current diagnostics
for a rectangular duct in one place.

Here the magnetic field is weak upstream, ramps up inside the magnet region,
and drops again downstream. That axial field variation is what “fringing” means
in this context. The committed overview is generated on a `24 × 24 × 33`
cross-section/station grid so the axial histories and cross-sectional panels
are not limited by a coarse axial sweep.

![LMX fringing-field overview](docs/_static/generated/fringing_benchmark_rect.png)

### Scaling and autodiff

The scaling panel below is a fixed-problem strong-scaling benchmark for a dense
structured-grid `extruded3d` inductionless MHD operator. Solid lines are
measured warm runtimes and speedups; the dashed line is ideal linear speedup.
The current figure uses a `2048×64×64` CPU case with `1024` operator
iterations and a `6144×96×96` GPU case with `4096` operator iterations, so the
device curves come from minute-scale kernels instead of short smoke tests.

Measured warm-runtime points:

- CPU: `79.45 s`, `68.68 s`, `64.09 s`, `66.16 s` at `1, 2, 4, 8`
- GPU: `78.58 s`, `62.52 s` at `1, 2`

On this workstation the CPU curve improves through `4` logical devices and then
flattens at `8`, which is still consistent with a host-memory-bandwidth and
communication limit on the current sharded operator path. The corrected
two-GPU path keeps the cleaner strong-scaling trend on the larger fixed
problem.

![LMX strong scaling](docs/_static/generated/strong_scaling.png)

The autodiff panel summarizes two things: the mean velocity and its sensitivity
to Hartmann number on the left, and a simple inverse-design loop recovering the
forcing on the right. The left panel shows the expected decrease in throughput
as Hartmann layers strengthen. The right panel shows the optimizer recovering
the forcing that generated the target profile while the loss falls by several
orders of magnitude.

## Meshing

LMX uses structured meshes. The user controls resolution directly through the
input file or Python case object:

- `ny`, `nz` for rectangular and layered ducts
- `wall_cells`, `wall_thickness`, and `insulator_cells` when wall materials are
  modeled explicitly
- `nx_stations` for the axial resolution of `extruded_inductionless`
- `nr`, `ntheta`, and `radius` for `pipe_ogrid`

The practical rule is to resolve the thin boundary layers before trusting a
benchmark or design study. For Hartmann and Hunt problems, increase `ny` and
`nz` until flow rate, current diagnostics, and interface-current residuals stop
moving materially. For fringing-field studies, refine both the cross-section and
the axial stations until pressure span, charge-balance residuals, and
throughput variation stabilize.

![LMX autodiff summary](docs/_static/generated/autodiff_summary.png)

## Validation status

The current validation surface includes:

- fast CI under a five-minute routine budget
- strict docs build
- restartable TOML and CLI workflows
- internal conservation and fringing-physics gates on `rect_duct`, `layered_duct`, and `pipe_ogrid`
- mapped-pipe external comparison now uses one shared velocity normalization
  across all transverse lines, and currently exposes a real high-`Ha`,
  high-`Re` parity gap rather than hiding it behind per-line normalization
- widened bounded manual fringing campaign at `Ha = 10, 20, 30`, `resolution = 8`
- a standalone quantitative Benchmark B summary driver for rectangular, layered,
  and mapped-pipe fringing cases

The widened bounded manual campaign is intentionally stricter than the release
gate. On the current tree it confirms the 3D fringing set at `Ha = 10, 20, 30`
for `rect_duct`, `layered_duct`, and `pipe_ogrid`, and the repaired fully
developed Hunt low-resolution row now also passes that heavier conservation
check. On the bounded `Ha = 10`, `resolution = 8` manual run, the Hunt
interface-current residual is now `≈ 1.27e-2` instead of the old failing
`≈ 4.20e-1`.

The heavier 3D validation campaign is generated with:

```bash
python scripts/run_full_validation_exercise.py --output artifacts/validation/full_validation_exercise --ha-values 10,20 --resolution 12 --fringing-resolutions 8,12 --skip-paraview --write-plot
python scripts/run_benchmark_b_quantitative.py --output artifacts/validation/benchmark_b_quantitative --ha-peak 20 --duct-ny 20 --duct-nz 20 --pipe-nr 20 --pipe-ntheta 80 --nx-stations 21 --max-steps 24 --coupling-iterations 12 --potential-iterations 80
python examples/extruded_validation_campaign.py --output artifacts/examples/extruded_validation_campaign --ha-values 10,20 --resolutions 10,14 --fringing-nx 5
python scripts/run_manual_solver_family_validation.py --output artifacts/manual_validation/solver_family_summary.json --ha-values 10,20 --resolutions 8,12 --include-fringing --fringing-geometries rect_duct,layered_duct,pipe_ogrid --fringing-nx 5 --max-steps 12 --potential-iterations 48 --coupling-iterations 8 --write-csv --write-plot
```

When run from the source tree, the closed-channel validation commands use the
bundled reference dataset under `external/FreeMHDPaperAllFigures/.../ClosedChannel`
by default, so an explicit `--reference-root` is only needed when comparing
against a different dataset.

## Examples

Useful entry points:

- `examples/readme_showcase_demo.py`: regenerates the README media bundle
- `examples/straight_duct_geometry_and_mesh.py`: geometry/setup and structured mesh figures for Shercliff/Hunt straight ducts
- `examples/shercliff_showcase.py`: Shercliff boundary-layer, annotated cross-section, 3D profile, and startup media
- `examples/hunt_showcase.py`: Hunt boundary-layer, annotated cross-section, 3D profile, and startup media
- `examples/straight_duct_profile_comparison.py`: analytical versus LMX Shercliff/Hunt profile overlay
- `examples/plotting_api_demo.py`: direct import-and-plot post-processing workflow
- `examples/geometry_panel_demo.py`: geometry previews plus paired geometry/simulation panel
- `examples/fringing_benchmark_demo.py`: 3D fringing benchmark plots
- `examples/extruded_summary_figures.py`: extra fringing-figure generator used by the docs asset workflow
- `examples/autodiff_sensitivity_demo.py`: Hartmann sensitivities
- `examples/autodiff_extruded_trajectory_demo.py`: deeper extruded autodiff target matching
- `examples/variable_field_geometry_demo.py`: Python-native geometry and field editing

## Documentation

The detailed documentation lives under [`docs/`](docs/) and covers:

- equations and physics model
- numerics and solver structure
- geometry and mesh handling
- input reference and CLI usage
- testing and validation strategy
- autodiff and performance workflows

Build locally with:

```bash
python -m sphinx -W -b html docs docs/_build/html
```

## Testing

Fast routine gate:

```bash
python -m pytest -m "unit or validation"
```

Focused coverage or solver work should stay bounded; runs over five minutes are
explicitly treated as failures for routine local development.

## License

LMX is released under the [MIT License](LICENSE).
