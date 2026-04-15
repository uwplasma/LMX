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

### 2D and 3D startup movies

These README assets are generated from `examples/readme_showcase_demo.py` and
show the Hunt startup sequence in 2D and 3D from `t = 0` to `t = 2 ms`. Time
is shown in physical units, the fluid domain is outlined explicitly, and the
2D view marks the Hartmann layers at the top and bottom walls and the side
layers at the insulating side walls. In this case, the Hartmann layers are the
thin boundary layers along the walls normal to the magnetic field, where the
strongest MHD damping occurs. The README movie uses a finer `6 × 6`
cross-sectional solve with all timesteps written to the GIF so the early
boundary-layer formation is easier to follow.

<p align="center">
  <img src="docs/_static/generated/readme_hunt_startup_2d.gif" alt="LMX 2D startup movie" width="48%">
  <img src="docs/_static/generated/readme_hunt_startup_3d.gif" alt="LMX 3D startup movie" width="48%">
</p>

### Fringing-field response

The 3D fringing overview shows the imposed magnetic-field ramp, the response of
the streamwise velocity, the pressure span, and the charge/current diagnostics
for a rectangular duct in one place.

Here the magnetic field is weak upstream, ramps up inside the magnet region,
and drops again downstream. That axial field variation is what “fringing” means
in this context.

![LMX fringing-field overview](docs/_static/generated/fringing_benchmark_rect.png)

### Scaling and autodiff

The scaling panel below is a fixed-problem strong-scaling benchmark for a dense
structured-grid inductionless MHD operator. Solid lines are measured warm
runtimes and speedups; the dashed line is ideal linear speedup. The current
figure uses a `4096×4096` CPU case with `1024` operator iterations and a
`10240×10240` GPU case with `4096` operator iterations, so the device curves
come from minute-scale kernels instead of short smoke tests.

Measured warm-runtime points:

- CPU: `56.96 s`, `45.89 s`, `62.82 s`, `62.51 s` at `1, 2, 4, 8`
- GPU: `58.19 s`, `31.25 s` at `1, 2`

On this workstation the CPU curve reaches its best point at `2` logical
devices and then flattens, which is consistent with a bandwidth-limited kernel.
The GPU curve keeps a cleaner strong-scaling trend on the larger fixed problem.

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
- mapped-pipe external comparison kept explicitly qualitative
- widened bounded manual fringing campaign at `Ha = 10, 20, 30`, `resolution = 8`

The widened bounded manual campaign is intentionally stricter than the release
gate. On the current tree it confirms the 3D fringing set at `Ha = 10, 20, 30`
for `rect_duct`, `layered_duct`, and `pipe_ogrid`, and the repaired fully
developed Hunt low-resolution row now also passes that heavier conservation
check. On the bounded `Ha = 10`, `resolution = 8` manual run, the Hunt
interface-current residual is now `≈ 1.27e-2` instead of the old failing
`≈ 4.20e-1`.

The heavier 3D validation campaign is generated with:

```bash
python examples/extruded_validation_campaign.py --output artifacts/examples/extruded_validation_campaign --ha-values 10,20 --resolutions 10,14 --fringing-nx 5
python scripts/run_manual_solver_family_validation.py --output artifacts/manual_validation/solver_family_summary.json --ha-values 10,20 --resolutions 8,12 --include-fringing --fringing-geometries rect_duct,layered_duct,pipe_ogrid --fringing-nx 5 --max-steps 12 --potential-iterations 48 --coupling-iterations 8 --write-csv --write-plot
```

## Examples

Useful entry points:

- `examples/readme_showcase_demo.py`: regenerates the README media bundle
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
