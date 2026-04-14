# LMX

LMX is a JAX-native toolkit for laminar inductionless magnetohydrodynamics on
structured meshes. Version `1.0` targets a research-grade core for fully
developed duct flows, benchmark-quality validation, explicit runtime
diagnostics, restartable CLI workflows, and a clean differentiable lane for
inverse problems and design studies.

The `1.0` ship gate is now closed on the fast release lane:

- fast validation suite passes within the five-minute budget
- docs build cleanly
- combined `lmx/` + `scripts/` coverage is at `90%`
- CLI and restart smokes pass on the shipped TOML workflow
- publication-facing strong-scaling and autodiff artifacts are committed

## What LMX is for

- Hartmann, Shercliff, and Hunt benchmark problems
- layered conducting and insulating wall models
- scripted and input-file-driven studies
- publication-ready plots, movies, and benchmark reports
- differentiable steady and transient workflows in JAX
- CPU and multi-GPU strong-scaling benchmark tooling
- executable fringing-field benchmark scaffolds for the next solver phase

## Current solver status

- `fully_developed_inductionless`
  - default solver for `rect_duct` and `layered_duct`
  - steady and transient streamwise-velocity / electric-potential solves
  - research path for Hartmann, Shercliff, and Hunt cases
- `reduced_inductionless`
  - reduced-model alternative for lightweight sweeps, regression checks, and
    side-by-side method studies
- `extruded_inductionless`
  - staged next solver family for 3D/fringing-field work
  - current repo ships a first rectangular-duct low-Re 3D projection slice
  - layered fringing ducts now use the same projection path through the Python
    API
  - full CLI/TOML support, mapped-pipe support, and production-grade 3D
    validation remain post-`1.0` work

## Installation

### Minimal install

```bash
git clone https://github.com/uwplasma/lmx
cd LMX
python -m pip install -e .
```

LMX supports Python `3.10+`. On Python `3.10`, TOML parsing falls back
automatically to `tomli`, and the package now accepts the installed `jax`
family directly rather than pinning a narrow version window.

Recent compatibility checks were run on:

- local Python `3.13` with JAX `0.9.2`
- remote Python `3.10.12` on `office` with JAX `0.6.2` and two RTX A4000 GPUs

### Development install

```bash
git clone https://github.com/uwplasma/lmx
cd LMX
python -m pip install -e '.[dev,plotting,docs,extras]'
```

## Quick start

### Run from the CLI

```bash
lmx examples/hartmann_case.toml
lmx examples/shercliff_case.toml
lmx examples/hunt_case.toml
lmx run hartmann --ha 20 --verbose
lmx run hunt --ha 20 --verbosity debug
JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 lmx examples/hartmann_case.toml
XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 lmx examples/hartmann_case.toml
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 lmx examples/hunt_case.toml
ssh office 'cd /home/rjorge/tmp/lmx_scaling_repo && PYTHONPATH=/home/rjorge/tmp/lmx_scaling_repo CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda python3 -m lmx examples/hunt_case.toml'
```

The module entrypoint works as well:

```bash
python -m lmx examples/hartmann_case.toml
```

### Run from Python

```python
from lmx.cases import make_hartmann_case
from lmx.config import LoggingSpec
from lmx.runtime_logging import StreamingSolverLogger
from lmx.solvers import solve_steady

case = make_hartmann_case(ha=20.0, ny=48, nz=48)
logger = StreamingSolverLogger(LoggingSpec.from_user_controls(verbose=True, verbosity="debug"))
solution = solve_steady(case, logger=logger)
print(solution.diagnostics.residual_history[-1])
```

### Run tests and docs

```bash
python -m pytest
python -m sphinx -W -b html docs docs/_build/html
```

### CI modes

The repository now uses two lanes:

- fast default CI on pushes and pull requests
  - unit and validation tests
  - docs
- manual research-artifact workflows via GitHub Actions `workflow_dispatch`
  - regression and physics suites
  - heavy validation artifact generation
  - benchmark artifact generation
  - extended coverage collection

This keeps the default gate practical while preserving reproducible benchmark
and reporting workflows for release work and paper figures.

Current local baseline:

- full fast test suite: about `31 s`
- full coverage lane: about `37 s`

Both are intentionally kept below a five-minute routine validation budget.

## Geometry and mesh preview

LMX now ships a dedicated geometry preview workflow:

```bash
python examples/geometry_preview_demo.py --output artifacts/examples/geometry_preview
python examples/geometry_preview_demo.py --with-post-run --post-case hartmann --output artifacts/examples/geometry_preview_full
```

That example shows:

- a rectangular Hartmann duct
- a layered Hunt duct with explicit wall regions
- a mapped pipe O-grid

The default invocation is preview-only so it stays fast. Add `--with-post-run`
to append a short steady Hartmann or Hunt solve and matching overview plots in
the same output tree.

This is the intended preprocessing/postprocessing bridge for users who want to
inspect the geometry and mesh before launching longer runs.

For Python-native variable fields and custom geometry edits, use:

```bash
python examples/variable_field_geometry_demo.py --output artifacts/examples/variable_field_geometry
```

That driver shows how to:

- start from a benchmark constructor
- modify geometry fields with `dataclasses.replace(...)`
- define an analytic magnetic-field callable
- preview the mesh and material layout
- run a short solve and emit the same summary/plot artifacts as the other examples

## Typical outputs

An LMX run can produce:

- live solver logs with verbosity control
- JSON summaries
- restartable `.npz` state bundles
- ParaView VTK output
- CSV centerline and midplane profiles
- publication-style plots and GIF movies from the examples

The runtime logger is intentionally detailed. It reports solver-family
information, linear-solve residuals, integral MHD diagnostics, conservation
checks, initial and final linear/potential residuals, and transient progress in
a format intended for long research runs.

At `verbosity = "detailed"` or `verbosity = "debug"`, each step reports:

- initial and final residuals for the potential and velocity solves
- linear iteration counts
- `max|div J|`
- charge-balance residual
- interface-current continuity residual
- volumetric flow rate, mean current magnitude, and Lorentz power

## Performance and autodiff examples

### Strong scaling

The repository ships a publication-oriented strong-scaling example for the
dominant stencil/linear-solve kernel:

```bash
python examples/strong_scaling_demo.py --output artifacts/examples/strong_scaling_cpu
python examples/strong_scaling_demo.py --remote-host office --output artifacts/examples/strong_scaling_full
```

This writes raw timing JSON plus polished `PNG`/`PDF` scaling plots suitable for
docs and paper drafts.

![LMX strong scaling](docs/_static/generated/strong_scaling.png)

Current publication artifact highlights:

- local CPU warm-runtime sweep on a `1024 x 1024` cross-section:
  - `1` device: `0.0898 s`
  - `4` devices: `0.0563 s`
  - `8` devices: `0.0549 s`
- remote GPU warm-runtime sweep on a `2048 x 2048` cross-section:
  - `1` GPU: `0.0524 s`
  - `2` GPUs: `0.0392 s`

The remote GPU workflow automatically prefers the highest-index single GPU for
the one-device baseline, which avoids workstation display contention on desktop
GPU hosts.

Standard CLI runs are not themselves a multi-device scaling benchmark, but they
inherit the active JAX backend from the shell. Use `examples/strong_scaling_demo.py`
for publication scaling studies and use `JAX_PLATFORMS` / `CUDA_VISIBLE_DEVICES`
to steer normal CLI runs to CPU or GPU backends. If you want multiple logical
CPU devices visible to JAX from the CLI, also set
`XLA_FLAGS=--xla_force_host_platform_device_count=<N>`.

The recent remote smoke validation on `office` confirmed that a `512 x 512`
GPU kernel run is faster than the matching local one-device CPU run on the
current post-`1.0` tree:

- local CPU, `512 x 512`, `32` iterations:
  - `warm_seconds ≈ 4.31e-3`
- remote office GPU, `512 x 512`, `32` iterations:
  - `warm_seconds ≈ 6.65e-4`

Very small multi-GPU problems can still scale poorly. Use the dedicated scaling
example, not routine CLI runs, when you need publication-quality strong-scaling
data.

### Autodiff sensitivity and inverse design

The repository also ships a differentiable Hartmann example:

```bash
python examples/autodiff_design_demo.py --output artifacts/examples/autodiff_design
python examples/autodiff_sensitivity_demo.py --output artifacts/examples/autodiff_sensitivity
python examples/autodiff_profile_design_demo.py --output artifacts/examples/autodiff_profile_design
python examples/autodiff_fringing_design_demo.py --output artifacts/examples/autodiff_fringing_design
python examples/autodiff_fringing_response_demo.py --output artifacts/examples/autodiff_fringing_response
```

Together, those examples demonstrate:

- `jax.grad` sensitivity of mean velocity with respect to Hartmann number
- finite-difference cross-checks for autodiff sensitivities with respect to
  Hartmann number and forcing
- inverse recovery of a synthetic forcing parameter from a target velocity profile
- full-profile inverse design recovering both forcing and Hartmann number
- fringing-history inverse design recovering axial field-profile parameters
- fringing multi-observable inverse design recovering axial field-profile
  parameters against both mean-velocity and current-response histories
- polished `PNG`/`PDF` summary figures for publication use

![LMX autodiff summary](docs/_static/generated/autodiff_summary.png)

Current publication artifact highlight:

- synthetic target forcing: `1.0`
- recovered forcing after `24` gradient steps: `0.999863`
- final profile loss: `2.9e-12`

### Fringing-field scaffold

The repository now also ships a publication-facing fringing benchmark scaffold:

```bash
python examples/fringing_benchmark_demo.py --output artifacts/examples/fringing_benchmark
python examples/fringing_benchmark_demo.py --geometry-kind layered_duct --output artifacts/examples/fringing_benchmark_layered
```

That example now writes the first retained rectangular-duct
`extruded_inductionless` 3D projection slice through the explicit
`solve_extruded_inductionless(...)` Python entry point: a stacked axial field
bundle with `u`, `v`, `w`, `p`, `phi`, current, Lorentz, charge-balance
diagnostics, axial-current histories, wall-current leakage audits, contour
plots for `u(x, y, zmid)` and `u(x, ymid, z)`, and a compact validation
summary for the slice.

That workflow generates a smooth axial fringing profile together with
cross-sectional response metrics. Rectangular and layered ducts now both go
through the low-Re pressure-velocity-potential projection slice in the Python
API, so the workflow is no longer limited to a single fluid-only cross-section.

## User workflows

### Input-file workflow

The primary executable path is:

```bash
lmx examples/hartmann_case.toml
```

The TOML schema is documented in
[docs/input_reference.md](docs/input_reference.md). The important high-level
blocks are:

- `[case]`
- `[geometry]`
- `[magnetic_field]`
- `[solver]`
- `[time_stepper]`
- `[output]`
- `[logging]`
- `[restart]`

The `[logging]` block supports both:

- `verbose = true|false`
- `verbosity = "quiet" | "normal" | "detailed" | "debug"`

Use `verbose = false` for quiet batch runs and `verbosity = "debug"` when you
need the most detailed live runtime output.

The `[time_stepper]` block also now uses bounded step-count logic:

- `t_final` is a stop horizon, not a target that is rounded up
- `max_steps` is treated as a hard ceiling
- fractional `t_final / dt` ratios do not trigger a spurious extra step

### Example workflow

Examples live in [examples/README.md](examples/README.md). They are meant to be
teachable, explicit templates rather than black-box wrappers. Each example shows
how to:

- define geometry and resolution
- choose a solver family
- configure time stepping and logging
- save fields and diagnostics
- generate 2D and 3D visualizations

## Documentation

- [User and theory docs](docs/index.md)
- [Getting started](docs/getting_started.md)
- [Theory and equations](docs/theory.md)
- [Numerics and implementation](docs/numerics.md)
- [Geometry and mesh workflows](docs/geometry.md)
- [Input reference](docs/input_reference.md)
- [Case cookbook](docs/case_cookbook.md)
- [Testing and validation strategy](docs/testing.md)
- [Benchmark matrix](docs/benchmark_matrix.md)
- [Performance and scaling](docs/performance.md)
- [Autodiff and inverse design](docs/autodiff.md)
- [Developer guide](docs/developer_guide.md)
- [Validation report](docs/validation_report.md)

## Research directions and references

LMX is being positioned around the benchmark ladder used in liquid-metal MHD
verification and validation:

- [Samper et al., benchmark review for MHD validation and verification](https://www.scipedia.com/wd/images/b/b8/Draft_Samper_360028846_6045_art042.pdf)
- [JAX gradient checkpointing](https://docs.jax.dev/en/latest/gradient-checkpointing.html)
- [Lineax linear solvers](https://docs.kidger.site/lineax/api/solvers/)
- [Diffrax adjoints](https://docs.kidger.site/diffrax/api/adjoints/)
- [Φ-Flow differentiable PDE tooling](https://proceedings.mlr.press/v235/holl24a.html)

The near-term research targets are:

- benchmark-grade fully developed Hartmann, Shercliff, and Hunt cases
- laminar fringing-field benchmarks in square ducts and pipes
- differentiable inverse studies over magnetic field, geometry, and wall
  conductance parameters

## Security and reproducibility notes

- The repository avoids hard-coded machine-local absolute paths in public docs
  and examples.
- CLI and plotting utilities are allowed to use pragmatic non-differentiable
  utilities where that improves robustness, while the solver core remains JAX
  based.
- External benchmark comparisons are treated as secondary validation assets and
  are kept separate from the governing solver implementation.
