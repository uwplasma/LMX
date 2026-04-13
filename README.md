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
- `legacy_reduced`
  - retained only for regression and historical comparison
- `extruded_inductionless`
  - staged next solver family for 3D/fringing-field work
  - current repo ships executable fringing benchmark scaffolds, not the full solver yet

## Installation

### Minimal install

```bash
git clone https://github.com/uwplasma/lmx
cd LMX
python -m pip install -e .
```

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
checks, and transient progress in a format intended for long research runs.

## Performance and autodiff examples

### Strong scaling

The repository ships a publication-oriented strong-scaling example for the
dominant stencil/linear-solve kernel:

```bash
python examples/strong_scaling_demo.py --output artifacts/examples/strong_scaling_cpu
python examples/strong_scaling_demo.py --remote-host <your_gpu_host> --output artifacts/examples/strong_scaling_full
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

### Autodiff sensitivity and inverse design

The repository also ships a differentiable Hartmann example:

```bash
python examples/autodiff_design_demo.py --output artifacts/examples/autodiff_design
python examples/autodiff_sensitivity_demo.py --output artifacts/examples/autodiff_sensitivity
```

Together, those examples demonstrate:

- `jax.grad` sensitivity of mean velocity with respect to Hartmann number
- finite-difference cross-checks for autodiff sensitivities with respect to
  Hartmann number and forcing
- inverse recovery of a synthetic forcing parameter from a target velocity profile
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
```

That workflow generates a smooth axial fringing profile together with stationwise
cross-sectional response metrics. It is the executable bridge from the current
fully developed solver family to the future `extruded_inductionless` research
solver.

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
- [Theory and equations](docs/theory.md)
- [Input reference](docs/input_reference.md)
- [Case cookbook](docs/case_cookbook.md)
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
