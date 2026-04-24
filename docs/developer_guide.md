# Developer Guide

## Architecture

The codebase is organized around a small number of core modules:

- `lmx/specs.py`
  - dataclasses and typed configuration models
- `lmx/config.py`
  - TOML parsing and schema validation
- `lmx/mesh.py`
  - structured mesh generation
- `lmx/operators.py`
  - discrete gradient, divergence, and diffusion operators
- `lmx/physics.py`
  - magnetic fields, materials, and benchmark-specific physical helpers
- `lmx/linear.py`
  - iterative linear solves
- `lmx/solvers.py`
  - compatibility facade for steady and transient solver-family implementations
- `lmx/_solvers.py`
  - current fully developed solver implementation pending the next extraction
- `lmx/io.py`
  - restart/state bundles and output serialization
- `lmx/runtime_logging.py`
  - live terminal logging
- `lmx/validation.py`
  - compatibility facade for analytical and reference-output comparisons
- `lmx/_validation.py`
  - current validation implementation pending the next extraction
- `lmx/_fringing_types.py`
  - stable private data containers for the extruded/fringing solver family

## Planned module split

The first split stage is in place: the historical public modules remain as
compatibility facades, while implementation lives in private modules such as
`lmx/_fringing.py`, `lmx/_autodiff.py`, `lmx/_plotting.py`, `lmx/_solvers.py`,
and `lmx/_validation.py`. The first private extraction is
`lmx/_fringing_types.py`, which holds the extruded/fringing problem, bundle,
solution, and validation containers while the public `lmx.fringing` facade
continues to resolve the same names. The next stages should extract cohesive
submodules from those private implementations without changing public import
paths.

Target structure:

- `lmx/fringing/`
  - problem builders, projection solve, conservative metrics, benchmark gates,
    and reference comparison helpers
- `lmx/solvers/`
  - a thin public facade plus fully developed solve logic, potential helpers,
    diagnostics, and logging adapters
- `lmx/validation/`
  - profile comparisons, reference-data loading, and report construction
- `lmx/plotting/`
  - profiles, benchmark figures, fields, media, and scaling/autodiff panels
- `lmx/autodiff/`
  - objectives, gradient checks, design loops, and uncertainty propagation

Refactor rules:

- preserve existing `import lmx` and module-level public paths during the
  split cycle
- do not mix numerical changes with file moves
- move or add tests with each extracted module
- keep benchmark-specific acceptance logic out of the low-level solver kernels
- update module docstrings with governing equations, shape conventions, units,
  and literature anchors

## Solver families

### `fully_developed_inductionless`

This is the default research path.

- cross-sectional unknowns: `u(y, z)` and `phi(y, z)`
- geometries:
  - `rect_duct`
  - `layered_duct`
- intended benchmarks:
  - Hartmann
  - Shercliff
  - Hunt

### `extruded_inductionless`

Current 3D/fringing-field solver family for rectangular ducts, layered ducts,
and mapped-pipe research slices.

## Differentiable lane

The intended differentiable core is:

- `lmx/operators.py`
- `lmx/physics.py`
- `lmx/linear.py`
- `lmx/solvers.py`

The CLI, plotting, reporting, and docs utilities are intentionally allowed to be
more pragmatic. They do not define the differentiable contract.

Useful references:

- [JAX gradient checkpointing](https://docs.jax.dev/en/latest/gradient-checkpointing.html)
- [Lineax solvers](https://docs.kidger.site/lineax/api/solvers/)
- [Diffrax adjoints](https://docs.kidger.site/diffrax/api/adjoints/)

## Performance lane

The current performance rules are:

- keep core operator assembly vectorized
- avoid dense matrix materialization for the default solver path where possible
- keep file writing and plotting out of the JIT/differentiable core
- prefer NumPy or SciPy for CLI-only postprocessing utilities if that improves
  startup or runtime cost without affecting the core solver

## Validation philosophy

Primary correctness comes from:

- analytical/semi-analytical benchmark checks
- mesh/time/convergence studies
- conservation diagnostics

Reference-solver comparisons are secondary benchmark evidence and should be
based on observable outputs, not source-coupled behavior.

## CI strategy

The repository uses a split CI model:

- fast default CI
  - runs on pushes and pull requests
  - covers unit and validation tests
- dedicated docs workflow
  - runs on pushes, pull requests, and manual dispatch
  - builds the Sphinx site as an independent status surface
- manual research-artifact workflows
  - run only through GitHub Actions `workflow_dispatch`
  - run the heavier regression and physics suites
  - generate benchmark, validation-artifact, and extended coverage outputs

This separation is intentional. The fast lane protects the `1.0` public
surface, the docs lane keeps the documentation badge honest, and the manual
lane preserves reproducible research artifacts without exhausting routine CI
runtime.

## Release and publishing

The repository has a conservative release workflow in
`.github/workflows/release.yml`. It is intentionally separate from routine CI:
normal pushes stay fast, while release candidates run selected validation
artifact gates before package artifacts are built.

- default push/PR CI:
  - install `.[dev]`
  - run `python -m pytest -m "unit or validation"`
  - keep the lane below five minutes
- docs CI:
  - install `.[docs]`
  - run `python -m sphinx -W -b html docs docs/_build/html`
- manual release validation:
  - run physics/regression suites
  - regenerate selected validation artifacts
  - summarize FreeMHD observable-gate pass/fail counts in the release report
  - run broad branch coverage with `--cov-fail-under=95`
- packaging:
  - build sdist and wheel artifacts
  - inspect/install the wheel in a clean environment
  - publish to TestPyPI first
  - publish to PyPI from tagged releases using PyPI Trusted Publishing

Release workflow behavior:

- `workflow_dispatch` with `publish_target = none` runs the selected
  validation gate, builds docs, builds the wheel/sdist, checks metadata, and
  uploads release artifacts.
- `workflow_dispatch` with `publish_target = testpypi` does the same checks and
  publishes to TestPyPI through Trusted Publishing.
- a published GitHub Release runs the same gates and publishes to PyPI through
  Trusted Publishing.

Before first publication, configure the GitHub `testpypi` and `pypi`
environments and register this repository as a trusted publisher in TestPyPI
and PyPI. Do not bypass the selected validation-artifact gate for a public
release.

External executable parity artifacts are portable: when the configured
FreeMHD reference tree is not available on a runner, the suite writes an
explicit `skipped` summary instead of failing the packaging workflow. Release
candidates intended for publication should run the same workflow on a runner
with `LMX_FREEMHD_INSTALL_DIR` or `LMX_FREEMHD_PROCESSED_ROOT` configured so
the parity artifact is completed rather than skipped.

## Test runtime baseline

The latest local evidence pass on this workstation shows:

- default push/PR lane, `python -m pytest -m "unit or validation" -q`:
  passes and remains inside the five-minute guard
- broad coverage lane over `lmx/` and `scripts/`: passes at about `95.4%`
  combined line/branch coverage

The hard rule for routine validation is that these lanes must stay under five
minutes. When a new test exceeds that budget, prefer:

- synthetic or manufactured-solution fixtures
- monkeypatched orchestration tests for CLI/reporting/example paths
- direct operator/kernel tests instead of full solver runs

That pattern is now the default test-design rule. The cheap numerical core
(`lmx/operators.py`, `lmx/linear.py`) should be validated primarily through
manufactured fields and direct kernel contracts, while the heavier solver
families are covered through a smaller number of focused acceptance tests.

Do not use the default test suite as a vehicle for long benchmark or artifact
generation. Those belong in the manual research-artifact workflows.

## Logging surface

LMX now exposes the same runtime logging controls through all public entry
points:

- TOML: `[logging] verbose = true|false`, `verbosity = "quiet"|"normal"|"detailed"|"debug"`
- CLI: `--quiet`, `--verbose`, `--verbosity ...`
- Python: `LoggingSpec.from_user_controls(...)`

Use `verbosity="debug"` only for active solver investigation; it is intentionally
the noisiest path and prints extra runtime ratios for current and Lorentz
diagnostics.
