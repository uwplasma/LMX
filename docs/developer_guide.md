# Developer Guide

## Architecture and slimming audit

Run the live architecture audit after any module, public API, curated example,
dependency, or large generated-asset change:

```bash
.venv/bin/python scripts/audit_architecture.py --check --measure-import
```

The current checkpoint records 34 package modules and 34,084 total
package lines. The maintained stable core is 7,614 lines;
the rest is explicitly classified as research-stage extensions,
validation/evidence tooling, or visualization. The stable root surface is 30
exports and the curated catalog contains 11 workflows. Lightweight import is
about 19 ms on the audited development machine. Advanced APIs import from their
owning submodules; see the [migration guide](migration.md).

The current tracked checkout is 4,598,958 bytes, below its 4.5 MiB hard cap.
Sixty-five generated files larger than 128 KiB were bundled in the versioned release
indexed by [`release-assets.json`](release-assets.json)
and removed only after a fresh download passed archive membership, size, and
SHA-256 verification. Twenty compressed web derivatives (1,256,482 bytes total)
remain in `docs/_static/` for direct README and documentation display; full-resolution media and
field bundles remain release assets and never enter the wheel.

Maintain that boundary with:

```bash
python scripts/manage_release_assets.py --check
python scripts/manage_release_assets.py --require-uploaded
python scripts/manage_release_assets.py --verify-archive PATH_TO_ARCHIVE
```

New generated files above 128 KiB fail the manifest check until deliberately
assigned to a new versioned asset release.

The live architecture gate caps the package at 34 modules, 34,950 source lines,
7,800 maintained-core lines, 30 test files, 20,900 test lines, 13 maintenance
scripts, 30 root exports, 12 curated examples, a 4.5 MiB tracked checkout, and a
0.25 s median lazy root import. Release builds cap the wheel at 384 KiB and the
source distribution at 512 KiB. The wheel contains only `lmx/` and metadata;
the source distribution adds the README, license, and build metadata. Tests
remain in the repository because their scripts and external fixtures are not
part of the slim user distribution.
Both reject benchmark, documentation, or generated payloads. The
[media provenance index](media.md) records the web
derivatives without adding a gallery to the primary navigation.

## Architecture

Public concepts map directly to source modules:

| Module | Responsibility |
|---|---|
| `specs.py`, `config.py` | typed cases and TOML parsing |
| `mesh.py`, `operators.py` | structured meshes and discrete operators |
| `physics.py`, `linear.py` | material/field terms and linear solves |
| `solvers.py` | fully developed steady and transient solvers |
| `fringing.py` | extruded and mapped-pipe research solvers |
| `validation.py`, `freemhd.py` | analytical and independent-code evidence |
| `autodiff.py`, `scaling.py` | gradients and parallel performance |
| `io.py`, `plotting.py` | restart/output and compressed visual results |

`_fringing_types.py` is the only private shared container module. Avoid facade
files, one-function modules, and benchmark-specific logic in low-level kernels.
When a large module is simplified, preserve its public import path and separate
the structural commit from numerical changes. Docstrings should state array
shape, units, solver assumptions, and literature anchors where these are not
obvious from the type signature.

Exact B2 restarts use schema `b2_diagnostics_v5`; the pressure-linear history
has shape `(completed_steps, 5)`. The loader remains compatible with v2, and a
continued v2 run marks its unknown earlier rows `[NaN, NaN, 0, 0, -1]`.

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

The repository uses a bounded CI model:

- complete portable CI
  - runs on pushes, pull requests, and manual dispatch
  - runs all portable unit, numerics, physics, workflow, and branch-coverage tests on Python 3.10 and 3.13
- dedicated docs workflow
  - runs on pushes, pull requests, and manual dispatch
  - builds the Sphinx site as an independent status surface
- manual research-artifact workflows
  - run only through GitHub Actions `workflow_dispatch`
  - run hardware, FreeMHD, large-data, and long benchmark lanes
  - generate external validation and research artifacts

This separation keeps all portable functionality in the routine gate while
leaving hardware- or data-dependent campaigns explicit.

## Release and publishing

The repository has a conservative release workflow in
`.github/workflows/release.yml`. It is intentionally separate from routine CI:
normal pushes stay fast, while release candidates run selected validation
artifact gates before package artifacts are built.

- default push/PR CI:
  - install `.[dev]`
  - run `python scripts/run_full_test_suite.py --budget-seconds 600`
  - enforce at least 95% combined line/branch coverage on Python 3.13
  - run the same battery without coverage on Python 3.10
- docs CI:
  - install `.[docs]`
  - run `python -m sphinx -W -b html docs docs/_build/html`
- bounded release validation:
  - run the complete portable physics, numerics, workflow, and branch-coverage gate
  - build documentation with warnings as errors
  - keep external validation and large scaling outputs as checksummed release assets
- packaging:
  - build sdist and wheel artifacts
  - inspect/install the wheel in a clean environment
  - publish to TestPyPI first
  - publish to PyPI only from an explicit manual release workflow dispatch
    using PyPI Trusted Publishing

Release workflow behavior:

- `workflow_dispatch` with `publish_target = none` runs the portable validation
  gate, docs build, wheel/sdist audits, installed-wheel smoke, metadata check,
  and artifact upload.
- `workflow_dispatch` with `publish_target = testpypi` does the same checks and
  publishes to TestPyPI through Trusted Publishing.
- `workflow_dispatch` with `publish_target = pypi` does the same checks and
  publishes to PyPI through Trusted Publishing.
- a published GitHub Release runs the same gates and build checks, but PyPI
  publication is intentionally skipped unless the manual `pypi` target is
  selected.

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

Local release dry-run:

```bash
python -m pip install --upgrade build twine
rm -rf dist build lmx.egg-info
python -m build
python -m twine check dist/*
python scripts/audit_architecture.py --check \
  --wheel dist/*.whl --sdist dist/*.tar.gz
python scripts/run_full_test_suite.py --budget-seconds 600
python -m sphinx -W -b html docs docs/_build/html
```

The release workflow also installs the built wheel in a clean environment.
Open research lanes remain explicit in the validation report and authoritative
plan rather than being maintained by a separate status-dashboard subsystem.

## Test runtime baseline

The latest local evidence pass on this workstation shows:

- Python 3.10 compatibility lane: the complete battery without coverage
  instrumentation, under the 10-minute wall-clock target
- Reference coverage lane: 867 tests pass with 95.41% combined line/branch
  coverage over `lmx/` in 117.7 seconds; workflow behavior is exercised by the
  same suite

The hard rule for routine CI/CD is that the parallel workflow must stay under
10 minutes. When a new test exceeds that budget, prefer:

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
