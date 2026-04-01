# Developer Guide

## Package structure

- `lmx.mesh`: structured and mapped-structured mesh builders.
- `lmx.specs`: immutable case, region, boundary, and time-step dataclasses.
- `lmx.physics`: conductivity, material, and magnetic-field helpers.
- `lmx.operators`: mesh-aware finite-volume style kernels.
- `lmx.solvers`: laminar inductionless solver entrypoints.
- `lmx.io`: ParaView XML and CSV outputs.
- `lmx.validation`: analytical validation and optional external comparison helpers.
- `lmx.reference_data`: processed paper-data loaders.

## Data layout

- Cross-section fields use shape `(ny, nz)` for duct-style problems.
- Mapped geometries keep static shapes so JAX compilation remains effective.
- Region masks, conductivity, and boundary metadata stay on the same cell-centered
  layout where possible.

## JAX strategy

- Use `jax.jit` for hot paths and `jax.lax.scan` for time marching.
- Prefer matrix-free operators and preconditioners that preserve static shapes.
- Keep boundary-condition handling explicit instead of encoding geometry-specific
  hacks in scripts.
- Use `lineax` when the linear system formulation benefits from it.
- Use `equinox` or `diffrax` only when they improve the solver or optimization
  workflow; do not add them as incidental dependencies.
- Clustered duct meshes should use actual center-to-center spacing in diffusion
  and potential operators; avoid reintroducing uniform-grid shortcuts in solver
  stencils.
- Boundary gradients on clustered meshes should also use center-to-center spacing
  between the first two cell centers, not the first cell width, so electric-field
  reconstruction remains consistent near side layers.

## Case and boundary design

- `CaseSpec` should describe physics, geometry, and initial conditions without
  assuming a specific backend.
- Boundary conditions should remain declarative and reusable across analytical runs,
  validation runs, and future geometries.
- Defaults should be derived from the case geometry and materials, not from
  fixed parity thresholds.

## Solver development rules

- Keep the solver self-consistent: the same `lmx` case should run without any
  validation-only heuristics.
- Prefer physically motivated nondimensional or geometry-derived scales over
  hardcoded limits.
- When a new case family needs special handling, move the rule into the case
  specification layer instead of a parity script.
- `solve_transient` is the fixed-step path for trajectory-like runs; `solve_steady`
  now uses the configured steady tolerance and maximum step budget rather than
  aliasing the transient path.

## Validation and benchmarking

- `pytest -m unit`: mesh, operators, I/O, and report helpers.
- `pytest -m regression`: deterministic field and profile checks.
- `pytest -m physics`: invariants and low-dimensional smoke cases.
- `pytest -m validation`: analytical and optional external comparison checks.
- `python scripts/run_validation_suite.py --output artifacts/validation`: writes
  validation CSV, JSON, and VTK artifacts.
- `python scripts/run_convergence_suite.py --output artifacts/convergence`: writes
  native mesh-convergence study summaries for the currently supported duct cases.
- `python scripts/run_time_convergence_suite.py --output artifacts/time_convergence`:
  writes native pseudo-time convergence study summaries at fixed mesh resolution.
- `python scripts/run_solver_control_sweep.py --output artifacts/control_sweep`:
  writes parameter sweeps for selected time-stepper controls when a case shows a
  nontrivial coupling tradeoff. The CI summary now reports both the first/last
  sweep points and the best `y_l2` / `z_l2` points, because the retained Hunt
  sweeps are not monotone. It also reports acceptance counts when the underlying
  sweep data includes analytical pass/fail information, which is useful for the
  current Hartmann refinement blocker.
- `python scripts/run_benchmark_suite.py --output artifacts/benchmarks/benchmark.json`:
  writes the current benchmark report.
- `python -m sphinx -W -b html docs docs/_build/html`: builds the documentation with
  the same entrypoint used by Read the Docs.

## Docs workflow

- Read the docs entry point is [`docs/index.md`](index.md).
- Local docs builds use Sphinx plus MyST Markdown.
- `.readthedocs.yaml` is the canonical cloud docs build config.
- Validation backends remain optional and are documented separately from the core
  solver workflow.

## External backend workflow

External backend tooling exists to compare LMX against archived validation cases and
historical solver outputs. Keep that logic in `scripts/` and `lmx.validation` rather
than in the core solver path so LMX remains usable as a standalone liquid-metal flow
solver.
