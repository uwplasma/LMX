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
  - steady and transient solver-family implementations
- `lmx/io.py`
  - restart/state bundles and output serialization
- `lmx/runtime_logging.py`
  - live terminal logging
- `lmx/validation.py`
  - analytical and reference-output comparison helpers

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

### `legacy_reduced`

Retained for regression only.

### `extruded_inductionless`

Planned 3D/fringing-field solver family.

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
  - covers unit, regression, and validation tests plus the documentation build
- manual research-artifact workflows
  - run only through GitHub Actions `workflow_dispatch`
  - run the heavier physics suite
  - generate benchmark, validation-artifact, and extended coverage outputs

This separation is intentional. The fast lane protects the `1.0` public
surface, while the manual lane preserves reproducible research artifacts
without exhausting routine CI runtime.
