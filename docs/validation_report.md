# Validation Report

## What is validated

- Hartmann: analytical profile checks and smoke-case runtime behavior.
- Shercliff: insulating-wall duct validation against processed paper data.
- Hunt: conducting-wall duct validation against processed paper data.
- Optional external backend comparisons: recovered case archives and regenerated
  backend runs when available locally.

## Current validation coverage

- Structured duct meshes with optional conducting-wall layers.
- Laminar inductionless fields `U`, `phi`, `J`, and `J x B`.
- Analytical comparison JSON generation from the CLI and the validation runner.
- Hartmann acceptance reports with explicit `l2` and `linf` thresholds.
- Sampled midplane and line-cut comparison JSON generation when processed slice data
  is present.
- Native mesh-convergence study summaries for the currently supported duct cases.
- CSV and ParaView outputs for centerline and field inspection.
- Validation artifact generation in GitHub Actions.
- Benchmark artifact generation in GitHub Actions.

## External backend context

Recovered external assets are used as validation backends only. They help compare LMX
against archived paper cases, but they are not required for native solver use.

The backend harness currently supports:

- asset discovery and setup reporting
- optional short-smoke case execution
- sampled line-cut comparison reports
- structured skipped reports when the backend assets are not present on the runner

## Current quality status

- Hartmann is stable on the default fine mesh and serves as the main smoke test.
- Shercliff is stable and gives a strong insulating-wall validation path.
- Hunt is bounded and now produces finite comparison metrics, but the remaining error
  shows that conducting-wall fidelity still needs solver work.
- The current short-time closed-channel validation reports are useful regression
  signals, but they are not yet final acceptance criteria for all case families.

## Planned improvements

- Add stronger parity thresholds once the solver is stable across more geometries.
- Extend the current mesh-convergence tooling to pseudo-time convergence studies.
- Expand native LMX validation to additional mapped-geometry cases.
- Improve high-Ha Hunt fidelity without introducing case-specific hardcoded limits.
- Keep the external backend checks optional and separate from the core LMX identity.
