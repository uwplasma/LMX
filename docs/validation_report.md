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
- Native mesh-convergence study summaries now include estimated Hartmann-layer and
  side-layer cell counts for the duct cases.
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
- The current convergence-aware `solve_steady` implementation confirms that the
  native Hunt gap is not just a steady-stop criterion issue; the remaining work is
  in the update physics/control law, not only in iteration bookkeeping.
- The native Hunt case now uses wall-conductance-ratio semantics by default
  instead of treating `0.05` as a raw wall conductivity. That corrects the case
  API, but the retained Hunt validation gap persists, which points back to solver
  fidelity rather than case normalization.
- The convergence artifacts now make the mesh-side diagnosis sharper. On the
  current Hunt `Ha20` sweep, the reported Hartmann-layer resolution grows from
  about `3.2` to `9.7` cells and the side-layer resolution from about `5.2` to
  `15.8` cells between `16^2` and `48^2` fluid resolutions, while the validation
  errors barely improve. That points to solver/update fidelity as the dominant
  remaining issue for native Hunt, not just missing boundary-layer clustering.
- The current retained operator update improves that diagnosis and the solver at
  the same time:
  - diffusion and potential coefficients now use actual center-to-center spacing
    on nonuniform meshes instead of uniform-spacing shortcuts
  - the masked velocity Laplacian now uses half-cell wall distances for
    no-slip-style boundaries instead of treating boundary values as if they lived
    at neighboring cell centers
  - Hartmann `Ha20` remains accepted with `l2_error ≈ 3.9e-3`
  - native Hunt `Ha20` improves to about `y_l2 ≈ 1.05e-1`, `z_l2 ≈ 2.90e-1`
  - native Hunt `Ha100` improves to about `y_l2 ≈ 1.90e-1`, `z_l2 ≈ 3.62e-1`
  - the Hunt `Ha20` convergence sweep now shows strong improvement in the `y`
    profile with refinement instead of the previous near-zero observed order,
    although the `z` profile still lags
- The current short-time closed-channel validation reports are useful regression
  signals, but they are not yet final acceptance criteria for all case families.

## Planned improvements

- Add stronger parity thresholds once the solver is stable across more geometries.
- Extend the current mesh-convergence tooling to pseudo-time convergence studies.
- Expand native LMX validation to additional mapped-geometry cases.
- Improve high-Ha Hunt fidelity without introducing case-specific hardcoded limits.
- Keep the external backend checks optional and separate from the core LMX identity.
