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
- Native pseudo-time convergence summaries now exist for fixed-resolution duct
  studies, so temporal sensitivity can be separated from mesh sensitivity.
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
- The new fixed-resolution pseudo-time convergence runner clarifies the remaining
  Hunt behavior further:
  - at Hunt `Ha20` and fixed `48^2` resolution, reducing `dt` from `0.002` to
    `0.0005` improves the `z` profile only modestly (`z_l2 ≈ 0.209 -> 0.175`)
  - over the same sweep, the `y` profile degrades (`y_l2 ≈ 0.023 -> 0.108`)
  - this means the remaining Hunt discrepancy is not a simple “smaller dt is
    always better” problem; the current pseudo-time path still carries a genuine
    transient/control-law tradeoff that needs solver-side treatment
- The new control-sweep runner makes the same point from the coupling side:
  - for Hunt `Ha20` at `48^2`, increasing `outer_iterations` improves `z_l2`
    from about `0.283` at `2` iterations to about `0.185` at `10`
  - over the same sweep, `y_l2` improves strongly at first (`0.099 -> 0.023`
    by `6` outer iterations) and then degrades again (`0.042`, `0.062`)
  - this is a real coupled-control tradeoff, not a single-parameter monotone fix
- A later local steady-momentum experiment narrows the diagnosis further:
  - Hartmann `Ha20` at `96^2` stayed accepted (`l2 ≈ 3.87e-3`) when the
    pseudo-time identity term was removed in a local experiment
  - native Hunt `Ha20` at `48^2` stayed essentially unchanged
    (`y_l2 ≈ 2.25e-2`, `z_l2 ≈ 2.09e-1`)
  - that larger solver rewrite was not retained, which means the Hunt gap is
    not fixed by changing only the linear stencil form of the momentum update
- The CI artifact summary now surfaces both pseudo-time convergence and solver
  control sweeps directly, so these Hunt diagnostics appear in the normal
  validation report bundle instead of only in raw per-run JSON files. The sweep
  summary now also reports the best interior `y_l2` and `z_l2` points instead of
  only the first and last parameter values, which is important because the
  retained Hunt control sweeps are non-monotone.
- An attempted direct Hartmann `Ha20`, `32^2` analytical guardrail exposed a
  broader current limitation: the retained solver on `main` does not yet hold
  the present Hartmann acceptance target at that refinement level. The failing
  test was not kept, but the CI summary now reports how many sweep levels pass
  acceptance so this blocker is visible in normal artifacts.
- The Hartmann refinement failure is now characterized a bit more directly:
  - `16^2` remains acceptable
  - `32^2` develops a sign-changing centerline and fails badly
  - `48^2` is positive again but still outside the current analytical target
  - `96^2` returns to the previously good fine-mesh behavior
  This is now reflected in explicit sign-pathology metrics in the validation
  summaries, so oscillatory branches can be identified directly instead of only
  through aggregate L2 error.
- A retained sensitivity result now sharpens the likely root cause:
  - on the problematic Hartmann `Ha20`, `32^2` branch, the solution depends
    strongly on `potential_iterations`
  - `50` iterations is acceptable, `200` is strongly oscillatory, `400` is much
    better but still outside the analytical target, and `800` is acceptable
  - this points to the current electric-potential coupling budget as a central
    part of the blocker, not just generic pseudo-time controls
  - CI now runs this Hartmann `potential_iterations` sweep as a normal artifact
    so the blocker stays visible during future solver work
- Native validation summaries now also report a normalized electric-potential
  equation residual from the latest solve step:
  - the current bad Hartmann `Ha20`, `32^2` branch lands around
    `potential_residual ≈ 6.7e-2`
  - the current native Hunt `Ha20`, `32^2` branch lands around
    `potential_residual ≈ 5.6e-1`
  - that metric is not an acceptance threshold by itself, but it now makes the
    electric-potential solve quality visible in the same artifact bundle as the
    profile/pathology metrics and gives the next solver iteration a more direct
    signal than profile L2 alone
- The boundary gradient operator on clustered meshes is now also corrected to use
  center-to-center spacing at the domain edges. This did not materially shift the
  current retained duct validation metrics, but it removes a nonuniform-mesh
  inconsistency in the electric-field reconstruction near side boundaries.
- The current short-time closed-channel validation reports are useful regression
  signals, but they are not yet final acceptance criteria for all case families.

## Planned improvements

- Add stronger parity thresholds once the solver is stable across more geometries.
- Extend the current mesh-convergence tooling to pseudo-time convergence studies.
- Use the new pseudo-time convergence runner to decide which remaining Hunt
  discrepancies are temporal versus spatial before changing solver defaults.
- Expand native LMX validation to additional mapped-geometry cases.
- Improve high-Ha Hunt fidelity without introducing case-specific hardcoded limits.
- Keep the external backend checks optional and separate from the core LMX identity.
