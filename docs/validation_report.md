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
- solver-diagnostic-first Hunt reports that pair FreeMHD run metadata with the
  native LMX validation summary before profile errors are considered

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
    so the older Jacobi-only branch remains visible during future solver work
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
- The solver now also supports an optional residual-based stopping rule for the
  electric-potential solve, together with reporting of the actual iteration
  count used in the latest step:
  - on a Hartmann `Ha20`, `32^2` probe, setting
    `potential_tolerance in {1e-2, 1e-3, 1e-4}` while allowing
    `potential_iterations = 800` improved the branch from
    `l2 ≈ 1.20` to `l2 ≈ 3.0e-2`
  - but the run used the full `800` iterations in every tested tolerance case,
    so the improvement came from the larger iteration ceiling rather than early
    convergence on the tolerance itself
  - on the same probe family, native Hunt `Ha20`, `32^2` shifted from about
    `y_l2 ≈ 6.7e-2`, `z_l2 ≈ 1.67e-1` to about
    `y_l2 ≈ 7.7e-2`, `z_l2 ≈ 1.50e-1`, again using the full `800` iterations
  - that means the retained infrastructure is useful, but the next default-solver
    change should be framed as a question of iteration budget and coupling law,
    not as “tolerance alone solved the Hartmann blocker”
- Closed-channel validation artifacts now also report a combined profile error
  from the `y` and `z` cuts. That metric already prevents one misleading solver
  conclusion:
  - a Hunt `Ha20`, `32^2` candidate with `outer_iterations = 4` and
    `potential_iterations = 400` looks better than the current default on the
    `y` profile alone (`0.038` vs `0.067`)
  - but its combined profile error is actually worse than the current default
    (`0.134` vs `0.127`)
  - so the retained conclusion is to keep the current Hunt defaults and treat
    that candidate as a documented tradeoff, not a real improvement
- A follow-on probe tightened the same conclusion on the single-region side:
  - at Hartmann `Ha20`, `48^2`, increasing `potential_iterations` from `200` to
    `400` and `800` does not improve the branch monotonically
  - the retained values were roughly:
    - `200`: `l2 ≈ 7.9e-2`
    - `400`: `l2 ≈ 1.51`
    - `800`: `l2 ≈ 5.56e-1`
  - that rules out a blanket “just raise the rect-duct `phi` budget” default
    change at this stage
  - the safer retained move is to use the sweep summaries, which now report the
    best combined closed-channel error directly, as the decision surface for the
    next solver iteration
- A later solver-control probe added one more retained tool without changing
  defaults:
  - the electric-potential solver now supports weighted Jacobi through
    `potential_relaxation`
  - the current retained probe results are mixed:
    - Hartmann `Ha20`, `32^2`, `potential_iterations=200` improves strongly as
      `potential_relaxation` drops to `0.5`
      (`l2 ≈ 1.20 -> 0.20`)
    - Hartmann `Ha20`, `48^2`, `potential_iterations=400` also improves strongly
      at `0.5` (`l2 ≈ 1.51 -> 7.7e-2`)
    - Hunt `Ha20`, `32^2` gets slightly worse in combined error
      (`0.127 -> 0.135`)
    - Hunt `Ha100`, `32^2` gets modestly better in combined error
      (`0.343 -> 0.327`)
    - Shercliff `Ha20`, `32^2` improves at the current default `225` iterations
      but degrades at `400`
  - that makes `potential_relaxation` worth keeping as a sweepable solver
    control, but not yet safe as a new default policy
- The next retained solver step improved the electric-potential backend story in
  a more structural way:
  - LMX now supports a matrix-free CG backend for the electric-potential solve
  - the current retained backend probe results are strongly split by region
    structure rather than by case name
  - single-region duct cases improved sharply:
    - Hartmann `Ha20`, `32^2`: `l2 ≈ 1.20 -> 1.4e-2` and
      `potential_residual ≈ 6.7e-2 -> 1.2e-4` when moving from Jacobi to CG
    - Hartmann `Ha20`, `48^2`: `l2 ≈ 7.9e-2 -> 7.0e-3`
    - Shercliff `Ha20`, `32^2`: combined error
      `≈ 0.844 -> 0.162`
  - multi-region Hunt cases moved in the opposite direction:
    - Hunt `Ha20`, `32^2`: combined error
      `≈ 0.127 -> 0.400`
    - Hunt `Ha100`, `32^2`: combined error
      `≈ 0.343 -> 0.377`
  - retained conclusion:
    - CG is the right default for the current single-region duct path
    - Hunt-style layered ducts should stay on the damped Jacobi path until the
      coupled conducting-wall update is improved
    - `main` now encodes that as a principled `potential_solver="auto"` policy:
      use CG when the solved cross-section is a single fluid region, and keep
      Jacobi when explicit solid layers are present
- The next retained control pass widened the Hunt diagnosis without changing
  defaults:
  - `velocity_update_limit` is now a first-class control-sweep parameter
  - retained Hunt `Ha20`, `32^2` results:
    - `5e-4`: combined error `≈ 0.135`
    - `1e-3`: combined error `≈ 0.131`
    - `2e-3`: combined error `≈ 0.127`
    - `4e-3`: combined error `≈ 0.127` but slightly worse than `2e-3`
  - retained Hunt `Ha100`, `32^2` results:
    - `1e-3`: combined error `≈ 0.343`
    - `2e-3`: combined error `≈ 0.342`
    - `4e-3`: combined error `≈ 0.343`
  - retained interpretation:
    - the velocity-update cap matters enough to keep as part of the documented
      control surface
    - but it does not create a strong new default change on its own
    - the remaining Hunt discrepancy is still in the coupled update law, not in
      a missing sweep parameter
- The next retained tooling step made that tradeoff easier to measure directly:
  - LMX now has `scripts/run_solver_grid_sweep.py` for two-parameter solver
    control grids
  - the first retained Hunt grid used
    `outer_iterations in {4, 6}` against
    `potential_relaxation in {1.0, 0.5}`
  - retained Hunt `Ha20`, `32^2` result:
    - the current default-like point remains best in this grid at
      combined error `≈ 0.127`
    - lowering `potential_relaxation` helps the `y` profile for some points but
      degrades the combined error overall
  - retained Hunt `Ha100`, `32^2` result:
    - `potential_relaxation = 0.5` improves the combined error from
      `≈ 0.343` to `≈ 0.327`
    - but changing `outer_iterations` from `4` to `6` is essentially neutral
  - retained conclusion:
    - there is still no cross-case Hunt default shift worth keeping
    - the remaining gap is in the update law, not in an untried interaction
      among the currently exposed controls
- CI/reporting now supports those grids directly:
  - `summarize_ci_artifacts.py` can ingest a two-parameter control-grid summary
  - the markdown artifact now reports the best combined, `y`, and `z` points for
    the grid, including the parameter pair where they occur
  - this keeps the current Hunt tradeoff surface visible in normal artifacts
    instead of requiring manual JSON inspection
- A later solver change improved the nonuniform electric-potential discretization
  itself:
  - layered and clustered meshes now use resistance-weighted face conductance
    and face electromotive terms, rather than equal-spacing harmonic shortcuts
  - retained numerical effect at the current Hunt controls:
    - Hunt `Ha20`, `32^2` improved slightly:
      combined error `≈ 0.12714 -> 0.12666`
    - Hunt `Ha100`, `32^2` regressed slightly at the current default-like point:
      combined error `≈ 0.34266 -> 0.34927`
    - Hartmann `Ha20`, `32^2` remained accepted on the improved single-region
      CG path
  - retained interpretation:
    - this is a principled finite-volume correction on nonuniform meshes, so it
      is worth keeping
    - but it does not by itself solve the remaining higher-Ha Hunt gap
    - the Hunt control surface still needs a better coupled-update law on top of
      the improved discretization
- Another attempted solver-side change was explicitly rejected:
  - I tried reconstructing `J` and `J×B` from face-flux-consistent currents so
    the Lorentz update would use the same flux form as the potential solve
  - that improved Hartmann modestly, but it degraded Hunt badly:
    - Hunt `Ha20`, `32^2`: combined error rose to about `0.130`
    - Hunt `Ha100`, `32^2`: combined error rose to about `0.414`
  - that branch was rolled back
  - retained interpretation:
    - a more consistent flux form alone is not enough
    - the remaining Hunt issue is still in how the layered coupled update uses
      the current reconstruction, not just in whether that reconstruction is
      face-based or center-based
- The steady solver semantics are now slightly stricter and more honest for
  layered cases:
  - `solve_steady(...)` can optionally require both velocity residual and
    electric-potential residual convergence
  - this is exposed through `steady_potential_tolerance`
  - retained interpretation:
    - it does not change the current Hunt parity numbers by itself, because the
      current reference sweeps are already running to their full step budgets
    - but it prevents future layered runs from being classified as “steady” on
      velocity residual alone while the potential solve is still loose
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
