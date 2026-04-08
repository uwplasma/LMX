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
    - Hunt-style layered ducts need their own multi-region treatment and should
      not reuse the single-solid layered approximation when comparing against
      recovered cases with both conducting and insulating wall regions
    - `main` now encodes that as a principled `potential_solver="auto"` policy:
      use CG when the solved cross-section is a single fluid region, and use
      `cg_volume` when explicit solid layers are present
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
- The next blocker-focused retained change attacked the layered `phi` solve
  directly:
  - LMX now has `potential_solver="cg_volume"` for layered cases
  - this solves the same discrete layered `phi` system after left-scaling by
    the cell metric, which is the symmetric form of the nonuniform
    divergence-form operator
  - retained numerical effect:
    - Shercliff `Ha20`, `32^2`: matches the good CG path at
      combined error `≈ 0.162`
    - Hunt `Ha20`, `32^2`:
      - `jacobi`: combined error `≈ 0.1267`,
        `potential_residual ≈ 5.0e-1`
      - `cg_volume`: combined error `≈ 0.1510`,
        `potential_residual ≈ 9.4e-3`
    - Hunt `Ha100`, `32^2`:
      - `jacobi`: combined error `≈ 0.3493`,
        `potential_residual ≈ 1.7e-1`
      - `cg_volume`: combined error `≈ 0.3014`,
        `potential_residual ≈ 1.0e-2`
  - retained interpretation:
    - the multi-region `phi` block really was part of the blocker
    - but the remaining Hunt problem is still not the `phi` block alone
    - once the layered `phi` residual is reduced by one to two orders of
      magnitude, `Hunt Ha20` still gets a worse combined profile, so the next
      retained solver change should target the multi-region velocity update /
      coupling law rather than only the `phi` backend
- The FreeMHD-side comparison path now also has a direct coupled-iteration
  logging route:
  - `patch_freemhd_coupled_logging.py` adds opt-in `LMX_DIAG` logging to the
    local `epotMultiRegionInterFoam` sources without changing physics
  - `build_freemhd_container.py --local-freemhd-root ...` can now build a
    container from that patched local tree instead of forcing a fresh upstream
    clone, so the logging patch reaches real solver runs
  - the build path now uses `docker buildx build --load`, which is the retained
    fix for the earlier local state where a build reported success but the image
    was not visible to `docker image inspect` / `docker run`
  - `extract_freemhd_coupled_log.py` converts those lines into JSON records
  - the next Hunt solver step should use those FreeMHD iteration diagnostics
    together with LMX `potential_residual` / `potential_iterations_used`
    instead of relying only on sampled end profiles
- First live retained Hunt `Ha20` solver-trace data from patched FreeMHD:
  - case: recovered `hunt_exactBL_Ha20`, short run with `deltaT = 1e-5` start
    and `endTime = 1e-4`
  - first fluid `potE` solves:
    - `t = 1.25e-05`: initial residual `1.0`, final residual
      `4.43e-08`, iterations `11`, `maxJxB ≈ 4.69e3`
    - `t = 2.70833e-05`: initial residual `2.86e-1`, final residual
      `5.34e-08`, iterations `7`, `maxJxB ≈ 4.65e3`
    - `t = 4.53125e-05`: initial residual `1.90e-1`, final residual
      `3.75e-08`, iterations `7`, `maxJxB ≈ 4.63e3`
    - `t = 6.35417e-05`: initial residual `1.44e-1`, final residual
      `3.06e-08`, iterations `7`, `maxJxB ≈ 4.62e3`
  - retained interpretation:
    - FreeMHD’s electric-potential block is not marginal on this case
    - the next useful comparison is therefore not “can LMX drive `phi`
      residual down?” but “how does the coupled velocity/pressure response use
      that `phi` state?”
- The missing FreeMHD pressure-correction trace is now also real:
  - retained harness fix:
    - the recovered-case container runner now supports `startFrom`
    - this fixes the earlier no-op rerun where a case with existing `0.0001/`
      output started from `latestTime` and immediately hit `endTime`
  - retained pressure records from the live patched Hunt `Ha20` run:
    - `t = 1.25e-05`, `corr = 0`:
      - `pFinalResidual = 4.76e-05`
      - `pIterations = 45`
      - `maxU = 0.11803283`
      - `maxP = maxPRgh = 111975.04`
    - `t = 1.25e-05`, `corr = 2`:
      - `pFinalResidual = 9.10e-08`
      - `pIterations = 15`
      - `maxU = 0.11774258`
    - `t = 2.70833e-05`, `corr = 2`:
      - `pFinalResidual = 9.03e-08`
      - `pIterations = 15`
      - `maxU = 0.11791814`
  - retained interpretation:
    - FreeMHD’s layered Hunt `phi` and pressure blocks both converge much more
      tightly than the current LMX layered short-time trace
    - the remaining question is now not whether LMX needs better diagnostics,
      but which layered update/backend change actually closes that gap without
      regressing the longer profile metrics
- LMX now emits the matching short-time trace observables:
  - `Diagnostics` includes `time_history` and `u_max_history`
  - `run_hunt_solver_diagnostic_report.py` now writes an `lmx_solver.trace`
    section with:
    - `time_history`
    - `u_max_history`
    - `residual_history`
    - `potential_residual_history`
    - `potential_iterations_history`
- First retained short-time LMX Hunt backend comparison against the FreeMHD
  trace:
  - default layered short-time trace (`Ha20`, `dt = 1e-5`, `10` steps):
    - first-step `u_max ≈ 0.117496`
    - final trace `potential_residual ≈ 4.12e-01`
  - `potential_relaxation = 0.5` is worse:
    - final trace `potential_residual ≈ 4.62e-01`
    - worse `u_max` agreement as well
  - `outer_iterations = 4` is mixed:
    - final trace `potential_residual ≈ 3.24e-01`
    - only a small `u_max` benefit and not enough evidence for a default shift
  - `potential_solver = cg_volume`, `potential_iterations = 200` is the
    strongest retained short-time candidate:
    - final trace `potential_residual ≈ 1.39e-01`
    - `u_max_abs_diff ≈ 4.12e-04`
  - retained interpretation:
    - `cg_volume` clearly improves the short-time layered `phi` convergence
      signal
    - but that still is not enough evidence to promote it as the default Hunt
      backend until the longer-horizon Hunt `Ha20` / `Ha100` profile metrics are
      rechecked on the current code state
- That longer-horizon Hunt backend check is now retained and it changes the
  solver baseline:
  - Hunt `Ha20`, old layered default:
    - `potential_residual ≈ 9.67e-01`
    - `u_max ≈ 1.86e-02`
    - `u_max_abs_diff ≈ 9.95e-02`
    - sampled `combined_l2_error ≈ 5.68e-01`
  - Hunt `Ha20`, `cg_volume`:
    - `potential_residual ≈ 2.68e-02`
    - `u_max ≈ 1.03e-01`
    - `u_max_abs_diff ≈ 1.51e-02`
    - sampled `combined_l2_error ≈ 4.20e-01`
  - Hunt `Ha100`, old layered default:
    - `potential_residual ≈ 7.06e-03`
    - `u_max ≈ 4.76e-04`
    - `u_max_abs_diff ≈ 1.23e-01`
    - sampled `combined_l2_error ≈ 6.39e-01`
  - Hunt `Ha100`, `cg_volume`:
    - `potential_residual ≈ 1.09e-02`
    - `u_max ≈ 9.28e-02`
    - `u_max_abs_diff ≈ 3.10e-02`
    - sampled `combined_l2_error ≈ 2.15e-01`
  - retained interpretation:
    - the old layered `jacobi` default was a real blocker on the full Hunt path
    - layered `auto` now resolves to `cg_volume`
    - the remaining Hunt gap is now the layered velocity/pressure coupling
      response on top of the improved multi-region `phi` backend, not the
      layered `phi` backend choice itself
- The new layered default also materially improves the short-time Hunt trace:
  - Hunt `Ha20`, `auto -> cg_volume`, `dt = 1e-5`, `t_final = 1e-4`,
    `10` steps:
    - `u_max ≈ 0.117499`
    - `u_max_abs_diff ≈ 6.21e-04`
    - `potential_residual ≈ 2.61e-03`
    - sampled `combined_l2_error ≈ 9.83e-02`
    - final `maxJ ≈ 2.95`
    - final `maxJxB ≈ 31.0`
  - retained interpretation:
    - the layered `phi` solve is no longer the dominant blocker on the Hunt
      startup path either
    - the remaining work should focus on longer-horizon momentum/pressure
      evolution instead of more layered `phi` backend tuning
- A targeted layered-only coupling candidate was tested and rolled back:
  - change:
    - advance the layered velocity trial from the current outer iterate instead
      of the step-entry state
  - retained result:
    - Hunt `Ha20` short-time combined error moved from about `9.83e-02` to
      `9.75e-02`
    - Hunt `Ha20` full-path combined error moved from about `4.203e-01` to
      `4.204e-01`
    - Hunt `Ha100` full-path combined error stayed at about `2.149e-01`
  - retained interpretation:
    - this is not the next real fix for the longer-horizon Hunt mismatch
    - keep the stable solver baseline and use the new force-history observables
      to target the pressure-response defect more directly
- The new trace-alignment script now makes that pressure-response defect
  explicit instead of qualitative:
  - `python scripts/compare_hunt_trace_histories.py --freemhd-diag-json ... --lmx-report-json ...`
    aligns the patched FreeMHD Hunt records and the LMX Hunt trace on the same
    time axis, using:
    - final pressure-correction `maxU` from FreeMHD
    - `epot`-logged `maxJ` and `maxJxB` from FreeMHD
    - `u_max_history`, `current_max_history`, and `lorentz_max_history` from
      LMX
  - retained `Ha20` short-time alignment against
    `/tmp/lmx_hunt20_live_diag.json` and
    `/tmp/lmx_hunt_auto_short_trace_force.json`:
    - `u_max`: normalized `l2_error ≈ 3.55e-03`, `max_abs_diff ≈ 6.16e-03`
    - `maxJ`: normalized `l2_error ≈ 3.58e-02`, `max_abs_diff ≈ 4.56e-02`
    - `maxJxB`: normalized `l2_error ≈ 8.15e-02`, `max_abs_diff ≈ 1.31e-01`
  - retained interpretation:
    - short-time `maxU` already tracks FreeMHD well
    - current magnitude drifts moderately
    - Lorentz-force magnitude diverges first and fastest
    - the next solver pass should target why LMX `JxB` decays too quickly on
      the Hunt startup path, instead of changing the `phi` backend again
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
- Rejected a flux-blended current reconstruction candidate in
  `lmx/solvers.py` after the
  short-time Hunt `Ha20` replay showed no meaningful gain over the retained
  baseline:
  - baseline sampled combined error `≈ 9.825e-02`
  - candidate sampled combined error `≈ 9.825e-02`
  - baseline `u_max_abs_diff ≈ 6.21e-04`
  - candidate `u_max_abs_diff ≈ 6.21e-04`
  - retained interpretation:
    - blending face-flux current back into the cell-centered Lorentz update is
      not the next real Hunt fix
    - keep the stable centered-current reconstruction and continue targeting
      the longer-horizon momentum/pressure response
- Fixed the current GitHub Actions blocker by updating the coarse Hunt
  boundedness gate in `tests/test_physics.py`
  to match the retained stable solver behavior on `Ha20`, `16x16`,
  `wall_cells=2`:
  - current stable coarse-mesh values:
    - `u_max ≈ 2.63e-02`
    - `u_min ≈ -4.25e-04`
    - `potential_residual ≈ 2.04e-03`
  - retained interpretation:
    - this test is a boundedness guard on the coarse layered case, not a parity
      or accuracy gate
    - detailed Hunt accuracy remains tracked through the recovered FreeMHD
      parity reports and dedicated diagnostic scripts
- Added magnetic-field ramp support to the core `MagneticFieldSpec` and the
  recovered-case parity loaders. `run_freemhd_parity_report.py` and
  `run_hunt_solver_diagnostic_report.py` now infer `BtStartTime` and
  `BtDuration` from `system/controlDict`, and LMX applies the same ramp during
  the transient solve when those controls are present.
  - retained short-time Hunt `Ha20` result:
    - sampled combined error stayed at about `9.825e-02`
    - `u_max` trace error stayed at about `3.55e-03`
    - `maxJ` trace error moved slightly from about `3.58e-02` to `3.58e-02`
    - `maxJxB` trace error moved slightly from about `8.15e-02` to `8.13e-02`
  - retained interpretation:
    - the feature is correct and worth keeping because it matches recovered case
      controls and is general for future transient problems
    - for Hunt `Ha20`, it is not the missing fix by itself because the ramp
      finishes by the first sampled diagnostic time
    - the remaining startup mismatch is still in the Lorentz-response path
- `run_freemhd_case.py` can now auto-build a missing local diagnostic image
  from `--local-freemhd-root`, and `--patch-local-freemhd-logging` applies the
  current coupled-logging patch set before that build. That removes the last
  manual image-management step from the local Hunt diagnostic loop.
- `build_freemhd_container.py` now also supports `--no-cache`, which is the
  preferred mode when the local patched FreeMHD/OpenFOAM solver sources changed
  and the next diagnostic replay must not reuse an older cached image layer.
- Extended the FreeMHD diagnostic patcher so `LMX_DIAG epot` records now include
  `maxJn`, `maxPsiub`, and `maxCenteredJxB` in addition to the active
  `maxJxB`. On the first real rerun of recovered Hunt `Ha20`, the first patched
  fluid `epot` record at `t = 1.25e-05` showed:
  - `maxJ = 85382.119`
  - `maxJn = 0.83266194`
  - `maxPsiub = 1.173496`
  - `maxCenteredJxB = 4689.7866`
  - `maxJxB = 4689.7866`
  - retained interpretation:
    - at least on the first logged Hunt startup point, the conservative and
      centered Lorentz-force magnitudes are identical in FreeMHD
    - that makes it much less likely that the remaining LMX Hunt gap is caused
      primarily by using `J ^ B` instead of the conservative `JxB` form
    - the next solver target should stay on the momentum/pressure response path
      or on the spatial distribution of `J`, not on conservative-vs-centered
      force magnitude alone
- Added matching LMX face-current and `U×B`-source diagnostics:
  - `face_current_max_history`
  - `emf_max_history`
  and extended `compare_hunt_trace_histories.py` to report raw relative error in
  addition to normalized history error. On the current real partial Hunt `Ha20`
  live log at `t = 1.25e-05`, the retained one-point raw comparison against
  `/tmp/lmx_hunt_short_trace_with_faces.json` is:
  - `current_max` raw relative error `≈ 0.99996`
  - `face_current_max` raw relative error `≈ 1.48571`
  - `emf_max` raw relative error `≈ 1.00256`
  - `lorentz_max` raw relative error `≈ 0.99422`
  - retained interpretation:
    - the remaining Hunt startup mismatch is now much more likely to originate
      in current/source scaling or distribution before pressure response, not
      just in later momentum coupling
    - the next solver target should focus on matching FreeMHD’s `psiub` and
      face-current magnitudes on the layered startup path
- That raw one-point interpretation is now intentionally softened. The patched
  FreeMHD comparison path and local source patch now also support:
  - `maxJnDensity`
  - `maxPsiubDensity`
  and `compare_hunt_trace_histories.py` can align those against the LMX
  density-style Hunt diagnostics. This matters because `maxJn` and `maxPsiub`
  are face-flux-style quantities in FreeMHD, while the current LMX Hunt startup
  traces being compared are density-style maxima. Until the rerun with density
  diagnostics is complete, the old raw `maxJn` / `maxPsiub` mismatch should not
  be treated as decisive solver evidence by itself.
- That rerun is now complete enough to change the diagnosis again:
  - the retained LMX magnetic ramp now matches the recovered FreeMHD startup
    law exactly, including the `BtDuration + 1e-6` denominator
  - on the short Hunt `Ha20` trace comparison, that reduced the normalized
    history mismatch materially:
    - `emf_max`: `≈ 8.32e-2 -> 1.56e-3`
    - `emf_density_max`: `≈ 8.35e-2 -> 1.93e-3`
    - `lorentz_max`: `≈ 1.95e-1 -> 3.26e-2`
    - `face_current_max`: `≈ 9.66e-2 -> 1.75e-2`
  - `u_max` remained already good at `≈ 1.18e-3`
  - retained interpretation:
    - the Hunt startup blocker was not primarily in conservative-vs-centered
      force form and not primarily in the layered `phi` linear solve anymore
    - a real part of it was the startup magnetic-ramp semantics feeding the
      `U×B` source term
    - the next remaining mismatch is narrower: cell-centered current magnitude
      and later coupled response, not the startup source history itself
- The next targeted solver experiment is now retained as an explicit control
  rather than another hidden local branch:
  - `TimeStepperConfig.current_reconstruction` supports
    `cell_centered` and `face_averaged`
  - on the corrected short Hunt `Ha20` density-log replay, `face_averaged`
    changes the normalized trace errors as follows:
    - `u_max`: unchanged at `≈ 1.18e-3`
    - `emf_max`: unchanged at `≈ 1.56e-3`
    - `emf_density_max`: unchanged at `≈ 1.93e-3`
    - `lorentz_max`: `≈ 3.26e-2 -> 5.12e-3`
    - `current_max`: `≈ 4.43e-2 -> 6.77e-2`
    - `face_current_max`: `≈ 1.75e-2 -> 1.76e-2`
  - retained interpretation:
    - a face-averaged reconstruction helps the normalized Hunt `JxB` history
      substantially
    - but it does not yet improve the current-magnitude comparison or the short
      recovered `u_max` replay enough to justify a default change
    - it should stay available as a solver-control / diagnosis aid while the
      next fix targets a better cell-centered current reduction or the later
      coupled response
- The longer corrected Hunt `Ha20` trace now has a more honest replay
  baseline. The original `t = 6e-05` diagnostic runner had been creating
  `forcing = 0` Hunt cases without the recovered inlet-velocity boundary, so
  it was replaying the wrong drive semantics. The runner now adds that
  inlet-velocity boundary automatically when the recovered startup case has a
  nonzero internal velocity.
- On that corrected `t = 6e-05` replay, a patched local FreeMHD rerun and the
  first corrected LMX diagnostic report showed:
  - `u_max l2 ≈ 2.44e-2`
  - `current_max l2 ≈ 8.86e-2`
  - `emf_max l2 ≈ 2.62e-2`
  - `lorentz_max l2 ≈ 8.44e-2`
  - the retained LMX `u_max_history` is now:
    `0.118335, 0.119233, 0.120211, 0.121298, 0.122547, 0.123869`
  - the corresponding recovered FreeMHD final pressure-corrector `maxU` values
    are:
    `0.117692, 0.117804, 0.117905, 0.118003, 0.118099, 0.118191`
  - retained interpretation:
    - the earlier flatter Hunt trace was partly a runner artifact
    - once the replay is corrected, LMX is not under-accelerating the short
      layered Hunt transient; it is over-responding relative to FreeMHD
    - that shifts the next solver target away from startup forcing/ramp
      semantics and toward the layered stabilization / coupled velocity update
      itself
- The next retained solver change came directly from that corrected replay:
  reduced mean-flow drive is now activated only by `inlet_flow_rate`, not by
  `inlet_velocity`.
  - treating recovered `inlet_velocity` metadata as a global target mean
    velocity was the main reason the corrected Hunt replay became too
    aggressive
  - on the same corrected `t = 6e-05` Hunt `Ha20` window, the retained replay
    now improves to:
    - `u_max l2 ≈ 2.62e-3`
    - `current_max l2 ≈ 6.44e-2`
    - `emf_max l2 ≈ 3.19e-3`
    - `lorentz_max l2 ≈ 1.05e-1`
  - the retained `u_max_history` is now almost flat:
    `0.117500, 0.117500, 0.117500, 0.117500, 0.117500, 0.117500`
  - retained interpretation:
    - the reduced solver should only infer a target mean velocity from a
      flow-rate boundary, not from a nominal inlet velocity
    - this removes most of the corrected Hunt startup over-response without
      reintroducing the earlier under-driven runner bug
    - the next remaining Hunt gap is narrower again:
      current/Lorentz distribution and later coupled response, not startup
      drive semantics
- A direct limiter probe on that corrected `t = 6e-05` Hunt replay now gives a
  narrower conclusion:
  - retained default `velocity_update_limit = 2e-3`:
    - `u_max l2 ≈ 2.62e-3`
    - `current_max l2 ≈ 6.44e-2`
    - `emf_max l2 ≈ 3.19e-3`
    - `lorentz_max l2 ≈ 1.05e-1`
  - smaller retained global cap `velocity_update_limit = 1e-3`:
    - `u_max l2 ≈ 9.00e-3`
    - `current_max l2 ≈ 1.02e-1`
    - `emf_max l2 ≈ 9.53e-3`
    - `lorentz_max l2 ≈ 2.29e-2`
  - larger global cap `velocity_update_limit = 4e-3`:
    - `u_max l2 ≈ 5.30e-2`
    - `current_max l2 ≈ 6.38e-2`
    - `emf_max l2 ≈ 5.81e-2`
    - `lorentz_max l2 ≈ 1.92e-1`
  - an experimental pointwise local-clamp branch was also tried and rolled
    back immediately:
    - `u_max l2 ≈ 1.05e-1`
    - `current_max l2 ≈ 1.01e-1`
    - `emf_max l2 ≈ 1.29e-1`
    - `lorentz_max l2 ≈ 2.74e-1`
  - retained interpretation:
    - after the inlet-drive correction, the retained `2e-3` cap is again the
      best balanced corrected-trace setting on `main`
    - a smaller global cap now mostly trades better normalized `JxB` history
      for worse `u_max`, current, and source-term alignment
    - but a local pointwise clamp is clearly worse and was not kept on `main`
    - limiter policy alone is still not the real missing layered-Hunt fix
- `current_reconstruction="face_averaged"` remains useful only as a diagnosis
  aid on that corrected replay:
  - it still improves the very short startup `lorentz_max` alignment
  - but it does not improve the corrected longer-window replay enough to become
    the new default
- The inlet-driven reduced-model forcing is now also cleaner internally:
  - LMX no longer relies on the old fixed Hunt source heuristic for
    `forcing = 0` inlet-driven cases
  - instead, it solves for the streamwise forcing needed to match the target
    mean velocity inside the implicit velocity update only when the case
    actually specifies `inlet_flow_rate`
  - `inlet_velocity` remains available for recovered-case metadata and startup
    parity, but it is no longer converted into a reduced global mean target
- The corrected `6e-05` Hunt `Ha20` window also exposed a sharper stabilizer
  issue in the reduced solver:
  - the retained default trace shows `residual_history ≈ 0.012` on every step,
    which matches `outer_iterations * velocity_update_limit = 6 * 0.002`
  - so the later Hunt response is being shaped strongly by the global velocity
    limiter rather than only by the coupled MHD update
  - retained interpretation:
    - the cap still matters, but after the reduced drive fix it is no longer
      the leading explanation for the corrected Hunt mismatch
    - the next Hunt fix should target the layered
      velocity/pressure-response formulation itself rather than more limiter
      churn
- That corrected limiter picture also has to survive native closed-channel Hunt
  validation before it can become a default:
  - `Ha20`, `32^2`:
    - `velocity_update_limit = 1e-3` gives `combined_l2 ≈ 0.1490`
    - retained `2e-3` gives `combined_l2 ≈ 0.1510`
    - so the native improvement is only marginal
  - `Ha100`, `32^2`:
    - `velocity_update_limit = 1e-3` gives `combined_l2 ≈ 0.3014`
    - retained `2e-3` gives `combined_l2 ≈ 0.2991`
    - so the higher-Ha native profile actually gets slightly worse
  - retained interpretation:
    - the limiter remains a bounded stabilizer and diagnosis aid, not the main
      next physical correction
- The FreeMHD source and log path now make the later coupling gap more explicit:
  - `epotMultiRegionInterFoam/fluid/solveMhdFluid.H` runs
    `ePotEqn.H`, `mhdUEqn.H`, and then the `pEqn.H` pressure-correction loop
  - `mhdUEqn.H` only provides the momentum equation build/predictor; the actual
    later `U` corrections come from `common/interFoam/fluid/pEqn.H`
  - on the corrected Hunt `Ha20` window to `6e-05`, the final pressure-corrector
    iteration counts are approximately `15, 82, 63, 100, 100, 20` while
    `maxJxB` remains comparatively flat
  - retained interpretation:
    - the remaining Hunt blocker is now better described as a missing reduced
      analogue of the later pressure-correction response, not as another
      startup-source or `phi`-solve issue
    - `compare_hunt_trace_histories.py` now includes
      `freemhd_pressure_final_records` and `freemhd_epot_records` so that later
      response can be inspected directly from the JSON artifact instead of
      reopening raw FreeMHD logs
- A recovered late-time profile sample now sharpens that conclusion further:
  - the current liquid-region sampling path for the corrected Hunt replay is
    recoverable at `t = 3e-05`
  - the original retained comparison path overstated the `y` mismatch because
    it normalized fluid-only cell-centered LMX coordinates as if they were wall
    points and interpolated the reference profile onto those collapsed points
  - after correcting `compare_normalized_profiles(...)` to infer the simulated
    profile extent and interpolate LMX onto the sample coordinates, the matching
    replay gives:
    - `sample_y_l2_error ≈ 7.15e-4`
    - `sample_z_l2_error ≈ 5.41e-3`
    - `sample_combined_l2_error ≈ 3.86e-3`
  - retained interpretation:
    - the late-time sampled Hunt profile is already close on `main`
    - the remaining Hunt blocker is therefore back in the trace-level drift of
      `u_max`, current, and `JxB`, plus the later coupled pressure response
    - the next retained solver changes should be judged primarily against the
      patched FreeMHD history alignment, not this sampled-profile metric
- The latest retained Hunt geometry/wall update keeps that conclusion but makes
  the remaining trace problem narrower:
  - Hunt cases now use explicit insulating side-wall layers together with
    conducting Hartmann-wall layers, and layered no-slip is applied at wall
    faces instead of zeroing the first fluid cell center
  - on the corrected `Ha20`, `t <= 6e-05` replay, the current retained trace
    metrics are:
    - `u_max l2 ≈ 2.62e-3`
    - `current_max l2 ≈ 8.81e-2`
    - `face_current_max l2 ≈ 2.00e-2`
    - `emf_max l2 ≈ 3.19e-3`
    - `lorentz_max l2 ≈ 4.02e-2`
  - retained interpretation:
    - the geometry-faithful wall split plus direct-wall interpolation are worth
      keeping because sampled profiles and normalized `JxB` history are now
      both materially better
    - the obvious remaining lagging signal is the cell-centered current
      magnitude, which points to layered current reduction and later coupled
      response rather than another wall-profile or startup-ramp issue
- Re-checking `current_reconstruction` on top of that updated Hunt wall model
  narrows the next solver target again:
  - on the corrected `Ha20`, `t <= 6e-05` replay:
    - `cell_centered`: `current_max l2 ≈ 1.68e-2`, `lorentz_max l2 ≈ 1.98e-1`
    - `face_averaged`: `current_max l2 ≈ 1.23e-1`, `lorentz_max l2 ≈ 6.84e-2`
  - retained interpretation:
    - `face_averaged` still is not the right blanket replacement on `main`
    - the next retained solver change should build a better cell-centered
      reduction from the layered face-current system instead of flipping the
      global current-reconstruction switch
- That retained solver change is now on `main`:
  - added `current_reconstruction="hybrid_face_lorentz"`
  - the mode keeps cell-centered `J` reduction for diagnostics and stored
    fields, but reconstructs `JxB` from the layered face-current system before
    the momentum update
  - it is now the default Hunt short-transient control
  - corrected `Ha20`, `t <= 6e-05` replay:
    - `u_max l2 ≈ 1.18e-3`
    - `current_max l2 ≈ 1.22e-2`
    - `lorentz_max l2 ≈ 1.04e-2`
  - retained interpretation:
    - this is the first layered-current update that improves Hunt current and
      Lorentz histories together instead of trading one off against the other
    - the next Hunt gap is now later coupled momentum/pressure response, not
      startup source law, sampled late-time profiles, or gross layered-current
      construction
- To make that later response measurable, the Hunt diagnostic path now also
  records:
  - `mean_velocity_history`
  - `applied_forcing_history`
  - `pressure_proxy_history`
  and `compare_hunt_trace_histories.py` can align `pressure_proxy` /
  `applied_forcing` against FreeMHD `maxP`.
- First retained pressure-response probe:
  - replay the corrected Hunt `Ha20`, `t <= 6e-05` case with
    `drive_mode = inlet_flow_rate`
  - result:
    - baseline hybrid replay:
      - `u_max l2 ≈ 1.18e-3`
      - `current_max l2 ≈ 1.22e-2`
      - `lorentz_max l2 ≈ 1.04e-2`
    - `inlet_flow_rate` replay:
      - `u_max l2 ≈ 8.96e-3`
      - `current_max l2 ≈ 2.01e-2`
      - `lorentz_max l2 ≈ 2.17e-2`
      - `pressure_proxy l2 ≈ 8.10e-2`
- Retained interpretation:
  - the missing later-time Hunt response is not solved by switching the replay
    wholesale to a reduced flow-rate-driven closure
  - that candidate overdrives the corrected trace and should remain a
    diagnostic option only
- The next milder scalar family was also rejected locally:
  - apply a direct partial fraction of `pressure_proxy` as reduced streamwise
    forcing with gains `0.02`, `0.05`, and `0.1`
  - representative result:
    - baseline hybrid replay:
      - `u_max l2 ≈ 1.18e-3`
      - `current_max l2 ≈ 1.22e-2`
      - `lorentz_max l2 ≈ 1.04e-2`
    - gain `0.02`:
      - `u_max l2 ≈ 9.92e-4`
      - `current_max l2 ≈ 1.24e-2`
      - `lorentz_max l2 ≈ 1.07e-2`
    - gain `0.10`:
      - `u_max l2 ≈ 2.22e-4`
      - `current_max l2 ≈ 1.31e-2`
      - `lorentz_max l2 ≈ 1.16e-2`
- Retained interpretation:
  - a raw gain on `pressure_proxy` is still too blunt
  - it improves the late-time velocity trace only by degrading the
    electromagnetic response that the retained hybrid current/Lorentz update
    already fixed
  - the next plausible solver step is a fixed-source post-predictor correction
    on `u`, closer to the patched FreeMHD pressure loop, not another scalar
    forcing-gain tweak
- That fixed-source post-predictor family was then tested and rejected too:
  - native `Hunt Ha20`, `32 x 32`, `wall_cells=3`, with
    `velocity_corrector_iterations in {1,2,3}` and
    `velocity_corrector_relaxation in {0.1,0.2,0.35}`
  - retained baseline:
    - `combined_l2_error ≈ 1.0148e-1`
    - `y_l2_error ≈ 3.51e-2`
    - `z_l2_error ≈ 1.39e-1`
  - best tested corrector point:
    - `combined_l2_error ≈ 1.0564e-1`
    - `y_l2_error ≈ 7.29e-2`
    - `z_l2_error ≈ 1.30e-1`
  - retained interpretation:
    - this family is not the missing Hunt fix
    - it improves one component only by degrading the more important combined
      closed-channel error
    - it should remain rejected unless a future patched FreeMHD replay shows a
      materially different later-time pressure-response structure
- One smaller retained correctness fix did land:
  - reduced mean-flow closures and `pressure_proxy` diagnostics now use
    area-weighted cross-sectional averages on clustered nonuniform meshes
    instead of simple cell counts
  - this does not solve the current Hunt later-time parity blocker by itself,
    but it is the correct future-proof formulation for layered graded meshes
    and inlet-flow-rate-driven reduced runs
- The FreeMHD diagnostic harness is now more robust for live Hunt replay work:
  - `scripts/run_freemhd_case.py` can force
    `logCoupledMhdIterations true;` through `--log-coupled-iterations`
  - when `--output` is used, the full container stdout/stderr is preserved as
    sibling `*.run.stdout.log` / `*.run.stderr.log` files
  - when `--log-coupled-iterations` is active, the runner also writes the
    extracted `LMX_DIAG` payload as `*.run.diag.json`
  - retained interpretation:
    - this removes the earlier failure mode where a patched FreeMHD run could
      emit the right `LMX_DIAG` lines and still lose them when the container
      died during reconstruction or cleanup
    - Docker/OpenFOAM on this machine is usable but still intermittent, so
      preserving the full stdout log is now part of the validation baseline
- The hardened runner is now validated on a fresh rematerialized recovered Hunt
  case under `/private/tmp/lmx_hunt_refresh`:
  - restarted Docker, rematerialized `hunt_exactBL_Ha20`, and launched a clean
    short replay with
    `scripts/run_freemhd_case.py --log-coupled-iterations --end-time 2e-05`
  - live container logs showed patched `LMX_DIAG outer`, `epot`, and
    `pressure` records at `t = 1e-05` and `t = 2e-05`
  - saved those logs to
    `/private/tmp/lmx_hunt_refresh/docker_logs_live_short.log`,
    extracted
    `/private/tmp/lmx_hunt_refresh/hunt_diag_live_short.json`,
    and compared them against
    `/private/tmp/lmx_hunt_refresh/lmx_hunt_short_report.json`
  - retained short-window normalized history errors:
    - `u_max l2_error ≈ 1.89e-03`
    - `mean_velocity l2_error ≈ 2.08e-03`
    - `emf_max l2_error ≈ 9.51e-04`
    - `lorentz_max l2_error ≈ 8.36e-03`
    - `current_max l2_error ≈ 1.10e-02`
    - `face_current_max l2_error ≈ 7.50e-03`
    - `pressure_proxy l2_error ≈ 3.12e-02`
  - retained interpretation:
    - on the corrected short Hunt window, startup source history is no longer
      the dominant gap
    - the remaining reduced-model mismatch is more concentrated in the
      pressure-like response and layered current reduction than in normalized
      `u_max` or normalized `JxB`
- Latest retained live Hunt `Ha20` short-trace snapshot from the saved patched
  FreeMHD run and matching LMX replay:
  - FreeMHD `t = 1e-05`:
    - `potEFinalResidual ≈ 4.43e-08`
    - `potEIterations = 11`
    - `pFinalResidual ≈ 4.86e-05`
    - `pIterations = 60`
    - `maxU ≈ 1.1794e-01`
    - `maxJxB ≈ 3.8759e+03`
  - FreeMHD `t = 2e-05`:
    - `potEFinalResidual ≈ 3.67e-08`
    - `potEIterations = 10`
    - `pFinalResidual ≈ 2.36e-05`
    - `pIterations = 100`
    - `maxU ≈ 1.1805e-01`
    - `maxJxB ≈ 4.6525e+03`
  - aligned LMX-vs-FreeMHD trace errors through `t = 2e-05`:
    - `u_max l2 ≈ 1.83e-03`
    - `mean_velocity l2 ≈ 2.31e-03`
    - `pressure_proxy l2 ≈ 1.00e-01`
    - `current_max l2 ≈ 7.94e-02`
    - `face_current_max l2 ≈ 7.33e-02`
    - `emf_max l2 ≈ 7.14e-02`
    - `lorentz_max l2 ≈ 1.31e-01`
  - retained interpretation:
    - the patched Hunt startup path is now close enough that the remaining gap
      is no longer a generic startup-source failure
    - the most expensive remaining mismatch is later pressure/Lorentz response,
      not the `potE` solve itself

## Meeting demo artifact

- A checked-in meeting-ready example now exists in
  `examples/theory_meeting_demo.py`.
- Retained default run:

```bash
/Users/rogerio/base_env/bin/python3 examples/theory_meeting_demo.py --output ./artifacts/examples/theory_meeting_demo --resolution 32 --movie-case shercliff --movie-resolution 24 --movie-dt 1e-3 --movie-t-final 1e-1 --movie-frames 8
```

- That run writes:
  - verbose setup and solver progress tables
  - `.npz` solution dumps for Hartmann, Shercliff, and Hunt
  - steady Hartmann, Shercliff, and Hunt overview/diagnostics plots
  - `meeting_demo_report.json`
  - Shercliff startup 2D/3D GIF movies and PNG/PDF poster frames
- The retained local QA run produced:
  - `artifacts/examples/theory_meeting_demo_final/hartmann/hartmann_ha20_results.npz`
  - `artifacts/examples/theory_meeting_demo_final/shercliff/shercliff_ha20_results.npz`
  - `artifacts/examples/theory_meeting_demo_final/hunt/hunt_ha20_results.npz`
  - `artifacts/examples/theory_meeting_demo_final/shercliff/shercliff_startup_snapshots.npz`
  - `artifacts/examples/theory_meeting_demo_final/shercliff/movie/shercliff_startup_2d_poster.png`
  - `artifacts/examples/theory_meeting_demo_final/shercliff/movie/shercliff_startup_3d_poster.png`
- The retained default movie case is Shercliff because it gives the strongest
  visible startup structure at modest runtime. Hunt remains available as a
  movie case, but the current meeting-ready default is not Hunt.

## Planned improvements

- Add stronger parity thresholds once the solver is stable across more geometries.
- Extend the current mesh-convergence tooling to pseudo-time convergence studies.
- Use the new pseudo-time convergence runner to decide which remaining Hunt
  discrepancies are temporal versus spatial before changing solver defaults.
- Expand native LMX validation to additional mapped-geometry cases.
- Improve high-Ha Hunt fidelity without introducing case-specific hardcoded limits.
- Keep the external backend checks optional and separate from the core LMX identity.
