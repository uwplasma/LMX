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
- When mirroring recovered transient validation cases, keep the magnetic-field
  startup ramp law aligned with the backend controls. The current LMX ramp
  intentionally matches the recovered FreeMHD implementation:
  `(t - BtStartTime) / (BtDuration + 1e-6)`, clipped to `[0, 1]`.
- The current reconstruction used for `J` and `JxB` is now an explicit solver
  control:
  - `current_reconstruction="cell_centered"` uses the gradient-based
    cell-centered reconstruction
  - `current_reconstruction="face_averaged"` reconstructs cell currents by
    averaging the finite-volume face currents back onto cells
  Keep `cell_centered` as the default until a face-based path improves the real
  Hunt parity metrics, not just one traced quantity.
- Inlet- or flow-rate-driven reduced cases with `forcing = 0` now solve for the
  streamwise forcing required to hit the target mean velocity inside the
  implicit velocity update. That is the retained core semantics for reduced
  inlet-driven runs; avoid reintroducing a fixed case-specific startup source
  heuristic in the solver.
- When diagnosing layered Hunt transients, watch the interaction between
  `_limited_velocity_update(...)` and `outer_iterations`. The retained Hunt
  traces can become clamp-controlled, with `residual_history` effectively
  tracking `outer_iterations * velocity_update_limit`. That is solver-control
  behavior, not physics, and it should be treated as the next stabilization
  target rather than papered over with more case-specific parameter tuning.
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
  remaining Hunt gap and for tracking the older Jacobi-only Hartmann branch that
  motivated the current backend work. CI still runs a dedicated Hartmann
  `potential_iterations` sweep at `Ha20`, `32^2` for that reason. The same
  runner can now sweep `potential_tolerance`, `potential_relaxation`,
  `potential_solver`, `current_reconstruction`, and `velocity_update_limit`,
  which is useful when separating an insufficient
  `phi`-solve iteration ceiling from an early residual stop, a brittle raw
  Jacobi update, a backend choice, or an overly aggressive coupled velocity
  update.
- `python scripts/run_solver_grid_sweep.py --output artifacts/control_grid`:
  writes two-parameter control grids when one-parameter sweeps are not enough to
  distinguish a real improvement from a cross-case tradeoff. The current Hunt
  work uses this to compare `outer_iterations` against `potential_relaxation`
  directly instead of inferring interactions from separate runs.
- `python scripts/summarize_ci_artifacts.py` now also accepts
  `--control-grid-summary`, so grid sweeps can show up in the same CI markdown
  summary as validation, benchmarks, parity, and one-parameter sweeps.
- The electric-potential solve now exposes three explicit backends:
  - `jacobi`: weighted Jacobi with optional residual-based stopping
  - `cg`: matrix-free preconditioned conjugate gradient
  - `cg_volume`: CG on the same layered discrete system after left-scaling by
    the cell metric, which is the symmetric form of the nonuniform divergence
    operator
  - `lineax_cg`: optional external CG path
  The default `auto` policy resolves outside the traced JAX step:
  single-region duct solves use `cg`, while layered multi-region solves use
  `cg_volume`. That policy is based on region structure, not case-name
  heuristics.
- `python scripts/run_hunt_solver_diagnostic_report.py --freemhd-run-dir ...`:
  writes a solver-diagnostic-first Hunt comparison report. The JSON artifact has
  three top-level sections:
  - `freemhd_run`: run-directory inspection counts and latest `mag(U)` metadata
  - `lmx_solver`: the native case controls, including inferred magnetic-ramp
    settings from `BtStartTime` / `BtDuration` when they are present in the
    recovered case and the recovered inlet-driven startup boundary when the
    case has a nonzero initial velocity, plus `validation_summary(...)`
    metrics such as `residual`, `potential_residual`, and
    `potential_iterations_used`
  - `comparison`: `u_max` and sampled-profile comparisons against the recovered
    FreeMHD run when sample files are present
- Reduced mean-flow drive semantics are now stricter:
  - `inlet_flow_rate` activates the internal target-mean-velocity closure when
    `forcing = 0`
  - `inlet_velocity` does not. It is kept for recovered-case metadata and
    startup-state parity, because treating it as a global reduced mean target
    makes the Hunt replay too aggressive.
- `python scripts/patch_freemhd_coupled_logging.py --root ./external/FreeMHD`:
  patches the local `epotMultiRegionInterFoam` sources with opt-in `LMX_DIAG`
  logging in the outer loop, fluid `epot` solve, momentum predictor, and
  interFoam pressure-correction block. The patched `epot` log now includes:
  - `maxJ`
  - `maxJn`
  - `maxJnDensity`
  - `maxPsiub`
  - `maxPsiubDensity`
  - `maxCenteredJxB`
  - the active `maxJxB`
  so recovered Hunt runs can distinguish conservative-force effects from later
  momentum/pressure response effects. `maxJn` / `maxPsiub` are face-flux-style
  quantities in FreeMHD, while `maxJnDensity` / `maxPsiubDensity` divide back
  through `mesh.magSf()` so the log also exposes face-density-style quantities.
- `python scripts/compare_hunt_trace_histories.py --freemhd-diag-json ...`:
  now compares:
  - `u_max`
  - cell-centered `current_max`
  - face-current `face_current_max` when `maxJn` is available
  - face-current `face_current_density_max` when `maxJnDensity` is available
  - source-term `emf_max` when `maxPsiub` is available
  - source-term `emf_density_max` when `maxPsiubDensity` is available
  - `lorentz_max`
  and reports both normalized history error and raw relative error, which is
  necessary when only a partial live FreeMHD log is available. Treat the raw
  `maxJn` / `maxPsiub` comparisons cautiously unless the log semantics are
  known to match the LMX diagnostic being compared. The JSON payload now also
  carries `freemhd_pressure_final_records` and `freemhd_epot_records`, so later
  pressure-response tuning can use the extracted FreeMHD correction history
  directly instead of reopening the raw solver log.
- `python scripts/extract_freemhd_coupled_log.py log.txt --output diag.json`:
  extracts those `LMX_DIAG` lines into structured JSON for comparison with LMX
  solver diagnostics.
- `python scripts/build_freemhd_container.py --local-freemhd-root ./external/FreeMHD`:
  stages a minimal local FreeMHD tree into a temporary Docker build context so
  container runs can use patched local solver sources instead of recloning
  upstream. The build path now uses `docker buildx build --load` so the image is
  immediately visible to later `docker run` steps on the local Docker daemon.
- `python scripts/run_freemhd_case.py --local-freemhd-root ./external/FreeMHD`:
  can now auto-build a missing local image before launching the case, and
  `--patch-local-freemhd-logging` applies the current diagnostic patch set to
  that checkout before the build. Use that path for iterative Hunt trace work on
  this machine so the patched image does not have to be managed manually.
- The electric-potential discretization on nonuniform meshes now uses
  resistance-weighted face conductance and face electromotive terms instead of
  equal-spacing harmonic shortcuts. That is the finite-volume-consistent form
  when adjacent cells have different widths, and it is especially relevant for
  clustered layered Hunt meshes.
- Validation summaries now also include simple profile-pathology diagnostics such
  as sign-change counts and negative-value fractions on the extracted duct
  midplane profiles. These are useful when a solver branch becomes oscillatory
  even before its aggregate L2 error is inspected.
- Validation summaries now also include a normalized electric-potential equation
  residual from the latest steady/transient step. Use that metric alongside
  profile errors when diagnosing whether a branch is failing because the
  electric-potential solve itself is under-resolved or because the larger
  coupled MHD update is unstable.
- Validation summaries also report the actual electric-potential iteration count
  used by the latest solve. Use it with `potential_residual` to tell the
  difference between “the solve stopped early” and “the solve hit its iteration
  ceiling without converging enough.”
- `solve_steady(...)` can now optionally require both velocity and
  electric-potential convergence before stopping. Use
  `steady_tolerance` together with `steady_potential_tolerance` when a layered
  case should not be treated as steady while the potential equation is still
  under-resolved.
- Closed-channel validation, convergence, and control-sweep artifacts now also
  report a combined profile error built from the `y` and `z` cuts. Use that when
  a candidate improves one direction while degrading the other, which is exactly
  the current Hunt tuning pattern.
- CI sweep summaries now track the best combined closed-channel error as well as
  the best directional `y/z` errors, so solver-control decisions do not depend on
  manually comparing separate profile columns after each run.
- `python scripts/run_benchmark_suite.py --output artifacts/benchmarks/benchmark.json`:
  writes the current benchmark report.
- `python scripts/run_hunt_solver_diagnostic_report.py --current-reconstruction face_averaged ...`:
  replays the recovered Hunt startup path with the alternate current
  reconstruction. Use that only as a targeted diagnosis tool unless it also
  wins the corresponding parity metrics.
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
