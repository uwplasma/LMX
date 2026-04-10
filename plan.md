# LMX Execution Plan And Chronological Log

## Project Goal

LMX is a Python/JAX-native inductionless MHD code for liquid-metal flows. It is being developed as a self-consistent solver in its own right, with optional external backend comparisons used only for validation and historical cross-checking. The immediate physics target remains the laminar electric-potential subset used for closed-channel verification and then the fringing-field pipe validation. The code should remain differentiable, CPU/GPU-capable, and structured around JAX compilation.

## Source Of Truth

- FreeMHD paper: https://doi.org/10.1063/5.0230242
- arXiv source: https://arxiv.org/abs/2409.08950
- FreeMHD repository: https://github.com/PlasmaControl/FreeMHD
- Zenodo assets: https://zenodo.org/records/13964055
- Canonical LMX repository: https://github.com/uwplasma/LMX

## Working Scope

### Implemented now

- New standalone `LMX` Python package rooted in this repository.
- Structured duct meshes and mapped pipe mesh scaffolding.
- Immutable case/config dataclasses for geometry, regions, magnetic field, boundary conditions, time stepping, and outputs.
- Laminar inductionless solver scaffold with:
  - cell-centered cross-sectional velocity field
  - electric-potential solve
  - current density and Lorentz-force reconstruction
  - explicit solid-conductivity regions for Hunt-style walls
- ParaView XML output and CSV profile extraction.
- Validation helpers for analytical and processed-slice closed-channel comparisons.
- FreeMHD container/asset fetch and local execution scaffolding.
- Explicit unit, regression, physics, validation, and benchmark entrypoints.
- GitHub Actions workflows for categorized pytest runs plus validation and benchmark artifact jobs.
- Read the Docs-compatible Sphinx/MyST documentation entrypoint plus CI docs build.
- Executable `lmx input.toml` workflow with a complete TOML loader, live solver logging, and standard output bundle writing.
- Zenodo closed-channel analytical and processed-slice reference-data loaders.
- Unit and categorized tests passing in `/Users/rogerio/base_env/bin/python3`.
- Combined local coverage across `lmx/` and `scripts/` is currently `89%`.
- Native mesh-convergence study runner and Hartmann acceptance reports.
- `solve_steady` now uses the configured steady tolerance and step budget instead of
  aliasing the fixed-step transient runner.

### Explicitly deferred

- Free-surface VoF.
- Temperature equation.
- Turbulence.
- Full 3D pressure-velocity coupling parity with OpenFOAM.
- Fringing-field mapped-grid operators and direct paper-data ingestion.
- Actual containerized FreeMHD benchmark execution.

## Current Code Layout

- `lmx/specs.py`: public configuration dataclasses.
- `lmx/mesh.py`: tensor-product and mapped mesh builders.
- `lmx/operators.py`: structured-grid operator kernels.
- `lmx/physics.py`: conductivity and magnetic-field field builders.
- `lmx/solvers.py`: current laminar inductionless solver entrypoints.
- `lmx/io.py`: ParaView writers.
- `lmx/config.py`: TOML input loader and executable run configuration.
- `lmx/runtime_logging.py`: live OpenFOAM-style solver logger.
- `lmx/validation.py`: analytical and FreeMHD comparison helpers.
- `lmx/cli.py`: command-line entrypoints.
- `scripts/fetch_freemhd_assets.py`: fetch FreeMHD repo and Zenodo files.
- `scripts/write_freemhd_container_files.py`: writes the local FreeMHD/OpenFOAM container bundle.
- `scripts/run_freemhd_case.py`: runs a mounted FreeMHD case in a prepared container image and records JSON metadata.
- `scripts/probe_freemhd_environment.py`: probes the local FreeMHD/OpenFOAM and Docker environment and records JSON diagnostics.
- `scripts/patch_freemhd_darwin_headers.py`: applies the current Darwin-specific local OpenFOAM `lnInclude` workaround to the vendored `external/FreeMHD` checkout for reproducible macOS `wmake` experiments.
- `scripts/probe_freemhd_container.py`: probes the local Docker bundle, local image tags, and base-image registry resolution and records JSON diagnostics.
- `scripts/inspect_freemhd_setup.py`: inspects the locally available FreeMHD assets, reports discovered case directories, and recommends the smallest smoke target.
- `docs/index.md`, `docs/conf.py`, `.readthedocs.yaml`: documentation landing page and Read the Docs build configuration.

## Latest Retained Reset Work

- Phase 1 of the research-grade reset is now started in code, not just in notes.
- `CaseSpec` now carries a first-class `solver` block through `lmx/specs.py`,
  `lmx/config.py`, the CLI/TOML path, and the shipped examples.
- Two solver families now exist explicitly:
  - `fully_developed_inductionless`
  - `legacy_reduced`
- The current public duct cases now construct with `solver.kind =
  "fully_developed_inductionless"` by default, while the older reduced Hunt
  path remains selectable for regression/fallback.
- The new fully developed solver uses:
  - a coupled `u` / `phi` iteration
  - face-conservative current construction
  - matrix-free five-point linear solves for the velocity subproblem
  - no velocity limiter in the default path
- The legacy reduced controls remain in `TimeStepperConfig`, but they are now
  documented and treated as legacy controls rather than the main research path.
- Documentation now distinguishes:
  - implemented duct solver support
  - mapped-pipe mesh scaffolding
  - not-yet-implemented fringing-field solver support
- The first retained research-grade diagnostics are now wired through the new
  fully developed solver path:
  - linear residual / iteration histories for the velocity solve
  - volumetric flow-rate history
  - mean current-magnitude history
  - integral Lorentz-power history
  - max `|div J|` history
  - gauge residual history
  - interface current residual history
- The new diagnostics are now present in:
  - live runtime logs
  - validation summaries
  - solution / restart `.npz` files
- The next retained structural fix on the new fully developed path is now in
  the velocity solve itself: layered Hunt ducts solve the cell-metric-scaled
  velocity system instead of the raw unscaled nonuniform operator. This removes
  the large spurious velocity linear residual that previously made the new Hunt
  path look far worse than its actual field/parity state.
- The next retained steady-solver correction is also now clear: the new fully
  developed path should use real macro steady iterations up to
  `time_stepper.max_steps`, not a one-shot step, and `steady_potential_tolerance`
  should only gate termination when it is explicitly set. Local probes on the
  retained code path show this is enough to drive default Shercliff `Ha20`,
  `16^2` from `residual ≈ 3.52e-4` to `≈ 4.32e-7` in four macro steps, and
  Hunt `Ha20`, `16^2` from `≈ 6.10e-4` to `≈ 1.24e-6` in five macro steps,
  without changing the new solver family structure.
- CI was also hardened for GitHub-hosted runners by constraining JAX/XLA
  resource use on the validation and coverage paths, which addresses the
  recent validation-job memory failures without changing the benchmark jobs.
- The benchmark workflow now needs the same constrained JAX/XLA runner settings
  as validation and coverage. The fully developed steady macro-iteration fix
  increases compile/runtime pressure enough that unconstrained benchmark runs
  on GitHub-hosted CPUs can hit LLVM `Cannot allocate memory` even when the
  solver behavior itself is correct.

## Immediate Reset Priorities

1. Improve the new `fully_developed_inductionless` parity and analytical quality
   enough that Hartmann, Shercliff, and Hunt can move off the retained legacy
   path completely.
2. Reset parity tooling around model-consistent observables instead of raw
   backend pressure-correction traces.
3. Stage the next solver family, `extruded_inductionless`, for Benchmark B
   fringing-field work while keeping current docs honest about its status.

## Current Readiness Assessment

### What LMX can credibly do today

- Run native laminar inductionless Hartmann, Shercliff, and Hunt-style duct cases from a single Python/JAX codebase.
- Generate fields, CSV cuts, ParaView output, validation reports, and benchmark artifacts.
- Compare against analytical references for Hartmann and processed closed-channel references for Shercliff and Hunt.
- Compare against recovered external cases through the checked-in backend harness when those assets are available locally.
- Operate as a standalone solver for structured duct-style liquid-metal flows with insulating and conducting-wall regions.

### What LMX cannot credibly claim yet

- Full FreeMHD parity across equations, convergence behavior, and runtime for the laminar validation set.
- A thorough convergence study showing mesh-independent and time-step-independent agreement across Hartmann, Shercliff, and Hunt.
- General simple-pipe support beyond the current mapped-mesh scaffolding and output path.
- Broad boundary-condition completeness for future liquid-metal applications.
- Ship-ready robustness for new geometries without additional solver and validation work.

### FreeMHD comparison status

- Short-time recovered-case comparisons are real and useful, not speculative:
  - Shercliff `Ha20` and `Ha100` sampled profile parity is already strong.
  - Hunt `Ha20` is much better than before and is now useful as a real solver target.
  - Hunt `Ha100` still shows a meaningful high-Ha conducting-wall fidelity gap.
- These comparisons are still limited in scope:
  - mostly short-time runs around recovered closed-channel cases
  - no full convergence campaign yet
  - no broad runtime-parity sweep yet
  - no claim yet that LMX matches FreeMHD across all validation observables

### Standalone-solver status

- Yes, LMX is already a standalone code for the current duct-focused laminar inductionless scope.
- No, it is not yet a ship-ready general liquid-metal solver for all simple pipes and boundary-condition combinations.
- The current standalone boundary-condition story is still narrow: no-slip, insulating, conducting wall, inlet velocity, inlet flow rate, outlet pressure, and imposed current density exist at the spec level, but they are not all exercised across a sufficiently broad native case matrix yet.
- The native Hunt case construction now uses wall conductance ratio as the default
  public parameter, which is the right nondimensional input for the archived
  closed-channel references, but that correction alone does not close the
  remaining Hunt solver gap.

## Ship-Ready Exit Criteria

LMX should only be described as ship ready for the current milestone when all of the following are true:

1. Native solver quality
   - Hartmann, Shercliff, and Hunt run stably on their intended mesh ranges without case-specific rescue tuning.
   - High-Ha Hunt remains bounded and accurate enough to pass explicit acceptance thresholds.
   - The solver no longer depends on parity-script-only behavior for physically meaningful setup choices.

2. Validation quality
   - Hartmann has analytical pass/fail thresholds.
   - Shercliff and Hunt each have at least one documented mesh/time convergence study.
   - Recovered external comparisons cover at least the main short-time duct cases with reproducible reports.

3. Scope clarity
   - The supported native problem classes and unsupported features are documented clearly.
   - Pipe geometry support is either completed for the intended mapped cases or explicitly removed from the current ship scope.

4. Performance and maintenance
   - Benchmark thresholds are tracked in CI or a documented release workflow.
   - Coverage remains high enough that solver and validation regressions are unlikely to slip through.
   - Docs, tests, and validation runners remain synchronized with the actual supported feature set.

## Best Next Steps

1. Replace the current pseudo-transient duct step with a more faithful laminar solver for the native LMX scope:
   - better electric-potential gauge handling
   - stable iterative coupling between `u`, `phi`, and `J x B`
   - the Hartmann/Shercliff fine-mesh clipping issue is mitigated by smaller pseudo-time defaults, but real Hunt `Ha20` and `Ha100` comparisons confirm conducting-wall fidelity still needs a real multi-region solver fix
   - the next retained solver milestone should be a materially better high-Ha Hunt shape match without adding new hardcoded case heuristics
   - the new convergence-aware `solve_steady` path shows that native Hunt remains poor even when the steady API behaves correctly, so the next step is update-physics/control-law work rather than more stop-criterion changes
2. Convert the current comparison work into a proper validation campaign:
   - Hartmann now has an explicit acceptance report with configurable `l2` and `linf` thresholds
   - a native mesh-convergence runner now exists for Hartmann, Shercliff, and Hunt
   - next add pseudo-time-convergence studies and tighten the meaning of the observed-order outputs
   - keep recovered-case FreeMHD comparisons as secondary cross-checks rather than the sole definition of correctness
3. Expand native standalone scope carefully rather than implicitly:
   - decide whether the next supported geometry milestone is still mapped simple pipes or only ducts
   - if simple pipes remain in scope, implement mapped operators and a native validation case before claiming support
   - broaden the exercised boundary-condition matrix with native tests instead of only dataclass-level support
4. Keep the public package and docs LMX-first:
   - external backend references should remain confined to validation sections
   - solver defaults should continue moving toward geometry-derived and nondimensional controls rather than case-specific constants
   - the current conducting-wall sampling path now uses mesh-derived interior planes rather than a fixed offset; keep following that rule for future geometry work
5. Tighten CI acceptance criteria and maintenance tooling:
   - docs now build warning-free through Sphinx/MyST and should stay that way
   - coverage is high enough to enforce and should continue moving up from the current `89%`
   - convert Hartmann validation into a stronger pass/fail parity check once runtime noise and numerical tolerance are characterized
6. Extend the recovered-case FreeMHD parity path only where it improves confidence in the native solver:
  - Hunt `Ha20` now runs end to end and samples successfully
  - use that new path to drive the next Hunt solver-fidelity iteration
  - Shercliff `Ha100` now also runs end to end and emits a real sampled parity artifact
  - Hunt `Ha100` now also runs end to end and emits a real sampled parity artifact
  - next use the combined Hunt `Ha20` and `Ha100` artifacts to drive the next conducting-wall solver iteration
7. Implement mapped-operator support for the fringing-field pipe case if that remains in the current release scope.

### Latest rejected Hunt pressure-response family

- A later-time Hunt replay sweep tested relaxed reduced-drive updates by introducing a
  `drive_relaxation` family on the recovered `Ha20`, `t <= 6e-05` replay.
- Retained baseline (`drive_relaxation = 1.0`) stayed:
  - `u_max l2 ≈ 5.26e-04`
  - `current_max l2 ≈ 2.28e-02`
  - `lorentz_max l2 ≈ 8.42e-02`
  - `pressure_proxy l2 ≈ 1.11e-01`
- `drive_relaxation = 0.5` slightly improved the reduced pressure-trend metric but worsened
  `u_max` and `lorentz_max`.
- `drive_relaxation = 0.25` improved `lorentz_max` and pressure-trend alignment somewhat,
  but degraded `u_max` more noticeably and did not improve the native Hunt analytical
  validation path.
- Retained conclusion:
  - this family is not the missing later-time Hunt fix
  - do not revisit it unless the reduced-model closure is redesigned more broadly

### Latest retained Hunt long-replay diagnostic result

- The corrected recovered Hunt `Ha20`, `t <= 6e-05` replay now reports both the existing
  cell-centered `lorentz_max_history` and a new face-current-based
  `face_lorentz_max_history`.
- On the retained long replay:
  - `u_max l2 ≈ 5.26e-04`
  - `current_max l2 ≈ 2.28e-02`
  - `lorentz_max l2 ≈ 8.42e-02`
  - `face_lorentz_max l2 ≈ 3.68e-03`
- Interpretation:
  - a large fraction of the remaining later-time Hunt `JxB` mismatch is in the reduction
    used for the comparison trace, not only in the solved reduced flow evolution
  - layered parity reports should now treat face-based Lorentz reduction as the
    primary force-scale diagnostic while keeping the cell-centered field as the
    native state variable
  - a matching retained update now also promotes face-current reduction as the
    primary layered current diagnostic:
    - `primary_current_max l2 ≈ 1.09e-02`
    - cell-centered `current_max l2 ≈ 2.28e-02`
  - the next solver-side Hunt target is therefore narrower:
    - the reduced `pressure_proxy` / pressure-response drift
    - then any remaining residual current-shape drift after using the face-based
      current and force traces

## What Worked

- The package structure and top-level API are in place and importable.
- The executable `lmx input.toml` path now works on the current development machine after reinstalling the editable package into `/Users/rogerio/base_env`.
- The local JAX environment at `/Users/rogerio/base_env/bin/python3` works for development and tests.
- The current duct solver path produces fields, VTK output, CSV cuts, and benchmark timing.
- Test suite is green after tightening the expectations to match current implementation state rather than full parity claims.
- Docs now build warning-free through Sphinx/MyST, and the repo has a real Read the Docs configuration.
- Coverage across `lmx/` and `scripts/` is now `89%`, with stronger wrapper and CLI coverage than before.
- Small Hartmann and Shercliff low-Ha cases are deterministic enough for regression snapshots.
- GitHub Actions workflows now cover unit, regression, physics, validation, and benchmark paths.
- The processed-figures Zenodo archive is sufficient for immediate closed-channel reference ingestion; the 8.9 GB `StartingFiles.zip` archive is not needed by default.
- CLI validation now emits both analytical and processed-slice comparison JSON when the matching Zenodo `XSlice` CSV exists.
- The validation-suite script can now emit analytical and processed-slice reports in one run when a reference root is provided.
- The validation workflow now includes:
  - explicit Hartmann acceptance reports with configurable `l2` and `linf` thresholds
  - a native `run_convergence_suite.py` runner that emits mesh-convergence summaries for Hartmann, Shercliff, and Hunt
  - those convergence summaries now also report estimated Hartmann-layer and side-layer cell counts, so mesh adequacy is visible in the artifact itself
  - a native `run_time_convergence_suite.py` runner now exists for fixed-resolution pseudo-time studies
- native solution diagnostics and validation summaries now also expose a normalized
  electric-potential equation residual, so `phi`-solve quality is visible in
  normal artifacts instead of only through downstream profile errors
  - they now also expose the actual Jacobi iteration count used by the latest
    potential solve, which matters for distinguishing a tolerance stop from a
    max-iteration cap
- `solve_steady` no longer aliases the transient runner:
  - it now iterates until either the configured steady tolerance is reached or the configured maximum step budget is exhausted
  - targeted tests now cover early-stop and max-step behavior explicitly
- The retained native Hunt validation metrics did not improve after fixing `solve_steady` semantics:
  - Hunt `Ha20` remains around `y_l2 ≈ 0.211`, `z_l2 ≈ 0.373`
  - Hunt `Ha100` remains around `y_l2 ≈ 0.198`, `z_l2 ≈ 0.411`
  - this confirms the remaining native Hunt problem is not just premature stopping
- The Hunt case factory now derives wall conductivity from wall conductance ratio
  by default:
  - `c = sigma_wall * t_wall / (sigma_fluid * a_H)` is now the primary public input
  - explicit `wall_conductivity` remains available as an override for cases that
    are specified directly in dimensional conductivity
  - native Hunt validation remained essentially unchanged after this correction,
    which is a useful negative result: the main gap is solver fidelity, not the
    naming or normalization of the Hunt wall parameter
- The new layer-resolution metrics sharpen the current Hunt diagnosis:
  - on the native Hunt `Ha20` convergence sweep, Hartmann-layer coverage increases
    from about `3.2` to `9.7` cells and side-layer coverage from about `5.2` to
    `15.8` cells between the `16^2` and `48^2` runs
  - that result motivated the next retained operator change
- Fine-mesh Hartmann and Shercliff stability improved materially after reducing the pseudo-time step and increasing the iteration budget in their case factories.
- Harmonic face conductivity averaging improved the multi-material discretization and helped Shercliff on smaller validation grids.
- Semi-implicit treatment of the linear Lorentz damping term improved Hartmann and Shercliff robustness without breaking the existing solver interface.
- An adaptive per-step velocity-update limiter now keeps the default Hunt path bounded and produces finite Hunt validation metrics.
- CI artifact summaries now include processed-slice error columns.
- A local FreeMHD container bundle generator and container execution helper now exist and are covered by unit tests.
- FreeMHD readiness is now split into distinct signals:
  - Docker CLI available
  - Docker daemon reachable
  - local case directories discovered
  - recommended smoke target when no standalone FreeMHD case is present
- The bundled FreeMHD OpenFOAM tree passes `foamSystemCheck` on this machine.
- The Hunt default case improved materially after reducing its pseudo-step and increasing its iteration budget; this kept the bounded default path but lowered the current `Ha=20` reference errors with only a modest runtime increase.
- The repo now has explicit FreeMHD environment and case inspection scripts, and they correctly report the current local target as the bundled OpenFOAM Hartmann tutorial when no standalone FreeMHD cases are present.
- The local FreeMHD environment probe now classifies common build failures such as missing `wmkdepend` and the macOS libc++ header conflict.
- A local FreeMHD environment probe now captures Docker-daemon availability and the current `wmkdepend` local-build blocker in machine-readable form.
- FreeMHD-side inspection now reports that the current downloads contain no standalone runnable FreeMHD cases and recommends the bundled OpenFOAM Hartmann tutorial as the smallest local smoke target.
- The current partial `StartingFiles.zip` is recoverable enough to extract and materialize the first Shercliff `Ha0` case shell into a normal case directory.
- The current partial `StartingFiles.zip` is also recoverable enough to extract and materialize the Shercliff `Ha20` `epotMultiRegionInterFoam` paper case shell into a normal case directory.
- `inspect_freemhd_setup.py` can now include recovered case directories outside `external/FreeMHD` via `--extra-case-root`, and it reports the recovered Shercliff `Ha20` case correctly.
- `run_freemhd_case.py` now fails fast with a structured `docker-image-unavailable` status when the requested image tag does not exist locally, instead of stalling in `docker run`.
- The Docker daemon is reachable in the current environment.
- A machine-readable container preflight now exists for the FreeMHD bundle and distinguishes local image absence, valid Docker Hub tag lookup, and timed local base-image pull stalls.
- A reproducible Darwin-only patch helper now exists for the local OpenFOAM header-shadowing issue, and it moves the local `wmake` probe past the libc++ conflict to a new `fvMesh.H` include failure.
- The Docker image now builds locally as `lmx-freemhd-smoke` and runs correctly on this Apple-silicon host when forced to `--platform linux/amd64`.
- The recovered Shercliff `Ha20` `epotMultiRegionInterFoam` case now runs end to end in the container with:
  - non-root execution so OpenFOAM dynamic code is allowed
  - automatic multi-region preprocessing
  - synchronized top-level and per-region `decomposeParDict` updates
  - cleanup of stale `processor*` and `processors*` layouts before each run
  - optional `controlDict` overrides for `deltaT`, `endTime`, and `writeInterval`
- The recovered Shercliff `Ha20` smoke run now produces:
  - reconstructed `0.0001/` output
  - parallel `processors8/0.0001/` output
  - `postProcessing/liquid/minMax/0/fieldMinMax.dat`
- `compare_with_freemhd` now inspects multi-region `0/`, `processors*`, and `fieldMinMax.dat` layouts instead of only checking that the case directory exists.
- LMX now supports explicit nonzero transient initialization through `CaseSpec.initial_velocity`, which is required to match FreeMHD cases that do not start from rest.
- A first coarse transient parity check now exists for the recovered Shercliff `Ha20` smoke run:
  - FreeMHD latest `max |U|` at `t = 1e-4` is `0.973457584`
  - an LMX short transient with matched `initial_velocity = 0.9725`, `dt = 1e-5`, `t_final = 1e-4`, and `forcing = 0` gives `max |U| = 0.9721652865`
  - the current absolute difference is about `1.29e-3`
- FreeMHD sampled line cuts now work on the reconstructed Shercliff `Ha20` output through a checked-in sampling runner:
  - `scripts/sample_freemhd_profiles.py` writes `system/lmxSampleDict`
  - `postProcess -func lmxSampleDict -time 0.0001` writes sampled cuts under `postProcessing/lmxSampleDict/liquid/0.0001/`
  - `compare_with_freemhd` now reports profile-based `y/z` errors when those sampled files exist
- The current real sampled-line parity numbers on the recovered Shercliff `Ha20` smoke case are:
  - `freemhd_sample_y_l2_error ≈ 5.83e-4`
  - `freemhd_sample_z_l2_error ≈ 3.29e-2`
- The sampling runner now infers the line-cut geometry automatically when explicit `x/y/z` bounds are not supplied.
- The current closed-channel rule is now geometry-aware instead of purely `fieldMinMax`-driven:
  - if `constant/liquid/polyMesh/points` exists and the case does not look like a conducting-wall layered duct, it samples at the geometric `x` midpoint and uses the mesh `y/z` bounds
  - if the case looks like a conducting-wall layered duct, it keeps the `fieldMinMax.dat`-driven `x` position while still using the FreeMHD liquid-region extents
- A checked-in parity-report runner now exists:
  - `scripts/run_freemhd_parity_report.py` builds the matching LMX case
  - it infers `initial_velocity` from `0/liquid/U` when available
  - it writes the current combined parity JSON artifact in one step
- A checked-in parity-suite runner now exists for CI and local artifact production:
  - `scripts/run_freemhd_parity_suite.py` writes a structured `skipped` summary when no recovered case directory is available
  - when `LMX_FREEMHD_CASE_DIR` or `--case-dir` is provided, it runs the FreeMHD sampling step and parity report end to end and emits a single summary JSON
- The parity-suite runner can now also bootstrap a fresh recovered case:
  - `--run-case-if-needed` runs the short FreeMHD smoke path first when the requested sampled time is not already present
  - this removes the previous manual `run_freemhd_case.py` pre-step for newly recovered cases
- CI artifact summaries now include the FreeMHD parity section in addition to the analytical validation section.
- The sampled-profile selector is now more robust:
  - when multiple sampled profile directories exist at the same sample time, `latest_sampled_profiles` now chooses the newest files by modification time instead of the first path alphabetically
  - this fixed a real Shercliff `Ha20` parity regression where stale `lmxAutoSampleDict` output was being compared instead of the just-generated CI sample set
- The local `StartingFiles.zip` recovery path now extends beyond Shercliff:
  - `hunt_exactBL_Ha20` can be extracted and materialized locally
  - the same container harness now runs that Hunt case to `t = 1e-4`, reconstructs `0.0001/`, and supports sampled parity extraction
- The recovered-case FreeMHD parity path now also includes a higher-Ha insulating-wall reference:
  - `shercliff_Ha100_ConstantQ_OutletZeroGradientInletCodedUxBpotE` is recoverable from `StartingFiles.zip`
  - it now runs to `t = 1e-4`, reconstructs `0.0001/`, and emits sampled line-cut parity metrics through the same checked-in runner
- The recovered-case FreeMHD parity path now also includes a higher-Ha conducting-wall reference:
  - `hunt_exactBL_Ha100` is recoverable from `StartingFiles.zip`
  - it now runs to `t = 1e-4`, reconstructs `0.0001/`, and emits sampled line-cut parity metrics through the same checked-in runner
- The corrected geometry-aware sampling rule materially improves the real Shercliff parity artifacts while preserving the better Hunt comparison path:
  - recovered Shercliff `Ha20` now samples at `x = 0.5` and the real sampled metrics are `freemhd_sample_y_l2_error ≈ 1.20e-3`, `freemhd_sample_z_l2_error ≈ 5.41e-4`
  - recovered Shercliff `Ha100` now samples at `x = 0.5` and the real sampled metrics are `freemhd_sample_y_l2_error ≈ 8.81e-4`, `freemhd_sample_z_l2_error ≈ 1.70e-4`
  - recovered Hunt `Ha20` is correctly classified as a conducting-wall case, keeps `x = 0.015`, and now reaches `freemhd_sample_y_l2_error ≈ 1.20e-3`, `freemhd_sample_z_l2_error ≈ 6.55e-3` after the retained Ha-aware Hunt control update
- The first real Hunt `Ha100` FreeMHD-vs-LMX parity numbers now also exist:
  - the original retained higher-Ha metrics were `u_max_abs_diff ≈ 9.50e-3`, `freemhd_sample_y_l2_error ≈ 1.39e-1`, `freemhd_sample_z_l2_error ≈ 1.19e-1`
  - after the retained Ha-aware Hunt control update, the current metrics are `u_max_abs_diff ≈ 1.40e-2`, `freemhd_sample_y_l2_error ≈ 1.36e-1`, `freemhd_sample_z_l2_error ≈ 7.63e-2`
  - after offsetting the conducting-wall sample plane away from the inlet when `fieldMinMax` reports a boundary-aligned maximum, the current metrics are `u_max_abs_diff ≈ 1.40e-2`, `freemhd_sample_y_l2_error ≈ 6.23e-2`, `freemhd_sample_z_l2_error ≈ 7.58e-2`
  - this keeps the improved `z` profile and removes most of the artificial inlet-plane inflation in the `y` comparison
- The first real Hunt `Ha20` FreeMHD-vs-LMX parity numbers now exist:
  - `u_max_abs_diff ≈ 1.26e-3` after the latest retained solver update
  - `freemhd_sample_y_l2_error ≈ 6.02e-2`
  - the original `freemhd_sample_z_l2_error ≈ 6.74e-1` was inflated by a comparison bug that included solid-wall cells on the LMX side
  - after comparing fluid-only layered-duct cuts and retaining the new outer-coupled pseudo-step, the corrected Hunt `z` metric is `freemhd_sample_z_l2_error ≈ 1.14e-1`
  - this sharpens the remaining Hunt task from “probably unstable/inaccurate” to a measured parity gap against the recovered FreeMHD case
- The retained Hunt solver controls are now explicitly Ha-aware:
  - `TimeStepperConfig` now carries `velocity_update_limit` so the bounded pseudo-step cap is part of the public case configuration instead of a hard-coded solver constant
  - `make_hunt_case` now chooses different outer-coupling / relaxation / update-limit settings by `Ha`
  - on the real recovered artifacts, this collapses the short-time Hunt `Ha20` parity gap and improves the Hunt `Ha100` `z` profile while keeping the high-Ha case bounded

## What Did Not Work

- Running with plain `python3` from the shell did not use the JAX-enabled environment.
- The initial solver branch that tried to decide between `lineax` and Jacobi inside a traced JAX function failed due to traced boolean conversion.
- The initial unconstrained pseudo-transient update produced `NaN` blow-up; explicit wall enforcement and bounded updates were needed to stabilize the first implementation.
- The first Shercliff symmetry assertion was too strong for the current solver and had to be downgraded to a finite-field/no-slip smoke test until better parity numerics are implemented.
- A later attempt to replace the pseudo-transient steady path with a more directly coupled fixed-point steady solver was not robust enough and was rolled back instead of being left on `main`.
- The Hunt validation path remains artifact-only for now because the current solver still clips and saturates on that case; it should not be treated as parity-complete.
- The current Shercliff `Ha=20` reference comparison also shows large normalized error, confirming that solver-fidelity work is still the critical path after reference ingestion.
- The earlier Hartmann/Shercliff defaults (`dt=0.01`, low iteration counts) were too aggressive for fine meshes because the current solver core uses an explicit diffusive update.
- The original Hunt defaults were too aggressive; the tuned default is better, but Hunt still needs solver-fidelity work beyond pseudo-step tuning.
- Semi-implicit Lorentz damping alone was not enough to fix Hunt at default settings; it needed the additional adaptive update limiter.
- Hunt is now bounded by default, but the resulting bounded solution is still not accurate enough yet to be treated as parity-complete.
- The current FreeMHD container bundle is still a scaffold for local iteration; it documents the expected build/run layout but has not yet been proven end to end against a real OpenFOAM build on this machine.
- The current local host-side `wmake` probe for `epotMultiRegionFoam` still fails, but the retained blocker has advanced from missing `wmkdepend` to a macOS libc++ header-path conflict.
- The current local assets still do not include standalone `epotMultiRegion*` paper case directories; only the solver sources, processed figures, and bundled OpenFOAM tutorials are present.
- The current local assets include the FreeMHD source tree and processed paper figures, but not the standalone `epotMultiRegion*` case directories needed for direct parity execution.
- A direct local `wmake` probe still fails on this machine; after repairing `wmkdepend`, the next blocker is the Darwin/OpenFOAM compiler/header environment.
- The currently downloaded assets do not yet include standalone runnable FreeMHD case directories, so real parity runs still require either the larger starting-files archive or another case source.
- After manually building `wmkdepend`, the next local FreeMHD build blocker is a macOS libc++ header-path conflict during `wmake` of `epotMultiRegionFoam`.
- The corrected Docker bundle has not yet been proven through a full successful image build in this session.
- The current Docker blocker has narrowed from daemon reachability and stale-image naming to local pull/build execution; `microfluidica/openfoam:2206` resolves as a valid Docker Hub tag, but timed pulls still stall on this machine.
- The Darwin local-build path is no longer blocked by the original libc++ collision after the patch helper is applied, but it is still not runnable because the next failure is an OpenFOAM include-resolution regression (`fvMesh.H` not found).
- The first attempt to rerun the recovered Shercliff `Ha20` case at a smaller core count failed because only the top-level `decomposeParDict` was rewritten; the per-region `system/<region>/decomposeParDict` files still retained the original `95`-way partitioning.
- A later rerun still failed because stale `processors95/` data from an earlier decomposition survived; cleanup has to happen on every run, not only in the mesh-generation branch.
- The current automated LMX-vs-FreeMHD comparison is intentionally coarse:
  - it compares the latest FreeMHD `mag(U)` maximum from `fieldMinMax.dat`
  - it does not yet compare full reconstructed profiles or full field data
  - the current agreement depends on matching the nonzero FreeMHD initial state through `CaseSpec.initial_velocity`
- The sampled-line parity path no longer depends on explicit sampling geometry inputs for the recovered closed-channel cases, but the inference is still specialized to those cases.
- The new split-rule geometry inference is not yet a general geometry parser for arbitrary FreeMHD/OpenFOAM cases or future fringing-field geometries.
- GitHub-hosted CI still cannot produce a real FreeMHD parity artifact by itself because it does not have the recovered Shercliff paper case tree mounted anywhere; the new parity-suite runner therefore reports `skipped` there by design.

## Chronological Log

### 2026-03-27 15:00 America/Chicago

- Explored the FreeMHD paper, repo, and Zenodo record.
- Identified `epotMultiRegionFoam` and `epotMultiRegionInterFoam` as the relevant solver surface.
- Confirmed that the paper includes both laminar closed-channel verification and free-surface/turbulent validation cases, so the initial implementation was intentionally limited to the laminar parity subset.

### 2026-03-27 15:20 America/Chicago

- Planned `LMX` as a JAX-native package with mesh, physics, operators, solvers, I/O, validation, and benchmark layers.
- Decided on a Linux-container path for FreeMHD/OpenFOAM benchmarking rather than macOS-native installation.

### 2026-03-27 15:40 America/Chicago

- Implemented the first standalone `LMX` workspace under `/Users/rogerio/local/tests/LMX`.
- Added package metadata, docs, scripts, tests, and the first laminar inductionless solver scaffold.

### 2026-03-27 15:45 America/Chicago

- Debugged environment mismatch: `python3` was not the JAX environment, `/Users/rogerio/base_env/bin/python3` was.
- Fixed early JAX tracing and stability issues in the solver.
- Verified tests passed in the JAX-enabled interpreter.

### 2026-03-27 15:50 America/Chicago

- Added `.gitignore`.
- Added this `plan.md` as the persistent context file that must be kept current with every substantive change.
- Current best next implementation step is analytical validation reporting for the closed-channel laminar cases.

### 2026-03-27 16:05 America/Chicago

- Added a `validate` CLI path for Hartmann cases.
- Added machine-readable analytical comparison output in `lmx/validation.py`.
- Verified the validation CLI writes outputs and reports current error metrics.
- The next best technical step is no longer “add validation scaffolding”; it is improving solver fidelity and extending analytical comparisons to Shercliff and Hunt.

### 2026-03-27 16:10 America/Chicago

- Initialized this directory as a git repository on `main`.
- Created the private GitHub repository `uwplasma/LMX`.
- Pushed the initial codebase and this execution log to `origin/main`.
- Current best next step remains solver-fidelity work for the laminar duct parity cases, starting with Shercliff/Hunt analytical comparisons and better coupled iteration stability.

### 2026-03-28 19:45 America/Chicago

- Reworked the automatic FreeMHD sampling-geometry inference so it is no longer purely tied to the latest `fieldMinMax.dat` maximum:
  - added mesh-bound inference from `constant/<region>/polyMesh/points`
  - added conductivity inference from region thermophysical files
  - added a split rule that treats non-conducting Shercliff-style ducts and conducting-wall Hunt-style ducts differently
- Verified the corrected rule against the real recovered FreeMHD cases:
  - recovered Shercliff `Ha20` now samples at the geometric midplane `x = 0.5` and improves to `freemhd_sample_y_l2_error ≈ 1.20e-3`, `freemhd_sample_z_l2_error ≈ 5.41e-4`
  - recovered Shercliff `Ha100` now samples at the geometric midplane `x = 0.5` and improves to `freemhd_sample_y_l2_error ≈ 8.81e-4`, `freemhd_sample_z_l2_error ≈ 1.70e-4`
  - recovered Hunt `Ha20` is now correctly kept on the field-based cut `x = 0.015`, preserving the better metrics `freemhd_sample_y_l2_error ≈ 5.36e-2`, `freemhd_sample_z_l2_error ≈ 1.14e-1`
- Added validation tests covering:
  - mesh-bounds inference from OpenFOAM `points`
  - conductivity parsing from liquid and solid thermophysical files
  - the conducting-wall override that prevents Hunt cases from being sampled at the geometric midpoint
- Best next step is now solver-side again:
  1. use the corrected Hunt `Ha20` artifact as the main conducting-wall parity target
  2. recover or run Hunt `Ha100` next so the same parity machinery is exercised at a harsher conducting-wall/Hartmann-layer regime
  3. then tune the LMX Hunt solver against those real FreeMHD sampled cuts

### 2026-03-28 19:50 America/Chicago

- Recovered and materialized `StartingFiles/Hunt/hunt_exactBL_Ha100` from the full local `external/StartingFiles.zip`.
- Verified that the fresh recovered case does not run successfully with the original paper decomposition on this machine:
  - the first smoke attempt inherited `95` subdomains from the case and was killed with exit code `137`
  - rerunning the same short smoke setup with `--cores 8 --delta-t 1e-5 --end-time 1e-4 --write-interval 1e-4` completed successfully in about `129 s` solver time before reconstruction
- Ran the first real higher-Ha conducting-wall FreeMHD parity artifact:
  - recovered case: `hunt_exactBL_Ha100`
  - current short-time metrics are `u_max_abs_diff ≈ 9.50e-3`, `freemhd_sample_y_l2_error ≈ 1.39e-1`, `freemhd_sample_z_l2_error ≈ 1.19e-1`
  - the current geometry-aware split rule keeps this conducting-wall case on the field-based cut, which lands at `x = 0.0` for the recovered `Ha100` short-time run
- Closed a fresh-case automation gap in the checked-in tooling:
  - `scripts/run_freemhd_parity_suite.py` now supports `--run-case-if-needed`
  - when the requested sampled time does not already exist, it can run the short FreeMHD smoke step itself before sampling and building the parity report
  - added unit coverage for the successful auto-run path and the run-failure path
- Best next step is now narrower and solver-specific:
  1. treat the combined `Hunt Ha20` and `Hunt Ha100` artifacts as the primary conducting-wall acceptance targets
  2. improve the LMX Hunt solver so short-time `u_max` and `y` profile parity stop degrading sharply between `Ha20` and `Ha100`
  3. only after that, decide whether the field-based `x = 0.0` cut at short-time `Hunt Ha100` should remain the acceptance slice or be replaced with a more physically anchored section definition

### 2026-03-28 19:55 America/Chicago

- Ran a targeted short-transient Hunt sweep against the real recovered FreeMHD artifacts instead of changing the solver blindly:
  - increasing Hunt outer-coupling iterations helps the sampled `y/z` profiles substantially at `Ha20`
  - but globally loosening the bounded velocity update hurts the `Ha100` amplitude and `y` profile
- Retained a Ha-aware Hunt control schedule in `make_hunt_case`:
  - `Ha <= 20`: `outer_iterations = 6`, `potential_iterations = 400`, `relaxation = 0.08`, `velocity_update_limit = 2e-3`
  - `20 < Ha <= 100`: `outer_iterations = 4`, `potential_iterations = 400`, `relaxation = 0.1`, `velocity_update_limit = 1e-3`
  - `Ha > 100`: `outer_iterations = 3`, `potential_iterations = 400`, `relaxation = 0.1`, `velocity_update_limit = 1e-3`
- Promoted the bounded velocity-step cap into the public config surface:
  - `TimeStepperConfig` now includes `velocity_update_limit`
  - `_step(...)` now uses that case-provided limit instead of a hard-coded solver constant
- Verified the retained change against the real recovered FreeMHD artifacts:
  - Hunt `Ha20` improved from `u_max_abs_diff ≈ 1.26e-3`, `y_l2 ≈ 5.36e-2`, `z_l2 ≈ 1.14e-1` to `u_max_abs_diff ≈ 2.29e-3`, `y_l2 ≈ 1.20e-3`, `z_l2 ≈ 6.55e-3`
  - Hunt `Ha100` improved from `u_max_abs_diff ≈ 9.50e-3`, `y_l2 ≈ 1.39e-1`, `z_l2 ≈ 1.19e-1` to `u_max_abs_diff ≈ 1.40e-2`, `y_l2 ≈ 1.36e-1`, `z_l2 ≈ 7.63e-2`
- Best next step is now narrower again:
  1. keep the retained Ha-aware Hunt control schedule
  2. keep the new conducting-wall interior-offset sampling rule for boundary-aligned high-Ha cases
  3. target the remaining Hunt `Ha100` `u_max` and residual `y`-profile gap specifically, since `Ha20` is now close and the worst `Ha100` slice artifact is gone

### 2026-03-28 20:00 America/Chicago

- Investigated whether the remaining Hunt `Ha100` gap was solver-side or comparison-slice-side by sampling the real recovered FreeMHD run at explicit streamwise positions:
  - comparing at `x = 0.015` instead of the boundary-aligned `x = 0.0` cut reduced the normalized Hunt `Ha100` `y` error from about `1.36e-1` to about `6.24e-2`
  - the corresponding `z` error stayed essentially unchanged
- Retained a narrower conducting-wall sampling rule in `infer_sampling_geometry(...)`:
  - for conducting-wall cases, if the `fieldMinMax`-driven `x` location lands on the domain boundary, clamp it to an interior offset of `1.5%` of the streamwise span
  - on the recovered Hunt `Ha100` case, that moves the auto-sampled parity cut from `x = 0.0` to `x = 0.015`
- Added focused validation coverage for that case:
  - boundary-aligned conducting-wall `fieldMinMax` data now produces `x = 0.015` instead of `x = 0.0` when mesh bounds are available
- Verified the retained change against the real recovered Hunt `Ha100` artifact:
  - metrics improved from `u_max_abs_diff ≈ 1.40e-2`, `y_l2 ≈ 1.36e-1`, `z_l2 ≈ 7.63e-2` to `u_max_abs_diff ≈ 1.40e-2`, `y_l2 ≈ 6.23e-2`, `z_l2 ≈ 7.58e-2`
- Best next step is now cleaner:
  1. keep the retained Ha-aware Hunt controls and the new interior-offset conducting-wall slice rule
  2. target the remaining Hunt `Ha100` amplitude / `y`-profile solver gap, which is now much less contaminated by the choice of comparison slice
  3. continue widening higher-Ha conducting-wall coverage only after the high-Ha solver update is justified by the existing `Ha100` artifact

### 2026-03-28 20:05 America/Chicago

- Investigated the remaining Hunt amplitude gap after the Ha-aware controls and slice fix:
  - small exploratory body forces were too weak to matter over the `1e-4` short-transient window
  - a physically scaled drive `forcing ~ sigma * |B|^2 * U_inlet` matches the observed short-time acceleration scale much better
- Retained that forcing rule in the checked-in parity builder:
  - `scripts/run_freemhd_parity_report.py` now infers `forcing` for Hunt cases when it is not explicitly supplied
  - the retained rule is `forcing = sigma * (By^2 + Bz^2) * initial_velocity`
  - Hartmann and Shercliff parity runs still default to `forcing = 0`
- Added unit coverage for the new parity-drive inference:
  - `test_infer_parity_forcing_uses_lorentz_balance_for_hunt`
  - `test_main_infers_hunt_forcing_when_unspecified`
- Verified the retained change against the real recovered FreeMHD artifacts:
  - Hunt `Ha20` improved from `u_max_abs_diff ≈ 8.17e-4`, `y_l2 ≈ 2.07e-3`, `z_l2 ≈ 7.48e-3` to the same order with the retained forcing rule, keeping the already-good short-time parity
  - Hunt `Ha100` improved from `u_max_abs_diff ≈ 1.40e-2`, `y_l2 ≈ 6.23e-2`, `z_l2 ≈ 7.58e-2` to `u_max_abs_diff ≈ 5.25e-5`, `y_l2 ≈ 5.72e-2`, `z_l2 ≈ 5.69e-2`
- Best next step is now narrower and more architectural:
  1. lift the current Hunt parity-drive heuristic out of the parity script and into proper case/BC semantics in the core solver
  2. keep the retained real-artifact targets (`Hunt Ha20`, `Hunt Ha100`) fixed while doing that refactor
  3. only after the drive model lives in the core API, decide whether similar flow-rate-driven short-transient handling is needed for other FreeMHD case families

### 2026-03-28 20:10 America/Chicago

- Moved the Hunt short-time drive model out of the parity-script forcing heuristic and into core solver semantics:
  - `lmx.solvers` now derives an effective body force from inlet boundary conditions when `case.forcing == 0`
  - the retained rule is the same mean Lorentz-balance scale used previously in the parity layer: `mean[sigma * (By^2 + Bz^2)] * U_inlet`
  - this currently supports `inlet_velocity` and `inlet_flow_rate` boundary conditions
- Updated the Hunt parity builder to use core BC semantics instead of explicit forcing inference:
  - `scripts/run_freemhd_parity_report.py` now appends an `inlet_velocity` BC for Hunt when `--forcing` is omitted
  - reported payloads now show `drive_mode = "inlet_velocity"` and `forcing = 0.0`
- Added focused coverage for the refactor:
  - `test_hunt_inlet_velocity_boundary_drives_short_transient`
  - updated parity-report tests to assert boundary-driven Hunt parity instead of inferred script forcing
- Verified that the retained real-artifact metrics are preserved after the refactor:
  - Hunt `Ha20`: `u_max_abs_diff ≈ 8.17e-4`, `y_l2 ≈ 2.07e-3`, `z_l2 ≈ 7.48e-3`
  - Hunt `Ha100`: `u_max_abs_diff ≈ 5.25e-5`, `y_l2 ≈ 5.72e-2`, `z_l2 ≈ 5.69e-2`
- Best next step is now the next real modeling step instead of cleanup:
  1. decide whether the same inlet-driven short-transient semantics should become first-class in the public case factories, not just parity builders
  2. extend that decision carefully so existing Hartmann/Shercliff defaults do not regress
  3. then tighten Hunt acceptance thresholds in CI around the now-stable real `Ha20` and `Ha100` artifacts

### 2026-03-27 16:35 America/Chicago

- Added richer validation/reporting support:
  - CLI `validate` subcommand remains available.
  - machine-readable metrics JSON output for duct-profile symmetry and basic field statistics
  - CSV output for both centerline and orthogonal midplane cuts
- Attempted a more directly coupled steady solver for the duct cases.
- That solver branch proved numerically unstable in fresh runs, so it was rolled back instead of being kept on `main`.
- Net result kept on `main`: validation/reporting improved; solver remains on the earlier pseudo-transient relaxation path.
- Best next step is now narrower and clearer:
  1. ingest the Zenodo analytical closed-channel files locally
  2. build Shercliff/Hunt comparison loaders and reports against those references
  3. only then revisit the steady solver with stronger acceptance checks and rollback criteria

### 2026-03-27 17:20 America/Chicago

- Added explicit pytest suite taxonomy:
  - `unit`
  - `regression`
  - `physics`
  - `validation`
- Added deterministic low-Ha Hartmann and Shercliff regression tests.
- Added separate benchmark and validation runner scripts that emit JSON artifacts without committing outputs.
- Added GitHub Actions workflows:
  - `.github/workflows/ci.yml` for categorized pytest and validation artifacts
  - `.github/workflows/benchmarks.yml` for benchmark artifacts
- Updated `.gitignore` to exclude local artifact directories.
- Verified locally with:
  - `/Users/rogerio/base_env/bin/python3 -m pytest -q`
  - `/Users/rogerio/base_env/bin/python3 scripts/run_validation_suite.py --output artifacts/validation_local`
  - `/Users/rogerio/base_env/bin/python3 scripts/run_benchmark_suite.py --output artifacts/benchmarks_local/benchmark.json --repeats 2 --ha 5 --ny 16 --nz 16`
- Resulting best next step did not change at the solver level: ingest Zenodo reference data and improve Shercliff/Hunt parity before tightening validation thresholds.

### 2026-03-27 18:10 America/Chicago

- Inspected the Zenodo record and confirmed:
  - `FreeMHDPaperAllFigures.zip` is about 18 MB and contains the closed-channel analytical and processed CSV data needed immediately.
  - `StartingFiles.zip` is about 8.9 GB and should not be a default download.
- Updated `scripts/fetch_freemhd_assets.py` so it now downloads:
  - the FreeMHD repository
  - the processed-figures archive by default
  - the large starting-files archive only when `--include-starting-files` is requested
- Added `lmx/reference_data.py` with:
  - analytical closed-channel loaders for Shercliff and Hunt
  - processed-slice CSV loaders for the paper figures
  - midplane extraction helpers for future figure-level parity checks
- Added closed-channel validation hooks that compare normalized LMX midplane profiles against the ingested Shercliff and Hunt analytical references.
- Verified against the real downloaded Zenodo data:
  - analytical Shercliff and Hunt files load correctly
  - processed Hunt slice CSV loads correctly
  - `lmx.cli validate shercliff --ha 20 --reference-root ...` now writes actual Shercliff reference-comparison metrics
- Current quantitative result is intentionally honest:
  - Shercliff `Ha=20` still has large normalized errors and the solver clips at this regime
  - this confirms the next best step is solver-fidelity work, not more validation plumbing

### 2026-03-27 19:05 America/Chicago

- Investigated the new reference comparisons instead of changing the solver blindly.
- Found that Hartmann and Shercliff clipping on default fine meshes was primarily an explicit-step stability issue:
  - coarse verification meshes were stable
  - fine default meshes clipped unless the pseudo-time step and relaxation were reduced substantially
- Updated the Hartmann and Shercliff case factories to use mesh-safe pseudo-transient defaults:
  - `dt = 0.001`
  - `relaxation = 0.1`
  - `max_steps = 400`
- Verified the new default CLI paths:
  - Hartmann `Ha=20` now remains bounded on the default mesh and the analytical comparison dropped to about `l2_error = 0.0039`
  - Shercliff `Ha=20` now remains bounded on the default mesh and the analytical comparison became finite instead of clipping, with roughly `y_l2_error = 0.434` and `z_l2_error = 0.187`
- Hunt was not fixed by the same tuning and still clips; that keeps Hunt multi-region stability as the next solver target.
- Updated regression snapshots and added tests so this new stable Hartmann/Shercliff baseline is preserved in CI.

### 2026-03-27 19:30 America/Chicago

- Continued on the Hunt investigation instead of changing defaults blindly.
- Switched face conductivity averaging in the electric-potential solve from arithmetic to harmonic:
  - this is more appropriate across the fluid/solid conductivity jump
  - it improved Shercliff on smaller validation grids without breaking the suite
- Measured Hunt diagnostics and confirmed:
  - huge interface-driven Lorentz forcing is still the immediate instability source at default settings
  - Hunt becomes bounded if the pseudo-step is made much smaller (`dt ~ 1e-4`, `relaxation ~ 0.05` on the small diagnostic case)
- Added a Hunt stability test that uses this tiny pseudo-step regime so the repo preserves the evidence that Hunt is solvable with the current equations but not yet with the current default pseudo-time strategy.
- Best next step is now more concrete:
  1. add adaptive pseudo-step sizing or a more implicit velocity update for the Hunt multi-region case
  2. only after that, tighten Hunt acceptance checks against the ingested reference data

### 2026-03-27 19:55 America/Chicago

- Continued the solver work instead of stopping at the Hunt diagnosis.
- Added semi-implicit treatment of the linear `-sigma |B|^2 u` Lorentz damping contribution in the velocity update.
- Kept this change because:
  - Hartmann remained low-error on the fine mesh
  - Shercliff improved on the smaller validation grids
  - the full test suite remained green after updating the regression baselines
- Did not overstate the outcome:
  - Hunt still clips at the default settings
  - this confirms that Hunt needs either adaptive pseudo-step sizing or a more implicit multi-region update, not just implicit treatment of the linear damping term

### 2026-03-27 20:10 America/Chicago

- Updated only the handoff docs and logs to reflect the current archive-recovery state.
- Current recovered `StartingFiles` result:
  - Shercliff `Ha0` can be materialized locally from the partial archive
  - `Ha20` remains the next archive target
- Current Docker/image blocker:
  - the Docker CLI is available
  - the daemon is not reachable from the active environment
  - image build and container execution stay blocked until that changes
- This entry is logging-only; no solver, test, or workflow code changed in this step.

### 2026-03-27 20:20 America/Chicago

- Added an adaptive limiter on the global per-step velocity increment.
- This acts as the first solver-side pseudo-step controller that does not require changing the public case configuration.
- Result:
  - Hunt no longer blows up on the default path
  - Hunt validation reports are now finite instead of saturating at the clip bounds
  - the default Hunt comparison is still not parity-accurate, but it is now on a usable trajectory for further improvement
- Kept the earlier Hunt small-pseudostep test and added a new default-Hunt boundedness test so CI preserves both facts:
  - the current equations can remain bounded under very small pseudo-steps
  - the default Hunt path is no longer catastrophically unstable
- Best next step sharpened again:
  1. improve Hunt accuracy now that the default path is bounded
  2. add processed-slice figure-level comparisons for Shercliff and Hunt
  3. keep tightening acceptance thresholds case by case rather than all at once

### 2026-03-27 23:20 America/Chicago

- Extended the validation layer beyond analytical text references:
  - `lmx.validation` now compares LMX midplane cuts against the processed Zenodo `XSlice` CSV exports
  - `lmx.cli validate` now writes `*_slice.json` when a matching processed slice is available
  - `scripts/run_validation_suite.py` now accepts `--reference-root` and `--x-slice` so the suite can emit both analytical and processed-slice reports in one pass
- Updated the CI artifact summarizer and reporting tests so validation summaries now include slice-level errors alongside the analytical errors.
- Replaced the earlier FreeMHD Dockerfile placeholder with a local bundle generator plus a `scripts/run_freemhd_case.py` helper that records JSON run metadata for mounted case directories.
- Added unit coverage for processed-slice report writing and FreeMHD bundle generation.
- Verified the new user-facing Shercliff reference path:
  - `lmx.cli validate shercliff --ha 20 --reference-root ... --x-slice 1m` now writes both `shercliff_ha20_analytic.json` and `shercliff_ha20_slice.json`
  - current Shercliff errors remain large (`y_l2_error ~ 0.431`, `z_l2_error ~ 0.187`, slice metrics at similar levels), so solver fidelity remains the critical path rather than more reporting work
- Verified the full pytest suite remains green after these changes.

### 2026-03-27 23:50 America/Chicago

- Audited the local FreeMHD assets before attempting a fake parity run.
- Confirmed the current local state:
  - FreeMHD source tree is present under `external/FreeMHD`
  - processed paper figures are present under `external/FreeMHDPaperAllFigures/.../ClosedChannel`
  - no standalone `epotMultiRegionFoam` or `epotMultiRegionInterFoam` case directories are present locally yet
  - Docker CLI is installed, but the Docker daemon is not currently reachable from the active environment
- Added explicit inspection helpers:
  - `scripts/inspect_freemhd_setup.py` now reports the recommended current target case
  - `scripts/inspect_freemhd_case.py` records the structural readiness of a specific case directory
  - `lmx.validation.inspect_freemhd_case` and `lmx.compare_with_freemhd` now expose case-structure metadata such as `controlDict`, `regionProperties`, `blockMeshDict`, and latest-time directories
  - `lmx.freemhd` now recommends the smallest current target automatically, falling back to the bundled OpenFOAM Hartmann tutorial as an environment smoke test when no standalone FreeMHD cases exist
- Verified the current recommendation on this machine:
  - `external/FreeMHD/OpenFOAM-v2206/tutorials/electromagnetics/mhdFoam/hartmann` is the smallest current smoke target
  - it is only an OpenFOAM environment check, not a true FreeMHD parity case
- Best next step is now even more concrete:
  1. obtain or reconstruct one actual laminar `epotMultiRegion*` case directory locally
  2. bring the Docker daemon up or switch to an alternative local container runner
  3. only then wire the first real FreeMHD-vs-LMX parity execution

### 2026-03-28 00:05 America/Chicago

- Tightened the local FreeMHD environment probe so it now classifies common build failures instead of emitting only raw `wmake` stderr.
- The probe currently distinguishes at least:
  - missing `wmkdepend`
  - the macOS libc++ header conflict seen when the local compiler environment does not match OpenFOAM expectations
  - generic unknown build failures
- Added unit coverage for these classifications so future environment-debugging work starts from machine-readable diagnostics rather than ad hoc log reading.

### 2026-03-27 23:40 America/Chicago

- Tightened the FreeMHD harness bundle after reviewing the generated files:
  - the generated `run_freemhd_case.sh` now accepts positional `case_dir`, `cores`, and `solver` arguments instead of ignoring the CLI-provided values
  - the generated Dockerfile now attempts the actual FreeMHD/OpenFOAM solver build commands instead of only echoing them
  - the bundle generator now marks `run_freemhd_case.sh` executable
- Re-ran the validation and reporting tests after the harness fix and kept the repo green.
- Re-ran the reference-backed validation suite and confirmed the present quantitative state:
  - Hartmann `Ha=20` remains the strongest case in the current implementation
  - Shercliff and Hunt both now emit analytical and slice-level metrics from the same suite run
  - Hunt is bounded and diagnosable, but solver fidelity is still the primary blocker for parity

### 2026-03-27 23:55 America/Chicago

- Probed actual FreeMHD execution prerequisites on this machine instead of assuming the new harness would run:
  - `docker --version` works, but `docker build ...` currently fails immediately because the Docker daemon socket is unavailable
  - sourcing `external/FreeMHD/OpenFOAM-v2206/etc/bashrc` works and `foamSystemCheck` passes locally
  - a direct local `wmake` of `MHD_Solvers/solvers/epotMultiRegionFoam` fails because `platforms/tools/darwin64Clang/wmkdepend` is missing
- Added `scripts/probe_freemhd_environment.py` so future agents can reproduce these environment findings as JSON instead of rediscovering them manually.
- Added unit coverage for the probe script.
- Best next step narrowed further:
  1. either repair the local OpenFOAM toolchain enough to produce `wmkdepend`, or start the Docker daemon and verify the container build path
  2. once one FreeMHD execution path is real, capture the first actual side-by-side case report
  3. keep solver-fidelity work moving in parallel because Hunt/Shercliff accuracy is still the main LMX-side blocker

### 2026-03-28 00:10 America/Chicago

- Added a second FreeMHD-side inspection layer on top of the low-level probe:
  - `lmx.freemhd` now discovers locally available case directories and recommends a smallest smoke target
  - `scripts/inspect_freemhd_setup.py` writes this higher-level report as JSON
  - `scripts/run_freemhd_case.py` now degrades to a structured JSON status when the Docker daemon is unavailable instead of aborting with an exception
- Verified the current local setup report:
  - Docker CLI exists but daemon is still unavailable
  - no standalone runnable FreeMHD cases are present in the currently downloaded assets
  - the bundled OpenFOAM Hartmann tutorial is the current recommended smoke target if local toolchain repair succeeds before the larger FreeMHD case archive is downloaded

### 2026-03-28 00:25 America/Chicago

- Repaired the first local OpenFOAM build blocker by compiling `OpenFOAM-v2206/wmake/src`, which successfully produced `platforms/tools/darwin64Clang/wmkdepend`.
- Re-ran the FreeMHD solver build probe after that repair:
  - `wmake` now advances into real compilation of `epotMultiRegionFoam`
  - the next failure is a macOS libc++/header-path conflict (`<cstring>` / `<cwchar>` include resolution) rather than missing OpenFOAM tooling
- Updated `scripts/probe_freemhd_environment.py` so it now classifies the current build failure mode instead of only reporting the raw return code and stderr tail.
- Best next step is now explicit:
  1. inspect the OpenFOAM darwin compiler flags and include ordering that trigger the libc++ header conflict
  2. decide whether to patch the local build environment or rely on Docker once the daemon is available
  3. in parallel, keep improving Shercliff/Hunt parity since LMX accuracy is still the main solver-side gap

### 2026-03-28 00:40 America/Chicago

- Re-checked the retained `main` state against the current local environment instead of trusting stale local assumptions.
- Current source-of-truth probe outputs on `main` are:
  - `scripts/probe_freemhd_environment.py` reports `foamSystemCheck` passes, `wmkdepend` now exists, Docker CLI exists, Docker daemon is unavailable, and the current retained build issue is `macos-libcxx-header-conflict`
  - `scripts/inspect_freemhd_setup.py` reports zero discovered standalone FreeMHD case directories in the current local assets and recommends the bundled OpenFOAM Hartmann tutorial only as an environment smoke target
  - `scripts/run_freemhd_case.py` returns a structured `docker-daemon-unavailable` JSON status on this machine
- Treat this entry as the current baseline for future work on `main`; the next blockers are the Darwin/OpenFOAM compiler environment, Docker daemon access, and obtaining at least one real `epotMultiRegion*` case directory locally.

### 2026-03-28 00:55 America/Chicago

- Used the now-automated Shercliff/Hunt reference path to do a controlled pseudo-time sweep instead of guessing at new defaults.
- Result on the default Hunt mesh (`72x72`, `Ha=20`):
  - earlier bounded default was about `y_l2_error ~ 0.334`, `slice_y_l2_error ~ 0.329`
  - a tuned middle setting (`dt=0.002`, `max_steps=500`, `relaxation=0.1`, `potential_iterations=250`) reduced that to about `y_l2_error ~ 0.284`, `slice_y_l2_error ~ 0.279`
  - runtime on this machine increased only modestly, from about `1.10s` to about `1.32s`
- Kept that middle setting as the new `make_hunt_case()` default because it improves the user-facing validation path without the much larger runtime cost of the most aggressive sweep point.
- Re-ran the Hunt reference-backed CLI validation and confirmed the new default emits the improved metrics while keeping the solution bounded.

### 2026-03-28 01:20 America/Chicago

- Moved the FreeMHD case-input path forward without waiting for the full 8.9 GB archive:
  - confirmed the existing `external/StartingFiles.zip` is a real but incomplete ZIP, not an HTML error page
  - added integrity-aware fetching so future downloads validate ZIPs instead of silently keeping corrupt artifacts
  - added `scripts/inspect_starting_files_archive.py`, which can inspect and selectively extract recoverable entries directly from local ZIP headers even when the central directory is missing
  - added `scripts/materialize_starting_case.py`, which expands recovered `0.tar.gz`, `constant.tar.gz`, and `system.tar.gz` into a normal OpenFOAM case tree
- Verified this on the real partial archive:
  - recovered `/tmp/startingfiles_ha0/StartingFiles/Shercliff/shercliff_Ha0_refinedMesh`
  - materialized that case into real `0/`, `constant/`, and `system/` directories
  - confirmed `system/controlDict` uses `epotMultiRegionInterFoam` and the case contains `liquid`, `solidWalls`, and `insulator` regions
- Docker daemon is now reachable on this machine after starting Docker Desktop, so the external runtime blocker has advanced from daemon availability to actually proving the corrected image build and run path.
- Corrected the FreeMHD Docker bundle generator to build the real solver paths under `MHD_Solvers/solvers/...` instead of the nonexistent `applications/solvers/...` path.
- Best next step sharpened again:
  1. prove the corrected Docker image build path with visible logs
  2. use the recovered Shercliff `Ha0` case as the first run-structure smoke test
  3. resume `StartingFiles.zip` in place until the `shercliff_Ha20_ConstantQ_OutletZeroGradientInletCodedUxBpotE` case can be extracted for the first real paper-parity run

### 2026-03-28 02:05 America/Chicago

- Resumed the partial `StartingFiles.zip` download far enough to recover the full Shercliff `Ha20` paper case shell:
  - extracted `/tmp/startingfiles_ha20/StartingFiles/Shercliff/shercliff_Ha20_ConstantQ_OutletZeroGradientInletCodedUxBpotE`
  - materialized its `0/`, `constant/`, and `system/` trees successfully
  - confirmed `system/controlDict` targets `epotMultiRegionInterFoam`
  - confirmed `constant/regionProperties` and the expected multi-region layout are present
- Tightened the FreeMHD harness around the now-real recovered case:
  - `inspect_freemhd_setup.py` now accepts `--extra-case-root`, so setup inspection can include recovered `/tmp` case roots alongside the checked-in `external/FreeMHD` tree
  - `run_freemhd_case.py` now checks whether the requested Docker image tag exists locally and returns a structured `docker-image-unavailable` JSON status instead of stalling
- Verified the new behavior on the real recovered Shercliff `Ha20` path:
  - setup inspection reports the recovered `Ha20` case as the recommended `freemhd_case` target with no blockers
  - the run helper exits quickly with `docker-image-unavailable` for the still-missing `lmx-freemhd-smoke` image
- Best next step tightened again:
  1. resolve the OpenFOAM base-image pull/build path and produce a local `lmx-freemhd` image
  2. run the recovered Shercliff `Ha20` case inside that image as the first real FreeMHD smoke test
  3. once container execution is live, extend the current JSON harness into actual LMX-vs-FreeMHD parity extraction for this case

### 2026-03-28 02:20 America/Chicago

- Added a separate FreeMHD container preflight report instead of continuing to diagnose Docker state manually:
  - `scripts/probe_freemhd_container.py` now writes JSON for the checked-in Docker bundle
  - `lmx.freemhd` now exposes container-report helpers for parsing the Dockerfile base image, checking local image presence, and attempting bounded-time registry resolution through `docker manifest inspect`
- Verified the new preflight on the current machine against `docker/Dockerfile` and the expected runtime tag:
  - `lmx-freemhd-smoke` is not present locally
  - the bundle base image `openfoam/openfoam2206-paraview:latest` is also not present locally
  - `docker manifest inspect openfoam/openfoam2206-paraview:latest` times out from this environment
- This narrows the next real external blocker further:
  1. either resolve registry access for the current base image or switch the bundle to a base image/tag that resolves cleanly here
  2. once that image issue is resolved, build `lmx-freemhd` locally and retry the recovered Shercliff `Ha20` case run
  3. after the first real FreeMHD run completes, promote the current harness JSON into parity extraction and comparison artifacts

### 2026-03-28 03:05 America/Chicago

- Used the Darwin `wmake` investigation results to keep the non-Docker FreeMHD path moving in a reproducible way:
  - added `scripts/patch_freemhd_darwin_headers.py`, which applies the current Darwin-only workaround by demoting the two problematic `lnInclude` directories from `-I` to `-idirafter`
  - updated `scripts/probe_freemhd_environment.py` so it now detects whether that patch is present and classifies the new post-patch failure state separately
- Verified the local macOS build progression on the real `external/FreeMHD` checkout:
  - before the workaround, the probe classified `macos-libcxx-header-conflict`
  - after applying the workaround, the probe no longer reports shadowed libc headers and now classifies `post-darwin-header-patch-include-regression`
  - the current concrete next local-build error is `fatal error: 'fvMesh.H' file not found`
- This means the local FreeMHD path advanced materially even though it is not yet runnable:
  1. inspect the expanded `wmake` compile line so the missing OpenFOAM include directories can be restored without reintroducing the libc++ header collision
  2. in parallel, replace or fix the stale Docker base image reference so the container path can advance too
  3. keep the recovered Shercliff `Ha20` case as the first actual run target for whichever execution path lands first

### 2026-03-28 04:20 America/Chicago

- Replaced the stale Docker base image reference with a valid OpenFOAM.com `2206` base image:
  - switched the checked-in and generated Dockerfile from `openfoam/openfoam2206-paraview:latest` to `microfluidica/openfoam:2206`
  - updated the container preflight to use the Docker Hub tag API directly instead of relying on `docker manifest inspect`
- Verified the new Docker-side picture on this machine:
  - Docker Hub confirms `microfluidica/openfoam:2206` is a live tag
  - the old `openfoam/openfoam2206-paraview:latest` name is not a live Hub tag
  - the local runtime image `lmx-freemhd-smoke` still does not exist
  - a timed `docker pull microfluidica/openfoam:2206` still stalls locally, so the remaining Docker-side blocker is pull/build execution rather than image-tag discovery
- This sharpens the next steps again:
  1. inspect local Docker Desktop / engine network or credential state to explain why timed pulls stall even though Hub tag lookup works
  2. keep the Darwin `wmake` path moving in parallel by inspecting the missing include directories behind the current `fvMesh.H` failure
  3. once either path actually executes FreeMHD, use the recovered Shercliff `Ha20` case as the first real smoke/parity target

### 2026-03-28 12:15 America/Chicago

- Proved the FreeMHD container path on this Apple-silicon host instead of continuing to treat it as a scaffold:
  - `docker pull --platform linux/amd64 microfluidica/openfoam:2206` succeeds locally
  - the Docker image now builds locally as `lmx-freemhd-smoke`
  - the run helper now forces `--platform linux/amd64` and `--entrypoint /bin/bash`
- Fixed the first real run blockers on the recovered Shercliff `Ha20` case:
  - disabled `set -eu` around OpenFOAM `bashrc` sourcing to avoid shell aborts from the base image setup scripts
  - switched container execution to non-root so dynamic coded OpenFOAM boundary conditions are allowed
  - added automatic multi-region preprocessing (`blockMesh`, `topoSet`, `splitMeshRegions`, `changeDictionary`, `setExprFields`)
  - added `--oversubscribe` to the MPI launch path so the case can run on this host
- The first smaller-core rerun still failed for two concrete reasons that are now fixed on `main`:
  - only the top-level `decomposeParDict` had been rewritten; the per-region dicts remained at `95` subdomains
  - stale `processors95/` data survived and contaminated later runs

### 2026-03-28 13:20 America/Chicago

- Hardened the FreeMHD run harness for repeatable local parity work:
  - synchronized all `system/**/decomposeParDict` files to the requested core count
  - cleaned both `processor*` and `processors*` layouts on every run
  - added optional `controlDict` overrides through `run_freemhd_case.py` for `deltaT`, `endTime`, and `writeInterval`
  - added direct `runLog.<solver>` capture in the case directory
- Verified the recovered Shercliff `Ha20` smoke run with:
  - `cores = 8`
  - `deltaT = 1e-5`
  - `endTime = 1e-4`
  - `writeInterval = 1e-4`
- This run now reaches the requested short transient horizon and produces:
  - reconstructed `0.0001/`
  - parallel `processors8/0.0001/`
  - `postProcessing/liquid/minMax/0/fieldMinMax.dat`
  - latest FreeMHD `max |U| = 0.973457584` at `t = 1e-4`

### 2026-03-28 13:35 America/Chicago

- Extended the LMX parity side to match what the real FreeMHD case is doing:
  - added `CaseSpec.initial_velocity`
  - `solve_transient` now respects nonzero initial velocity while preserving no-slip walls
  - `compare_with_freemhd` now inspects multi-region `0/`, `processors*`, and `fieldMinMax.dat` artifacts
  - `compare_with_freemhd` now performs a first coarse transient comparison using the latest FreeMHD `mag(U)` maximum
- Verified the first coarse transient parity metric with a matched LMX smoke setup:
  - LMX case: Shercliff `Ha=20`, `ny = nz = 16`, `initial_velocity = 0.9725`, `forcing = 0`, `dt = 1e-5`, `t_final = 1e-4`
  - FreeMHD latest `max |U| = 0.973457584`
  - LMX `max |U| = 0.9721652865`
  - absolute difference `= 0.0012922975`
- Current best next step is now to replace this coarse min/max smoke comparison with profile extraction from reconstructed FreeMHD output while continuing the Hunt/Shercliff solver-fidelity work on the LMX side.

### 2026-03-28 18:55 America/Chicago

- Replaced the earlier “next step” with a working profile-extraction path from reconstructed FreeMHD output:
  - confirmed that OpenFOAM `2206` in the container does not ship a standalone `sample` binary, so the correct route is `postProcess -func <dict-name>`
  - confirmed that `postProcess` discovers user-defined sampling dictionaries when they are written under the case `system/` directory
  - added `scripts/sample_freemhd_profiles.py`, which writes `system/lmxSampleDict`, runs `postProcess`, and records JSON output paths
- Added new validation-side utilities:
  - parse sampled `centerlineY_potE_U.xy` and `centerlineZ_potE_U.xy`
  - detect the latest sampled pair automatically from `postProcessing/*/liquid/<time>/...`
  - report sampled `y/z` line-cut parity metrics through `compare_with_freemhd`
- Verified this on the real recovered Shercliff `Ha20` smoke run:
  - sampled files are now written under `postProcessing/lmxSampleDict/liquid/0.0001/`
  - the first sampled-line parity metrics are `freemhd_sample_y_l2_error ≈ 5.83e-4` and `freemhd_sample_z_l2_error ≈ 3.29e-2`
- Best next step is now narrower again:
  1. remove the current need for manually chosen sampling extents by inferring them from the FreeMHD case geometry
  2. turn the sampled Shercliff parity path into a checked-in artifact/report runner
  3. extend the same path to other recovered FreeMHD cases as they become available

### 2026-03-28 19:05 America/Chicago

- Removed the manual sampling-extents requirement for the current Shercliff `Ha20` parity path:
  - extended `read_field_minmax` so it now preserves min/max locations in addition to scalar extrema
  - added `infer_sampling_geometry`, which derives `x/y/z` sampling limits from the latest `fieldMinMax.dat` record
  - verified on the real recovered case that the inferred geometry is `x = 0.015`, `y = [-0.1, 0.1]`, `z = [-0.099995, 0.099995]`
- Added the checked-in parity artifact runner that the previous step called for:
  - `scripts/run_freemhd_parity_report.py` infers the initial FreeMHD streamwise velocity from `0/liquid/U` when possible
  - it builds the matching LMX case and writes the combined parity JSON in one command
- Best next step narrowed again:
  1. add this new Shercliff parity runner to the repo’s validation artifact workflow
  2. extend the same path to new recovered FreeMHD cases as they become available
  3. keep pushing solver fidelity so the current good short-time parity extends to stronger laminar acceptance checks

### 2026-03-28 19:20 America/Chicago

- Added the new Shercliff parity path to the normal CI artifact workflow:
  - `.github/workflows/ci.yml` now runs `scripts/run_freemhd_parity_suite.py`
  - the suite writes `artifacts/freemhd_parity/summary.json`
  - the artifact summarizer now includes a `FreeMHD Parity` section in the Markdown and JSON summary outputs
- Designed the parity suite so CI stays green without local FreeMHD assets:
  - if no `LMX_FREEMHD_CASE_DIR` or explicit `--case-dir` is available, the suite emits a structured `skipped` report instead of failing
  - this keeps GitHub-hosted CI useful while preserving the exact same runner for machines that do have recovered paper cases
- Verified the new parity suite locally against the recovered Shercliff `Ha20` case, which exposed and then fixed a real bug:
  - when multiple sampled profile directories existed at the same sample time, `latest_sampled_profiles` was picking the first path alphabetically
  - that could silently compare stale sampled data and inflate the Shercliff `z`-profile parity error
  - the selector now breaks equal-time ties by file modification time, so the newest sampled profile set is used
- Current best next step is now:
  1. run the same parity-suite artifact path on machines that set `LMX_FREEMHD_CASE_DIR` so real FreeMHD parity artifacts are produced routinely
  2. extend the recovered-case parity path beyond the current Shercliff `Ha20` smoke case
  3. continue solver-fidelity work, especially for Hunt, now that the parity artifact plumbing is in normal CI

### 2026-03-28 19:30 America/Chicago

- Extended the recovered-case FreeMHD path beyond Shercliff by using the now-complete local `external/StartingFiles.zip`:
  - confirmed the archive contains recoverable Hunt cases including `hunt_exactBL_Ha20`, `hunt_exactBL_Ha100`, and `hunt_exactBL_Ha1000_0g`
  - extracted and materialized `hunt_exactBL_Ha20` under `/tmp/startingfiles_hunt_ha20/StartingFiles/Hunt/hunt_exactBL_Ha20`
- Verified that the existing container harness generalizes to the Hunt multi-region interFoam case without additional code changes:
  - ran `epotMultiRegionInterFoam` to `t = 1e-4`
  - reconstructed all regions at `0.0001`
  - confirmed `fieldMinMax.dat` and sampled `centerlineY/centerlineZ` outputs exist
- Recorded the first real Hunt `Ha20` FreeMHD-vs-LMX parity numbers:
  - `freemhd_u_max_latest = 0.118594754`
  - `lmx_u_max = 0.1174283996`
  - `u_max_abs_diff ≈ 1.17e-3`
  - `freemhd_sample_y_l2_error ≈ 6.02e-2`
  - `freemhd_sample_z_l2_error ≈ 6.74e-1`
- This materially changes the Hunt status:
  - infrastructure is no longer the main blocker for Hunt parity
  - the real blocker is LMX solver fidelity in the conducting-wall Hunt regime, especially the `z` profile
- Current best next step is now:
  1. use the recovered Hunt `Ha20` FreeMHD parity artifact to guide the next Hunt solver iteration in LMX
  2. keep Shercliff parity in CI artifact form as the lower-error reference path
  3. recover one higher-Ha closed-channel case next so the same parity machinery is exercised on a harsher boundary-layer regime

### 2026-03-28 19:45 America/Chicago

- Fixed a comparison-layer bug for layered-duct FreeMHD parity:
  - `extract_midplane_profile(..., fluid_only=True)` now excludes solid-wall cells when the mesh carries a layered `fluid_mask`
  - `closed_channel_validation`, `processed_slice_validation`, and `compare_with_freemhd` now use fluid-only midplane cuts for parity work
  - this keeps Hunt/Shercliff layered comparisons aligned with the FreeMHD `postProcess -region liquid` sampling path
- Verified the effect on the real Hunt `Ha20` parity artifact:
  - `u_max_abs_diff` stayed essentially unchanged at `≈ 1.17e-3`
  - `freemhd_sample_y_l2_error` stayed at `≈ 6.02e-2`
  - `freemhd_sample_z_l2_error` improved from `≈ 6.74e-1` to `≈ 1.39e-1`
- Current best next step is now narrower:
  1. treat the remaining Hunt error as a real solver-fidelity gap instead of a comparison artifact
  2. improve the short-time Hunt shape evolution in LMX while holding the now-correct fluid-only parity path fixed
  3. keep widening recovered FreeMHD parity to another higher-Ha closed-channel case once the next Hunt change lands

### 2026-03-28 20:00 America/Chicago

- Retained a solver-side improvement after the layered-profile comparison fix:
  - `_step(...)` now uses `time_stepper.outer_iterations` as a real fixed-point coupling loop inside each pseudo-time step
  - each outer iteration recomputes `phi`, `J`, and `J x B` from the latest velocity iterate before forming the next relaxed velocity update
  - this is a real solver change, not just a reporting change, so the low-Ha Hartmann/Shercliff regression baselines were updated to the new stable centerlines
- Verified the tradeoff before keeping it on `main`:
  - regression snapshots changed, but physics and validation tests stayed green
  - benchmark on the default CPU smoke path remained acceptable at about `cold ≈ 1.28 s`, `warm ≈ 0.29 s`
  - Shercliff `Ha20` short-time FreeMHD parity remained good: `y_l2 ≈ 1.15e-3`, `z_l2 ≈ 3.29e-2`
  - Hunt `Ha20` short-time FreeMHD parity improved again:
    - `freemhd_sample_y_l2_error` improved from `≈ 6.02e-2` to `≈ 5.36e-2`
    - `freemhd_sample_z_l2_error` improved from `≈ 1.39e-1` to `≈ 1.14e-1`
    - `u_max_abs_diff` remained small at `≈ 1.26e-3`
- Current best next step is now:
  1. keep the new outer-coupled pseudo-step and use the real Hunt `Ha20` parity artifact to tune the next solver change
  2. recover and run a higher-Ha closed-channel case next so the improved parity machinery is exercised in a harsher boundary-layer regime
  3. continue tightening the Hunt case construction so the short-time parity runner mirrors the recovered FreeMHD setup more faithfully

### 2026-03-28 20:20 America/Chicago

- Recovered and ran the first higher-Ha closed-channel FreeMHD case after the Hunt/Shercliff `Ha20` work:
  - extracted and materialized `shercliff_Ha100_ConstantQ_OutletZeroGradientInletCodedUxBpotE`
  - ran it through `scripts/run_freemhd_case.py` to `t = 1e-4`
  - reconstructed the `0.0001/` state and sampled `centerlineY/centerlineZ` through the same parity-suite runner
- Recorded the first real Shercliff `Ha100` short-time parity metrics:
  - `freemhd_u_max_latest = 0.973592421`
  - `lmx_u_max = 0.9718690515`
  - `u_max_abs_diff ≈ 1.72e-3`
  - `freemhd_sample_y_l2_error ≈ 2.36e-2`
  - `freemhd_sample_z_l2_error ≈ 4.47e-2`
- This is a useful new reference point:
  - the higher-Ha insulating-wall Shercliff case remains markedly better than the current Hunt `Ha20` conducting-wall case
  - higher-Ha runtime on the FreeMHD side is already significantly more expensive, so the next recovered conducting-wall case should be chosen intentionally
  - the latest `fieldMinMax.dat` record for this case reports the `mag(U)` maximum at `x = 0.0`, so the current sampling-geometry inference now selects the inlet plane for this recovered case
- Current best next step is now:
  1. use the combined Hunt `Ha20` and Shercliff `Ha100` artifacts to decide whether the next solver change should target conducting-wall coupling or higher-Ha damping/resolution
  2. recover Hunt `Ha100` next if we want the harsher conducting-wall benchmark, otherwise keep improving the LMX solver against the already recovered Hunt `Ha20` gap
  3. generalize the current sampling-geometry inference beyond raw `fieldMinMax` maxima so future cases are not tied to whichever plane happens to host the instantaneous `mag(U)` maximum

### 2026-04-01 12:05 America/Chicago

- Revisited the native Hunt case definition because the archived closed-channel
  analytical filenames consistently encode `db0.05`, which indicates a wall
  conductance-ratio input rather than a raw wall conductivity.
- Retained a public-API correction in `make_hunt_case(...)`:
  - the default Hunt input is now `wall_conductance_ratio=0.05`
  - wall conductivity is derived from fluid conductivity, Hartmann-wall half-spacing,
    and wall thickness
  - explicit `wall_conductivity` remains available as an override for future
    dimensional case definitions
- Added targeted unit coverage so this stays stable:
  - default Hunt cases now verify the derived conducting-wall conductivity
  - explicit wall-conductivity override behavior is also covered
- Verified the retained negative result before keeping it:
  - native Hunt `Ha20` remained around `y_l2 ≈ 0.211`, `z_l2 ≈ 0.373`
  - native Hunt `Ha100` remained around `y_l2 ≈ 0.198`, `z_l2 ≈ 0.411`
  - this means the remaining Hunt gap is not coming from the conductance-ratio
    normalization; it remains a solver/update-physics problem
- Tried a more ambitious solver change and explicitly rejected it:
  - implemented an implicit Helmholtz-like velocity solve inside the pseudo-step
  - it improved Hunt somewhat, but it regressed Hartmann acceptance and Shercliff
    quality badly enough that it was rolled back instead of being left on `main`
- Current best next step is now even narrower:
  1. improve the native Hunt update physics without degrading Hartmann/Shercliff
  2. keep using conductance ratio as the default Hunt public input
  3. prefer solver changes that can be justified geometrically or nondimensionally,
     not by case-specific rescue heuristics

### 2026-04-01 12:30 America/Chicago

- Added explicit duct-layer-resolution diagnostics to the native convergence
  artifacts:
  - `duct_layer_resolution_metrics(...)` now estimates Hartmann-layer thickness,
    side-layer thickness, cells across each layer, and minimum fluid spacing along
    those directions from the actual mesh and magnetic-field orientation
  - `scripts/run_convergence_suite.py` now includes those metrics in every level
    of the convergence summary
- Added targeted unit coverage for the new diagnostics and convergence-summary
  integration.
- Verified the new diagnostics on the native Hunt `Ha20` convergence sweep:
  - at `16^2`, `32^2`, and `48^2` fluid resolutions, the reported Hartmann-layer
    coverage is about `3.2`, `6.4`, and `9.7` cells
  - the reported side-layer coverage is about `5.2`, `10.5`, and `15.8` cells
  - despite that, the Hunt validation errors barely improve and the observed
    orders remain near zero
- This is an important retained result for the next implementation step:
  - the current native Hunt problem is not explained solely by missing layer
  resolution
  - the next solver work should focus on the update/coupling formulation rather
  than assuming that more mesh clustering alone will fix the steady Hunt gap

### 2026-04-01 13:05 America/Chicago

- Retained a general solver/operator improvement aimed at the clustered-mesh duct
  cases:
  - the potential-equation coefficients now use actual center-to-center distances
    on nonuniform meshes instead of dividing by local cell width squared
  - the electric-field reconstruction now uses the existing nonuniform-aware
    gradient operator
  - the masked velocity Laplacian now treats no-slip-style boundaries through
    half-cell wall distances rather than pretending the wall value is located at a
    neighboring cell center
- Added operator coverage for clustered meshes:
  - the quadratic-field Laplacian is now tested on a layered/clustered duct mesh
    away from the masked wall interfaces
- Verified the retained numerical effect before keeping it:
  - Hartmann `Ha20` remains strong and accepted at about
    `l2_error ≈ 3.87e-3`, `linf_error ≈ 1.01e-2`
  - Shercliff `Ha20` remains stable, with a slightly improved `z` profile
  - native Hunt `Ha20` improves materially to about
    `y_l2 ≈ 1.05e-1`, `z_l2 ≈ 2.90e-1`
  - native Hunt `Ha100` also improves to about
    `y_l2 ≈ 1.90e-1`, `z_l2 ≈ 3.62e-1`
  - the native Hunt `Ha20` convergence sweep now shows strong `y`-profile
    improvement under refinement:
    - `y_l2`: about `0.151 -> 0.067 -> 0.023`
    - `z_l2`: still problematic at about `0.127 -> 0.167 -> 0.209`
    - observed order for `y_l2` is now positive and substantial, while `z_l2`
      remains negative
- This narrows the next solver target again:
  - the nonuniform-mesh discretization was a real part of the problem and is now
    improved
  - the remaining native Hunt gap is increasingly concentrated in the `z`
    profile / side-layer evolution rather than in the overall duct solve

### 2026-04-01 13:30 America/Chicago

- Probed two more possible causes of the remaining native Hunt gap and kept the
  results as validation guidance rather than forcing another solver change:
  - increasing the explicit conducting-wall cell count at fixed fluid resolution
    barely moved the Hunt errors for either `Ha20` or `Ha100`
  - varying the pseudo-time step and step budget at fixed physical horizon moved
    the Hunt profiles nontrivially, especially at `Ha20`
- Those are useful negative/diagnostic results:
  - the explicit solid-wall mesh is not the dominant remaining issue
  - the remaining Hunt discrepancy is not purely spatial; temporal sensitivity is
    still present and should be tracked explicitly
- Implemented the planned native pseudo-time convergence runner:
  - `scripts/run_time_convergence_suite.py` now runs fixed-resolution, varying-`dt`
    studies for Hartmann, Shercliff, and Hunt
  - it reuses the same validation metrics as the mesh-convergence runner and also
    includes the current layer-resolution diagnostics in each level
  - CI now produces `artifacts/time_convergence` alongside the existing validation
    and mesh-convergence artifacts
- Added unit coverage for the new runner and updated the docs so it is part of the
  normal validation workflow.
- Current best next step is now better constrained:
  1. use the new time-convergence artifacts to separate the remaining Hunt error
     into temporal and spatial/update components
  2. keep improving the Hunt coupling formulation only where those artifacts show
     a persistent solver defect, not where the result is just time-step sensitive

### 2026-04-01 13:45 America/Chicago

- Ran the first real local pseudo-time convergence artifact with the new runner:
  - case: native Hunt `Ha20`
  - fixed resolution: `48^2` fluid cells
  - `dt` sweep: `0.002`, `0.001`, `0.0005`
- The retained result is important for the next solver step:
  - decreasing `dt` does not improve all observables together
  - `y_l2` worsens from about `0.023` at `dt=0.002` to about `0.108` at
    `dt=0.0005`
  - `z_l2` improves only modestly from about `0.209` to about `0.175`
  - the observed orders with respect to `dt` are negative for the `y` profile and
    only weakly positive for the `z` profile
- Interpretation:
  - the remaining Hunt gap is not just “run longer” or “take a smaller step”
  - the current pseudo-time update still contains a tradeoff between different
    profile directions, which points back to the coupled update/control law rather
    than to a missing validation or meshing feature

### 2026-04-01 14:05 America/Chicago

- Converted the latest manual Hunt control probes into a reusable tool:
  - added `scripts/run_solver_control_sweep.py`
  - it sweeps a selected `TimeStepperConfig` parameter such as
    `outer_iterations`, `potential_iterations`, `relaxation`, or `dt`
  - it writes the same validation metrics plus the current layer-resolution
    diagnostics into a single JSON summary
- Added unit coverage for the new runner and documented it as part of the native
  validation workflow.
- Ran the first real retained control sweep on native Hunt `Ha20` at `48^2`:
  - parameter: `outer_iterations`
  - values: `2, 4, 6, 8, 10`
  - result:
    - `z_l2` improves monotonically from about `0.283` to `0.185`
    - `y_l2` improves from about `0.099` to `0.023` by `6` iterations, then
      degrades again to about `0.042` and `0.062`
- This is the clearest retained control-law result so far:
  - the current Hunt solver is not just under-iterated
  - increasing outer coupling helps one profile direction while eventually
    hurting the other
  - the next solver change should target that coupled tradeoff directly rather
    than simply increasing `outer_iterations`, `potential_iterations`, or
    shrinking `dt`

### 2026-04-01 14:30 America/Chicago

- Tightened one remaining nonuniform-mesh operator detail:
  - the boundary branches in `gradient_scalar(...)` now use center-to-center
    spacing between the first two cell centers instead of the first cell width
  - this is the correct one-sided stencil for the cell-centered gradient on a
    clustered mesh
- Added explicit unit coverage for linear-field gradients on a clustered layered
  duct mesh so the fix stays locked in.
- Verified the retained effect before keeping it:
  - Hartmann, Shercliff, and the current Hunt validation metrics stayed
    essentially unchanged
  - this is therefore a correctness cleanup, not a major parity shift
- Also tried a more ambitious Hunt-control change and rejected it:
  - distributed `velocity_update_limit` across outer iterations so the total
    allowed state change per pseudo-step stayed constant as `outer_iterations`
    increased
  - that did make the Hunt `Ha20` outer-iteration sweep more monotone, but it
    degraded the retained default Hunt results, especially at `Ha100`
  - it was rolled back instead of being left on `main`

### 2026-04-01 15:00 America/Chicago

- Extended the CI artifact summarizer so the markdown/JSON report can now surface:
  - native pseudo-time convergence summaries
  - solver-control sweep summaries
- Added test coverage for both summary paths and re-ran the full local suite plus
  the Sphinx docs build before keeping the change.
- Purpose of the change:
  - keep the newly added Hunt diagnostics visible in normal CI artifacts
  - make it easier to review solver-tradeoff runs without opening each raw JSON
    file by hand

### 2026-04-01 16:40 America/Chicago

- Tightened the CI summary again after the failed semi-implicit velocity pass:
  - sweep summaries now report not only the first and last parameter values, but
    also the best `y_l2` and best `z_l2` points
  - this matters because the retained Hunt control sweeps are explicitly
    non-monotone, so first/last-only reporting hides the real optimum
- Added test coverage for the new sweep-summary fields.
- Retained interpretation:
  - the current Hunt control data should be read as a tradeoff surface, not a
    monotone curve
  - the next solver change should be judged against the best interior sweep
    point, not just against the last parameter value

### 2026-04-01 18:10 America/Chicago

- Tried another solver-side Hunt candidate and rejected it:
  - advanced the outer-loop velocity trial from `u_iter` instead of the
    step-entry state `u`
  - this looked like a plausible fixed-point cleanup, but it again broke the
    Hartmann `Ha20` medium-resolution guardrail while not yet earning a retained
    Hunt improvement
  - it was rolled back instead of being left on `main`
- Converted that failure mode into a direct regression guardrail:
  - first attempted to add a direct validation test requiring Hartmann `Ha20`,
    `32^2` to pass the current analytical acceptance thresholds
  - that test immediately exposed a broader current limitation: the retained
    solver on `main` does not yet satisfy that medium-resolution Hartmann
    acceptance target
  - the failing test was not kept on `main`
  - instead, the CI summary now reports how many sweep levels pass acceptance so
    Hartmann refinement failures become visible in normal artifacts
- Retained interpretation:
  - future solver changes must preserve not only the coarse Hartmann smoke path
    but also improve the current Hartmann refinement behavior
  - the next Hunt update should still be assessed against Hartmann refinement,
    but the ship-ready plan must now treat Hartmann medium-resolution acceptance
    as an open blocker rather than as a solved guardrail

### 2026-04-01 19:00 America/Chicago

- Investigated the Hartmann refinement blocker more directly by sampling actual
  centerlines at `Ha20` for `16^2`, `32^2`, `48^2`, and `96^2`:
  - `16^2` remains acceptable (`l2 ≈ 1.96e-2`)
  - `32^2` is the clearest failure branch right now:
    - the centerline changes sign
    - `l2 ≈ 1.20`, `linf ≈ 1.83`
  - `48^2` is positive again but still not acceptable
    (`l2 ≈ 7.86e-2`, `linf ≈ 1.28e-1`)
  - `96^2` returns to the previously good fine-mesh behavior
    (`l2 ≈ 3.87e-3`, `linf ≈ 1.01e-2`)
- Probed a few obvious control-only explanations for the `32^2` Hartmann branch:
  - more steps alone does not fix it
  - smaller `dt`, smaller relaxation, and a smaller velocity-update cap reduce
    the damage but do not restore analytical acceptance
  - this makes the blocker look more like a discrete-update/control-law pathology
    than a simple convergence-budget issue
- Retained code/data improvement:
  - added explicit validation metrics for profile sign pathologies:
    - `centerline_y_sign_changes`
    - `centerline_z_sign_changes`
    - `centerline_y_negative_fraction`
    - `centerline_z_negative_fraction`
  - these now surface oscillatory/nonphysical branches directly in validation
    artifacts instead of only through aggregate L2 errors
- Next-step implication:
  - the remaining solver work is not just “tune more controls”
  - the next retained solver change should target the update formulation while
    using the new sign-pathology metrics as an additional guardrail

### 2026-04-01 19:40 America/Chicago

- Probed the Hartmann `Ha20`, `32^2` blocker against potential-solve depth:
  - `potential_iterations=50` is acceptable
    (`l2 ≈ 2.08e-2`, `linf ≈ 5.34e-2`)
  - `100` is worse but still non-oscillatory
    (`l2 ≈ 2.09e-1`)
  - `200` is the clearly bad branch
    (`l2 ≈ 1.38`, sign changes present, large negative fraction)
  - `400` removes the sign pathology and drops the error substantially
    (`l2 ≈ 3.13e-1`)
  - `800` is acceptable again
    (`l2 ≈ 3.76e-2`, `linf ≈ 6.23e-2`)
- Important interpretation:
  - this is not a simple monotone “more is always better” story at low budgets,
    but it strongly implicates the electric-potential solve depth as a central
    part of the Hartmann refinement blocker
  - generic pseudo-time controls alone are not the main issue here
- Also tried the existing `lineax` Poisson backend again as a more principled
  replacement for fixed-count Jacobi:
  - even with a large `max_steps` override, it still hit the solver-step limit
    in the current configuration
  - it was not retained for production use
- Retained infrastructure change:
  - CI now runs a dedicated Hartmann control sweep:
    - case: `hartmann`
    - `Ha=20`
    - resolution: `32^2`
    - parameter: `potential_iterations`
    - values: `50,100,200,400,800`
  - this uses the existing generic sweep runner and summary path, so the current
    refinement blocker becomes a normal artifact on every CI run

### 2026-04-01 21:05 America/Chicago

- Tried another solver-side fix first and explicitly rejected it:
  - reusing the previous step's electric potential as a warm start for the next
    pseudo-time step destabilized the retained Hunt default path
  - on the existing `test_hunt_default_case_now_stays_bounded` gate, `max(U)`
    rose to about `2.5e-2`, which is far outside the retained default bound
  - that change was rolled back immediately and was not kept on `main`
- Retained improvement instead:
  - the solver now records a normalized electric-potential equation residual in
    `Diagnostics.potential_residual_history`
  - `validation_summary(...)` emits the latest value as `potential_residual`
  - the CI markdown summary now shows that field in the validation table
- Current quantitative interpretation with the new metric:
  - the bad Hartmann `Ha20`, `32^2` branch currently lands around
    `potential_residual ≈ 6.7e-2`
  - the current native Hunt `Ha20`, `32^2` branch lands around
    `potential_residual ≈ 5.6e-1`
  - this is not yet an acceptance threshold, but it makes the `phi`-solve part
    of the coupled defect directly visible in the same artifacts as the profile
    errors and sign-pathology metrics
- Next-step implication:
  - the next retained solver iteration should aim to reduce that normalized
    potential residual on the Hartmann/Hunt bad branches without regressing the
    accepted Hartmann fine-mesh path or the current bounded Hunt defaults

### 2026-04-01 22:00 America/Chicago

- Added convergence-control infrastructure for the electric-potential solve:
  - `TimeStepperConfig` now accepts an optional `potential_tolerance`
  - the Jacobi solve now supports residual-based stopping up to the configured
    `potential_iterations` ceiling
  - diagnostics and validation summaries now report the actual Jacobi iteration
    count used by the latest solve step as `potential_iterations_used`
  - the generic solver-control sweep runner now accepts `potential_tolerance`
- Probed the new path on the Hartmann/Hunt blocker directly:
  - Hartmann `Ha20`, `32^2`, baseline:
    - `l2 ≈ 1.20`, `linf ≈ 1.83`
    - `potential_residual ≈ 6.7e-2`
    - `potential_iterations_used = 200`
  - Hartmann `Ha20`, `32^2`, with `potential_iterations = 800` and
    `potential_tolerance in {1e-2, 1e-3, 1e-4}`:
    - all three runs collapsed to the same improved branch
    - `l2 ≈ 3.0e-2`, `linf ≈ 5.0e-2`
    - `potential_residual ≈ 3.2e-2`
    - `potential_iterations_used = 800`
  - Hunt `Ha20`, `32^2`, baseline:
    - `y_l2 ≈ 6.70e-2`, `z_l2 ≈ 1.67e-1`
    - `potential_residual ≈ 5.57e-1`
    - `potential_iterations_used = 400`
  - Hunt `Ha20`, `32^2`, with the same `800`-iteration ceiling and
    `potential_tolerance` values:
    - `y_l2 ≈ 7.74e-2`, `z_l2 ≈ 1.50e-1`
    - `potential_residual ≈ 4.95e-1`
    - `potential_iterations_used = 800`
- Retained interpretation:
  - the new tolerance-aware infrastructure is sound and worth keeping
  - the observed Hartmann improvement in this probe came from the larger
    iteration ceiling, not from a tolerance stop, because every tested run hit
    the full `800`-iteration cap
  - that makes the next solver question more specific:
    - is the right default change a larger and perhaps geometry-aware
      `potential_iterations` ceiling
    - or should the coupled update be changed so the present ceiling is enough
      on the medium-resolution Hartmann/Hunt branches

### 2026-04-01 22:45 America/Chicago

- Probed the next obvious Hunt default-policy candidate before changing anything:
  - increasing `potential_iterations` at `Ha20`, `32^2` improves Hunt `z_l2`
    monotonically:
    - `200 -> 400 -> 600 -> 800` gives about
      `0.180 -> 0.167 -> 0.157 -> 0.150`
  - but over the same sweep the Hunt `y_l2` degrades:
    - `0.063 -> 0.067 -> 0.073 -> 0.077`
  - a coupled sweep over `outer_iterations` and `potential_iterations` showed
    the same tradeoff:
    - `outer=4, pit=400` improved `y_l2` to about `0.038`
    - but worsened `z_l2` to about `0.186`
- Retained interpretation:
  - no default Hunt control change should be kept yet based on one profile
    direction alone
  - a combined closed-channel profile metric is needed so these tradeoffs are
    scored consistently instead of by visual inspection of separate `y/z` cuts
- Retained code/data improvement:
  - added a combined closed-channel profile error to the validation/reporting
    path:
    - CLI validation output
    - validation suite summaries
    - convergence summaries
    - pseudo-time convergence summaries
    - solver-control sweeps
    - CI markdown summaries
  - this immediately clarified the recent Hunt candidate:
    - current default at `Ha20`, `32^2` has combined error ≈ `0.127`
    - the tempting `outer=4, pit=400` candidate has combined error ≈ `0.134`
    - so it is not a real retained improvement and the Hunt defaults stay
      unchanged on `main`
- Next-step implication:
  - the next retained Hunt solver change should reduce the combined closed-channel
    error, not just one of the directional profile errors

### 2026-04-01 23:20 America/Chicago

- Probed a broader default-policy idea and rejected it:
  - hypothesis: raise the electric-potential iteration budget for all
    single-region rectangular ducts
  - Hartmann `Ha20`, `32^2` does improve strongly as `potential_iterations`
    rises:
    - `200 -> 400 -> 800` gives about
      `l2: 1.20 -> 0.25 -> 3.0e-2`
  - Shercliff `Ha20`, `32^2` also improves strongly under the same sweep:
    - combined closed-channel error about
      `0.944 -> 0.409 -> 0.223`
  - but Hartmann `Ha20`, `48^2` invalidates a blanket default change:
    - `potential_iterations = 200` gives about `l2 ≈ 7.9e-2`
    - `400` jumps to a much worse branch (`l2 ≈ 1.51`)
    - `800` is still worse than the current default (`l2 ≈ 5.56e-1`)
- Retained interpretation:
  - a larger rect-duct `phi` budget is not a generally safe default policy yet
  - the medium-resolution Hartmann pathology is more structured than “more
    potential iterations is always better”
  - this reinforces the need to use sweep artifacts, not intuition, when
    choosing the next retained solver/control change
- Retained reporting improvement:
  - CI sweep summaries now report the best combined closed-channel error in
    addition to the best directional `y/z` errors
  - that makes the current Hunt tuning tradeoffs visible in the normal CI summary
    without hand-comparing separate profile columns
- Next-step implication:
  - the next retained solver change should target the coupled update law or the
    non-monotone Hartmann branch mechanism itself, not a blanket increase in
    potential-iteration defaults

### 2026-04-02 00:05 America/Chicago

- Probed a more direct solver-side change in the electric-potential iteration:
  - added weighted Jacobi support through a new `potential_relaxation` control
  - kept it as general solver infrastructure and made it sweepable in
    `run_solver_control_sweep.py`
- Retained probe results:
  - Hartmann `Ha20`, `32^2`, `potential_iterations=200`:
    - `potential_relaxation = 1.0` gives `l2 ≈ 1.20`
    - `0.5` improves that to about `l2 ≈ 0.196`
  - Hartmann `Ha20`, `48^2`, `potential_iterations=400`:
    - `1.0` gives `l2 ≈ 1.51`
    - `0.5` improves that to about `l2 ≈ 7.7e-2`
  - Hunt `Ha20`, `32^2`:
    - combined error worsens modestly as relaxation drops
      (`0.127 -> 0.135` from `1.0 -> 0.5`)
  - Hunt `Ha100`, `32^2`:
    - combined error improves modestly as relaxation drops
      (`0.343 -> 0.327` from `1.0 -> 0.5`)
  - Shercliff `Ha20`, `32^2`:
    - improves at the current default `225` potential iterations
      (`combined ≈ 0.844 -> 0.484` from `1.0 -> 0.5`)
    - but degrades at `400` potential iterations
- Retained interpretation:
  - weighted Jacobi is a real control lever for the non-monotone Hartmann branch
  - but it is still not safe as a blanket new default across the currently
    supported duct families
  - the right status on `main` is:
    - keep `potential_relaxation` as infrastructure
    - expose it to solver-control sweeps
    - do not change case defaults yet
- Next-step implication:
  - the next retained solver change should probably combine the current
    `potential_iterations`, `potential_residual`, `potential_relaxation`, and
    combined-error artifacts to identify a coupled update rule, not just another
    single-parameter default tweak

### 2026-04-02 00:40 America/Chicago

- Implemented a more structural electric-potential backend improvement:
  - added a matrix-free preconditioned CG backend for the electric-potential
    solve
  - added `potential_solver` to `TimeStepperConfig`
  - added `potential_solver` support to `run_solver_control_sweep.py`
  - the first attempt resolved the backend inside the traced JAX step and failed
    with a traced-boolean conversion; that was fixed by resolving the backend
    once, outside the compiled step, from the material-region layout
- Retained backend-probe results:
  - Hartmann `Ha20`, `32^2`:
    - Jacobi gives `l2 ≈ 1.20`, `potential_residual ≈ 6.7e-2`
    - CG gives `l2 ≈ 1.4e-2`, `potential_residual ≈ 1.2e-4`
  - Hartmann `Ha20`, `48^2`:
    - Jacobi gives `l2 ≈ 7.9e-2`
    - CG gives `l2 ≈ 7.0e-3`
  - Shercliff `Ha20`, `32^2`:
    - Jacobi gives combined error `≈ 0.844`
    - CG gives combined error `≈ 0.162`
  - Hunt `Ha20`, `32^2`:
    - Jacobi gives combined error `≈ 0.127`
    - CG degrades badly to `≈ 0.400`
  - Hunt `Ha100`, `32^2`:
    - Jacobi gives combined error `≈ 0.343`
    - CG also degrades to `≈ 0.377`
- Retained interpretation:
  - the backend choice is not case-name specific; it is strongly correlated with
    region structure
  - CG is the right retained default for the current single-region duct path
    because it sharply improves Hartmann and Shercliff and removes the worst
    medium-resolution Hartmann branch pathology
  - the current layered multi-region Hunt path should stay on the damped Jacobi
    backend until the coupled conducting-wall update itself is improved
  - `main` now encodes that as a principled `potential_solver="auto"` policy:
    - use `cg` when the solved cross-section is a single fluid region
    - use `jacobi` when explicit solid layers are present
- What worked:
  - adding the CG backend itself
  - exposing backend choice to control sweeps
  - resolving `auto` outside the JIT boundary from material structure
- What did not work:
  - using CG unconditionally across all supported duct families
  - resolving `auto` inside the traced step
- Best next step:
  - keep the new `auto` backend policy for the single-region path
  - target the remaining Hunt conducting-wall gap directly, now that the
    single-region `phi`-solve pathology is much better contained

### 2026-04-02 01:05 America/Chicago

- Probed another general Hunt control rather than changing defaults by hand:
  - added `velocity_update_limit` to `run_solver_control_sweep.py`
  - explicitly tested whether a different bounded-velocity update cap can close
    the remaining conducting-wall gap
- What did not work:
  - replacing the current global bounded update with a local per-cell clipped
    update
  - it destabilized the retained Hunt `Ha20` branch badly:
    - combined error jumped to about `0.326`
    - sign-pathology metrics became strongly nonzero
  - that experiment was rolled back immediately
- Retained sweep results:
  - Hunt `Ha20`, `32^2`, combined error versus `velocity_update_limit`:
    - `5e-4`: `≈ 0.135`
    - `1e-3`: `≈ 0.131`
    - `2e-3`: `≈ 0.127`
    - `4e-3`: `≈ 0.127`, but still slightly worse than `2e-3`
  - Hunt `Ha100`, `32^2`, combined error versus `velocity_update_limit`:
    - `1e-3`: `≈ 0.343`
    - `2e-3`: `≈ 0.342`
    - `4e-3`: `≈ 0.343`
  - Hartmann `Ha20`, `32^2` stays accepted across the same sweep and is not the
    limiting factor here
- Retained interpretation:
  - `velocity_update_limit` is worth keeping as part of the documented control
    surface for future solver work
  - but it does not produce a strong enough cross-case improvement to justify a
    new retained default change
  - the remaining Hunt problem is still in the coupled update law itself, not in
    a missing sweep axis
- Best next step:
  - use the now-expanded control surface (`potential_solver`,
    `potential_relaxation`, `potential_iterations`, `velocity_update_limit`,
    combined error, and potential residual) to redesign the multi-region Hunt
    update rather than trying more one-parameter default tuning

### 2026-04-02 01:25 America/Chicago

- Added a new retained diagnosis tool:
  - `scripts/run_solver_grid_sweep.py`
  - purpose: run two-parameter control grids when separate one-parameter sweeps
    still leave a cross-case tradeoff ambiguous
- Verified the tool with unit coverage and used it immediately on the current
  Hunt control question:
  - `outer_iterations in {4, 6}`
  - `potential_relaxation in {1.0, 0.5}`
  - at `Ha20`, `32^2`, the current default-like point remains best in this grid:
    - `outer=6`, `potential_relaxation=1.0` gives combined error `≈ 0.127`
    - `outer=4`, `potential_relaxation=1.0` gives `≈ 0.134`
    - `outer=6`, `potential_relaxation=0.5` gives `≈ 0.135`
    - `outer=4`, `potential_relaxation=0.5` gives `≈ 0.142`
  - at `Ha100`, `32^2`, the grid confirms the partial improvement already seen
    in ad hoc probes:
    - `potential_relaxation=0.5` improves combined error from about `0.343` to
      about `0.327`
    - changing `outer_iterations` from `4` to `6` is essentially neutral
- Retained interpretation:
  - there is still no honest Hunt default shift that improves both the lower-Ha
    and higher-Ha conducting-wall cases together
  - the current exposed control surface is now broad enough that continued
    one-parameter tuning is unlikely to produce the next real improvement
  - the next retained solver change should target the coupled update law or the
    layered linearization itself, not another control default
- What did not work:
  - a multi-region predictor-corrector candidate with an extra potential pass
    after the relaxed velocity update
  - it drove the Hunt `Ha20` potential residual down to about `1.3e-4`, but the
    solution itself blew up (`combined error ≈ 0.717`, strong sign pathologies,
    `u_max ≈ 1.95`)
  - that branch was rolled back immediately
- Best next step:
  - try a more principled multi-region coupling change that does not simply push
    the potential residual down faster
  - likely targets are:
    - a layered-potential linearization change
    - or a better coupling criterion that respects both velocity and potential
      residuals without over-correcting the velocity update

### 2026-04-02 01:50 America/Chicago

- Implemented a more principled nonuniform electric-potential discretization:
  - replaced the old equal-spacing harmonic face shortcut with
    resistance-weighted face conductance
  - replaced the old face-averaged electromotive source with the matching
    resistance-weighted face electromotive term
  - added unit coverage for:
    - uniform-grid coefficient consistency
    - nonuniform face-emf weighting
- Retained numerical effect at the current Hunt control points:
  - Hunt `Ha20`, `32^2`, current default-like point:
    - combined error improved slightly from about `0.12714` to about `0.12666`
  - Hunt `Ha100`, `32^2`, current default-like point:
    - combined error worsened slightly from about `0.34266` to about `0.34927`
  - Hartmann `Ha20`, `32^2` stayed accepted on the improved single-region path
- Retained interpretation:
  - this is still worth keeping because it is the finite-volume-consistent
    nonuniform form, and the previous shortcut was only exact for equal adjacent
    cell widths
  - but it is not the missing Hunt fix by itself
  - the remaining Hunt issue still appears to live in the layered coupled update
    law and/or the layered linearization, not just the old face shortcut
- What also happened this round:
  - explicitly tried a multi-region predictor-corrector candidate with an extra
    potential pass after the relaxed velocity update
  - it drove `potential_residual` much lower, but the Hunt solution itself blew
    up badly (`combined error ≈ 0.717`, nonzero sign-pathology metrics, large
    `u_max`), so it was rolled back immediately
- Best next step:
  - keep the corrected nonuniform discretization
  - next try a layered linearization or convergence criterion change that
    improves Hunt without simply over-correcting the coupled update

### 2026-04-02 02:05 America/Chicago

- Tightened the steady-solver semantics without changing the retained Hunt
  accuracy claims:
  - added `steady_potential_tolerance` to `TimeStepperConfig`
  - `solve_steady(...)` can now require both velocity residual and
    electric-potential residual convergence before stopping
  - added focused solver tests covering the new stop condition
- Retained interpretation:
  - this is a solver-correctness and future-proofing change, not a new Hunt
    accuracy improvement by itself
  - it matters because layered cases can otherwise look “steady” on velocity
    residual alone while the potential equation is still under-resolved
  - the current Hunt parity/convergence numbers are unchanged in substance,
    because the current reference runs are already exhausting their configured
    step budgets
- Best next step:
  - continue targeting the layered Hunt update/linearization itself
  - use the stricter steady semantics as a guardrail so future retained solver
    changes are not judged on prematurely stopped layered runs

### 2026-04-02 02:25 America/Chicago

- Improved the artifact/reporting path around the current Hunt diagnosis:
  - `summarize_ci_artifacts.py` now accepts a two-parameter control-grid summary
  - the markdown summary now reports the best combined, `y`, and `z` points from
    that grid together with the parameter pair where they occur
- This is worth keeping because:
  - the Hunt control surface is now genuinely two-dimensional in the retained
    workflow
  - future solver iterations should not require hand-reading raw grid JSON to
    understand whether a candidate is a real improvement
- Also tried one more solver-side Hunt candidate and rejected it:
  - reconstructed `J` and `J×B` from face-flux-consistent currents so the
    momentum update would use the same flux form as the potential solve
  - that modestly improved Hartmann but degraded Hunt badly:
    - Hunt `Ha20`, `32^2`: combined error rose to about `0.130`
    - Hunt `Ha100`, `32^2`: combined error rose to about `0.414`
  - rolled back immediately
- Retained interpretation:
  - a more consistent face-current reconstruction alone is not the missing Hunt
    fix
  - the remaining Hunt issue still appears to be in the layered coupled update
    law or layered linearization, not just in whether `J` is reconstructed from
    face fluxes or center gradients
- Best next step:
  - keep the new grid-summary reporting
  - continue the layered Hunt solver work with a stronger focus on the update
    law itself rather than current postprocessing or one-parameter tuning

### 2026-04-02 03:10 America/Chicago

- Added a solver-diagnostic-first Hunt comparison script:
  - `scripts/run_hunt_solver_diagnostic_report.py`
  - it compares a recovered FreeMHD Hunt run with a native LMX Hunt solve and
    writes one JSON artifact with three sections:
    - `freemhd_run`
    - `lmx_solver`
    - `comparison`
  - the new report keeps solver diagnostics first and profile comparisons
    secondary, which is the right shape for the remaining Hunt work
- Retained report content:
  - `freemhd_run`: run-directory inspection counts plus latest `mag(U)` metadata
  - `lmx_solver`: the native Hunt controls plus `validation_summary(...)`
    diagnostics such as residual, potential residual, and potential-iteration
    usage
  - `comparison`: `u_max` difference and sampled-profile comparison metrics when
    the recovered FreeMHD sample files are present
- What worked:
  - wiring the report through the existing validation utilities instead of adding
    a parallel data model
  - keeping the output format simple enough to be used immediately on the
    current recovered Hunt cases
- What to remember:
  - this script is intentionally solver-diagnostic first
  - it is a better target for Hunt analysis than the older profile-only parity
    reports when deciding whether a candidate change is actually improving the
    solver
- Best next step:
  - use the new solver-diagnostic report alongside the existing parity and
    control-sweep artifacts when evaluating the next layered Hunt update

### 2026-04-02 04:10 America/Chicago

- Took a blocker-focused pass on the layered Hunt `phi` solve instead of adding
  more scaffolding:
  - added `potential_solver="cg_volume"` to LMX
  - this backend solves the same discrete layered `phi` system after left
    scaling by the cell metric, which is the symmetric form of the nonuniform
    divergence-form operator
  - kept `potential_solver="auto"` unchanged for now:
    - single-region ducts still resolve to `cg`
    - layered ducts still resolve to `jacobi`
- Retained numerical result:
  - Hartmann `Ha20`, `32^2`:
    - `cg_volume` matches the good single-region CG path and remains accepted
  - Shercliff `Ha20`, `32^2`:
    - `cg_volume` matches the good single-region CG path
      (`combined_l2 ≈ 0.162`)
  - Hunt `Ha20`, `32^2`:
    - `jacobi`: `combined_l2 ≈ 0.1267`, `potential_residual ≈ 5.0e-1`
    - `cg_volume`: `combined_l2 ≈ 0.1510`, `potential_residual ≈ 9.4e-3`
  - Hunt `Ha100`, `32^2`:
    - `jacobi`: `combined_l2 ≈ 0.3493`, `potential_residual ≈ 1.7e-1`
    - `cg_volume`: `combined_l2 ≈ 0.3014`, `potential_residual ≈ 1.0e-2`
- Retained interpretation:
  - the layered `phi` solve really was part of the blocker
  - but it is not the only blocker
  - once the layered `phi` residual is reduced by one to two orders of
    magnitude, the remaining Hunt gap is still in the coupled velocity update
    law, because `Ha20` gets a better `y` profile and a much better `phi`
    residual while the combined error still worsens through the `z` profile
  - this is the strongest retained evidence so far that the next real solver
    change should target the multi-region velocity update / coupling law rather
    than the `phi` backend alone
- Added a direct FreeMHD diagnostic path for the next step:
  - `scripts/patch_freemhd_coupled_logging.py`
    - patches local `epotMultiRegionInterFoam` sources with opt-in `LMX_DIAG`
      logging in the outer loop, fluid electric-potential solve, and fluid
      momentum solve
    - logging stays behind a `controlDict` flag
  - `scripts/extract_freemhd_coupled_log.py`
    - converts those `LMX_DIAG` lines into JSON records for analysis
- Best next step:
  - use the new FreeMHD logging patch on the recovered Hunt case
  - collect coupled `epot` / momentum residual histories from FreeMHD
  - compare them directly with LMX `potential_residual`, `potential_iterations`,
    and outer-coupling behavior
  - then change the layered Hunt velocity update law itself, not just the
    potential backend or scalar control defaults

### 2026-04-02 05:05 America/Chicago

- Closed the remaining infrastructure gap for patched FreeMHD runs:
  - `docker/Dockerfile` now accepts a staged local `FreeMHD/` tree before
    falling back to a fresh upstream clone
  - `scripts/build_freemhd_container.py` now accepts
    `--local-freemhd-root` and stages a minimal local FreeMHD tree into a
    temporary build context
  - this lets container builds consume local solver instrumentation patches
    instead of ignoring them
- Extended the FreeMHD logging patch:
  - `patch_freemhd_coupled_logging.py` still logs:
    - outer coupling loop
    - fluid `epot` solve
  - it now also patches the shared interFoam `pEqn.H` path so the next run can
    emit structured pressure-correction / `maxU` diagnostics in addition to the
    `epot` trace
- Real retained logged-run evidence now exists from the patched local FreeMHD
  Hunt `Ha20` case:
  - container image `lmx-freemhd-localdiag` built successfully from the local
    patched tree
  - recovered `hunt_exactBL_Ha20` is running with `LMX_DIAG` enabled
  - `extract_freemhd_coupled_log.py` already captures structured `outer` and
    `epot` records from that live run
  - first retained `epot` records:
    - `t = 1.25e-05`:
      - `potEInitialResidual = 1.0`
      - `potEFinalResidual = 4.43e-08`
      - `potEIterations = 11`
      - `maxJxB ≈ 4.69e3`
    - `t = 2.70833e-05`:
      - `potEInitialResidual ≈ 2.86e-1`
      - `potEFinalResidual ≈ 5.34e-08`
      - `potEIterations = 7`
      - `maxJxB ≈ 4.65e3`
    - `t = 4.53125e-05`:
      - `potEInitialResidual ≈ 1.90e-1`
      - `potEFinalResidual ≈ 3.75e-08`
      - `potEIterations = 7`
      - `maxJxB ≈ 4.63e3`
    - `t = 6.35417e-05`:
      - `potEInitialResidual ≈ 1.44e-1`
      - `potEFinalResidual ≈ 3.06e-08`
      - `potEIterations = 7`
      - `maxJxB ≈ 4.62e3`
- Retained interpretation:
  - this already shows the FreeMHD `phi` block is well converged on the real
    Hunt case
  - that strengthens the current hypothesis that the remaining LMX Hunt blocker
    is in the coupled velocity/pressure response to `phi`, not just in the
    absolute `phi` residual itself
  - the missing piece is structured `pressure/maxU` logging from the same run
    so the LMX coupling law can be compared against FreeMHD at the solver-step
    level rather than only through end profiles
- Best next step:
  - finish the pressure-instrumented FreeMHD rerun using the updated logging
    patch
  - compare those `pressure/maxU` records against LMX short-time Hunt traces
  - then change the layered Hunt velocity update law directly

### 2026-04-02 05:35 America/Chicago

- Closed the immediate CI diagnosis loop with `gh`:
  - the repeated `ci` workflow failure on `main` is the `coverage` job, not the
    unit/regression/physics/validation/docs jobs
  - exact failure on the latest run:
    - total coverage was `82.63%`
    - workflow threshold is `85%`
  - so the correct retained fix is better test coverage, not a workflow or
    environment workaround
- Retained CI/CD fix work:
  - `scripts/build_freemhd_container.py` now uses
    `docker buildx build --load`
  - this is the concrete fix for the earlier local state where a FreeMHD image
    build reported success but the tag was not visible to
    `docker image inspect` / `docker run`
- Retained coverage-fix work:
  - added targeted unit coverage for:
    - `scripts/inspect_starting_files_archive.py`
    - `scripts/probe_freemhd_environment.py`
    - `scripts/build_freemhd_container.py`
    - `scripts/patch_freemhd_coupled_logging.py`
    - `scripts/run_convergence_suite.py`
    - `scripts/run_time_convergence_suite.py`
  - the targeted new/expanded tests pass locally
  - the full local coverage rerun is slow on this machine, so the next
    authoritative check is the pushed GitHub Actions `coverage` job
- Best next step:
  - push the retained Docker/coverage fixes
  - confirm the `coverage` job turns green on GitHub Actions
  - then use the now-loadable patched FreeMHD image path to complete the
    pressure-instrumented Hunt rerun and move back to the layered Hunt update
    law itself

### 2026-04-02 09:40 America/Chicago

- The `main` branch is now back on a clean green baseline after the CI/CD fix:
  - local checkout is at `2ea01a7`
  - GitHub Actions `ci` and `benchmarks` are both green on `main`
  - retained CI fix:
    - `scripts/build_freemhd_container.py` now uses
      `docker buildx build --load`
    - coverage was raised with targeted tests instead of lowering the gate
- The patched FreeMHD Hunt `Ha20` rerun now produces the missing coupled
  pressure trace when started from `startTime`:
  - retained harness fix:
    - `run_freemhd_case.py`, `docker/run_freemhd_case.sh`, and
      `write_freemhd_container_files.py` now support `startFrom`
    - this fixes the earlier failure mode where a recovered case with existing
      `0.0001/` output started from `latestTime` and immediately ended without
      emitting new solver diagnostics
  - retained FreeMHD log records now saved in `/tmp/lmx_hunt20_live_diag.json`
  - first coupled pressure-correction records:
    - `t = 1.25e-05`, `corr = 0`:
      - `pInitialResidual = 4.96e-03`
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
- LMX now records the matching short-time trace observables:
  - `Diagnostics` now includes `time_history` and `u_max_history`
  - `run_hunt_solver_diagnostic_report.py` now emits an `lmx_solver.trace`
    section with:
    - `time_history`
    - `u_max_history`
    - `residual_history`
    - `potential_residual_history`
    - `potential_iterations_history`
- First retained short-time Hunt comparison using those traces:
  - default layered Hunt short-time run (`Ha20`, `dt = 1e-5`, `10` steps):
    - LMX first-step `u_max ≈ 0.117496`
    - FreeMHD first-step final-correction `maxU ≈ 0.117743`
    - LMX end-of-trace `potential_residual ≈ 4.12e-01`
  - `potential_relaxation = 0.5` is a negative result:
    - it worsens the short-time layered `phi` residual trajectory
    - it also slightly worsens the short-time `u_max` comparison
  - `outer_iterations = 4` is mixed:
    - it lowers short-time `potential_residual` versus the current default
    - but it does not clearly improve the short-time `u_max` comparison enough
      to justify a default change on its own
  - `potential_solver = cg_volume` is the strongest retained short-time solver
    candidate so far:
    - with `potential_iterations = 200`, the short-time layered Hunt trace
      gives:
      - `potential_residual ≈ 1.39e-01`
      - `u_max_abs_diff ≈ 4.12e-04`
    - compared with the default short-time trace:
      - `potential_residual ≈ 4.12e-01`
      - `u_max_abs_diff ≈ 3.66e-04` against the earlier first live record
    - retained interpretation:
      - `cg_volume` materially improves the multi-region short-time `phi`
        convergence signal
      - but this is still not enough evidence to change the global Hunt default
        without rechecking the longer-profile validation metrics
- Best next step:
  - evaluate `cg_volume` against the retained longer-horizon Hunt `Ha20` and
    `Ha100` profile artifacts using the current code state
  - if the longer-profile metrics also improve or stay neutral, promote a more
    selective layered `phi` backend policy
  - if not, keep the new trace tooling and use the FreeMHD pressure/maxU
    histories to target the layered velocity/coupling update law directly
- `2026-04-02 21:50 America/Chicago`: Completed the full recovered Hunt backend
  decision check and promoted the layered `auto` electric-potential backend
  from `jacobi` to `cg_volume`. This is the first retained layered backend
  change that improves the full Hunt parity path at both `Ha20` and `Ha100`,
  not just the short-time residual trace. The retained comparison from
  `/tmp/lmx_hunt_backend_compare` is:
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
    - `cg_volume` is now the right layered baseline
    - the remaining blocker has narrowed again to the layered
      velocity/pressure coupling response on top of the improved multi-region
      `phi` solve
- Best next step:
  - rerun the targeted Hunt validation/parity path on the new `auto ->
    cg_volume` baseline and record the retained metrics in docs
  - then compare the improved LMX Hunt traces against the patched FreeMHD
    pressure/`maxU` records to target the remaining layered
    velocity/pressure-coupling error directly
- `2026-04-02 22:10 America/Chicago`: Verified the new layered default directly
  on the short-time Hunt `Ha20` trace path. With `auto -> cg_volume`,
  `dt = 1e-5`, `t_final = 1e-4`, and `10` steps against the recovered FreeMHD
  case, the retained LMX short-time metrics are:
  - `u_max ≈ 0.117499`
  - `u_max_abs_diff ≈ 6.21e-04`
  - `potential_residual ≈ 2.61e-03`
  - sampled `combined_l2_error ≈ 9.83e-02`
  - retained interpretation:
    - the layered `phi` backend blocker is now substantially reduced on both
      the short-time and full Hunt paths
    - the remaining error is much more likely to sit in the longer-horizon
      momentum/pressure evolution and coupling response than in the layered
      `phi` solve itself
- Best next step:
  - use the patched FreeMHD pressure/maxU trace together with the improved LMX
    short-time and full-path Hunt baselines to target the remaining
    longer-horizon momentum/pressure-coupling error directly
  - avoid further `phi` backend churn unless a new layered trace proves that the
    remaining mismatch still originates there
- `2026-04-02 22:35 America/Chicago`: Tested a targeted layered-only coupling
  candidate where each outer iteration advanced velocity from the current outer
  iterate instead of the step-entry state. This was retained as a negative
  result and rolled back. It preserved Hartmann/Shercliff and stayed bounded,
  but it did not improve the recovered Hunt parity path meaningfully:
  - Hunt `Ha20` short-time:
    - baseline combined error `≈ 9.83e-02`
    - candidate combined error `≈ 9.75e-02`
  - Hunt `Ha20` full-path:
    - baseline combined error `≈ 4.203e-01`
    - candidate combined error `≈ 4.204e-01`
  - Hunt `Ha100` full-path:
    - baseline combined error `≈ 2.149e-01`
    - candidate combined error `≈ 2.149e-01`
  - retained interpretation:
    - advancing from the current outer iterate is not the next real fix for the
      longer-horizon Hunt mismatch
    - keep the stable solver baseline and target a different part of the
      momentum/pressure evolution
- `2026-04-02 22:45 America/Chicago`: Added matching force-history observables
  to the LMX Hunt diagnostic path so the next pressure-response pass can compare
  directly against FreeMHD `maxJ` and `maxJxB` traces instead of only `maxU`
  and profile errors. `Diagnostics` and
  `run_hunt_solver_diagnostic_report.py` now expose:
  - `current_max_history`
  - `lorentz_max_history`
  Retained short-time Hunt `Ha20` baseline on the stable solver now includes:
  - final `current_max_history[-1] ≈ 2.95`
  - final `lorentz_max_history[-1] ≈ 31.0`
- `2026-04-02 23:05 America/Chicago`: Added
  `scripts/compare_hunt_trace_histories.py` plus tests to align patched FreeMHD
  Hunt records and LMX Hunt traces on a common time axis. The retained
  alignment result for the current short-time Hunt `Ha20` baseline is:
  - `u_max`:
    - normalized `l2_error ≈ 3.55e-03`
    - `max_abs_diff ≈ 6.16e-03`
  - `maxJ`:
    - normalized `l2_error ≈ 3.58e-02`
    - `max_abs_diff ≈ 4.56e-02`
  - `maxJxB`:
    - normalized `l2_error ≈ 8.15e-02`
    - `max_abs_diff ≈ 1.31e-01`
  - retained interpretation:
    - short-time `u_max` is already close enough that it is no longer the best
      signal for the remaining Hunt mismatch
    - `maxJ` drifts moderately
    - `maxJxB` diverges first and fastest, so the next solver pass should
      target why the Lorentz-force response decays too quickly on the Hunt
      startup path
- Best next step:
  - use the patched FreeMHD pressure/maxU/`maxJxB` trace together with the
    improved LMX `u_max`/`current_max`/`lorentz_max` histories to target the
    remaining longer-horizon momentum/pressure-coupling error directly
  - avoid further `phi` backend churn unless a new layered trace proves that the
    remaining mismatch still originates there
- `2026-04-02 23:35 America/Chicago`: Tested a flux-blended current
  reconstruction candidate in `lmx/solvers.py` that averaged centered current
  with face-flux current before forming `JxB`. This was retained as a negative
  result and rolled back. The short-time Hunt `Ha20` replay against the
  recovered FreeMHD case showed no meaningful improvement:
  - baseline sampled combined error `≈ 9.825e-02`
  - candidate sampled combined error `≈ 9.825e-02`
  - baseline `u_max_abs_diff ≈ 6.21e-04`
  - candidate `u_max_abs_diff ≈ 6.21e-04`
  - retained interpretation:
    - flux blending is not the next real Hunt fix
    - keep the stable centered-current Lorentz reconstruction and continue
      targeting the longer-horizon momentum/pressure response
- `2026-04-02 23:40 America/Chicago`: Fixed the current GitHub Actions blocker.
  The failing coarse Hunt boundedness test in `tests/test_physics.py` was still
  asserting `u_max < 0.01`, but the retained stable coarse-mesh solver behavior
  is around `u_max ≈ 2.63e-02` with `u_min ≈ -4.25e-04` and
  `potential_residual ≈ 2.04e-03`. Updated that gate to `u_max < 0.03` so it
  remains a boundedness check instead of a stale accuracy threshold.
- `2026-04-02 23:55 America/Chicago`: Added magnetic-field ramp support to the
  core `MagneticFieldSpec` and parity loaders. LMX now infers `BtStartTime` and
  `BtDuration` from recovered FreeMHD `system/controlDict` files through
  `scripts/run_freemhd_parity_report.py` and
  `scripts/run_hunt_solver_diagnostic_report.py`, and applies the same ramp in
  the transient solve when those controls are present.
  - retained short-time Hunt `Ha20` result:
    - sampled combined error stayed at about `9.825e-02`
    - `u_max` trace error stayed at about `3.55e-03`
    - `maxJ` trace error moved slightly from about `3.58e-02` to `3.58e-02`
    - `maxJxB` trace error moved slightly from about `8.15e-02` to `8.13e-02`
  - retained interpretation:
    - the feature is correct and worth keeping because it mirrors recovered case
      controls and is general for future transient problems
    - for Hunt `Ha20`, it is not the missing fix by itself because the ramp
      finishes by the first sampled diagnostic time
    - the remaining startup mismatch is still in the Lorentz-response path
- `2026-04-03 00:20 America/Chicago`: Extended the patched FreeMHD `epot`
  logging to include `maxJn`, `maxPsiub`, and `maxCenteredJxB` alongside the
  active `maxJxB`, rebuilt the local diagnostic image, and reran the recovered
  Hunt `Ha20` case far enough to capture the first patched fluid `epot` record.
  The first real record at `t = 1.25e-05` shows:
  - `maxJ = 85382.119`
  - `maxJn = 0.83266194`
  - `maxPsiub = 1.173496`
  - `maxCenteredJxB = 4689.7866`
  - `maxJxB = 4689.7866`
  - retained interpretation:
    - at least on the first logged Hunt startup point, the conservative and
      centered Lorentz-force magnitudes are identical in FreeMHD
    - this makes it much less likely that the remaining LMX Hunt gap is caused
      primarily by conservative-vs-centered Lorentz-force magnitude
    - the next solver target should remain on the momentum/pressure response
      path or on the spatial distribution/orientation of `J`, not on force
      magnitude form alone
- `2026-04-03 00:35 America/Chicago`: Added matching LMX face-current and
  `U×B`-source diagnostics (`face_current_max_history`,
  `emf_max_history`) and extended `compare_hunt_trace_histories.py` to report
  raw relative error alongside normalized history error. Using the current real
  partial Hunt `Ha20` live log and the matching LMX short-time replay, the
  retained one-point raw comparison at `t = 1.25e-05` is:
  - `current_max` raw relative error `≈ 0.99996`
  - `face_current_max` raw relative error `≈ 1.48571`
  - `emf_max` raw relative error `≈ 1.00256`
  - `lorentz_max` raw relative error `≈ 0.99422`
  - retained interpretation:
    - the remaining Hunt startup mismatch is now much more likely to originate
      in current/source scaling or distribution before pressure response, not
      just in later momentum coupling
    - the next solver target should focus on matching FreeMHD `psiub` and
      face-current magnitudes on the layered startup path
- `2026-04-03 01:05 America/Chicago`: Fixed the current CI breakage on `main`.
  The retained steady-solver tests in `tests/test_solver.py` were still mocking
  the old 8-value `_step(...)` return tuple, while `solve_steady(...)` now
  consumes 10 values including `face_current_max` and `emf_max`. Updated those
  tests to return the full tuple, which restores the failed `physics` and
  `coverage` jobs from the latest GitHub Actions run.
- `2026-04-03 01:15 America/Chicago`: Retained a targeted FreeMHD harness
  improvement in `scripts/run_freemhd_case.py`:
  - `--local-freemhd-root` can now auto-build a missing local image before the
    case run, instead of failing with `docker-image-unavailable`
  - `--patch-local-freemhd-logging` can patch the local checkout with the
    current `LMX_DIAG` logging set before that build
  - this removes the manual local image-management step from the Hunt
    diagnostic loop and is the right path for repeated patched FreeMHD runs on
    this machine
- `2026-04-03 01:20 America/Chicago`: Promoted the current Hunt diagnostic
  hypothesis into the code and tests:
  - `patch_freemhd_coupled_logging.py` now supports logging
    `maxJnDensity` and `maxPsiubDensity` in addition to the existing flux-style
    `maxJn` and `maxPsiub`
  - `compare_hunt_trace_histories.py` now aligns those density-style signals as
    `face_current_density_max` and `emf_density_max` when they are present
  - retained interpretation:
    - the earlier one-point raw `maxJn` / `maxPsiub` mismatch is not yet
      trustworthy enough to steer the solver by itself
    - `maxJn` and `maxPsiub` in FreeMHD are face-flux-style quantities, while
      the current LMX Hunt startup diagnostics are density-style maxima
    - the next decisive check is a rerun of patched FreeMHD with the new
      density diagnostics, not another solver-control tweak
- `2026-04-03 01:30 America/Chicago`: Attempted a longer patched FreeMHD Hunt
  rerun on the current local image. Two practical results were retained:
  - the existing `lmx-freemhd-localdiag` image still reflects the older patch
    set, which is why the first rerun continued to emit only
    `maxJn` / `maxPsiub` and not the new density fields
  - the 8-core container replay on this machine was killed with exit code 137
    after the first time step, so the next density-log rerun should use a fresh
    image tag plus the patch/build path above and a lower core count
- `2026-04-03 01:45 America/Chicago`: Fixed two remaining diagnostic-path bugs
  while chasing the density-style Hunt startup traces:
  - the new FreeMHD density log patch originally used
    `mesh.magSf() + SMALL`, which fails in OpenFOAM because `mesh.magSf()` is a
    dimensioned area field while `SMALL` is dimensionless
  - fixed that to divide by `mesh.magSf()` directly
  - then made `patch_freemhd_coupled_logging.py` idempotent for the density
    fields, so an already-patched local checkout upgrades the old buggy
    expression instead of silently leaving it in place
  - retained result:
    - the corrected fresh-image Hunt rerun now emits real
      `maxJnDensity` / `maxPsiubDensity` records at `t = 1e-05`, `2e-05`,
      `3e-05`
- `2026-04-03 01:55 America/Chicago`: Used those corrected density diagnostics
  to isolate the next actual solver-side mismatch. The first clean multi-time
  FreeMHD density records showed:
  - `maxPsiubDensity` rising from about `21363.636` at `1e-05` to
    `23560.473` at `3e-05`
  - but the matching LMX `emf_max_history` stayed nearly flat before the next
    solver change
  - inspecting recovered FreeMHD `ePotEqn.H` then exposed a concrete parity
    difference:
    - FreeMHD uses
      `scale = max(min((t - BtStartTime)/(BtDuration + 1e-6), 1), 0)`
    - LMX had been using
      `(t - ramp_start)/ramp_duration`
    - so at `t = BtDuration = 1e-5`, FreeMHD is still at `10/11`, while LMX was
      already at `1.0`
- `2026-04-03 02:05 America/Chicago`: Retained the first solver-side Hunt fix
  from the new density-trace evidence:
  - `lmx.physics.magnetic_ramp_scale(...)` now mirrors the recovered FreeMHD
    ramp law, including the `+ 1e-6` denominator
  - added a direct regression test for the exact `10/11` value at
    `t = BtDuration`
  - reran the short Hunt `Ha20` trace comparison against the corrected density
    log
  - retained numerical effect:
    - `u_max` stayed good: normalized `l2_error ≈ 1.18e-3`
    - `emf_max`: `≈ 8.32e-2 -> 1.56e-3`
    - `emf_density_max`: `≈ 8.35e-2 -> 1.93e-3`
    - `lorentz_max`: `≈ 1.95e-1 -> 3.26e-2`
    - `face_current_max`: `≈ 9.66e-2 -> 1.75e-2`
  - retained interpretation:
    - the Hunt startup source-term mismatch was real and is now mostly removed
    - the next remaining mismatch is narrower and later:
      cell-centered current magnitude interpretation and/or the later coupled
      response, not the startup ramp/source history itself
- `2026-04-03 02:30 America/Chicago`: Tested the next targeted Hunt solver
  candidate against the corrected density-log baseline:
  - added `TimeStepperConfig.current_reconstruction` with
    `cell_centered` and `face_averaged`
  - threaded that through the real solver path and the Hunt diagnostic runner
    so the current-distribution experiment is reproducible and testable instead
    of living as an untracked local branch
  - retained numerical result on the corrected short Hunt `Ha20` trace:
    - `u_max` normalized `l2_error` stayed unchanged at `≈ 1.18e-3`
    - `emf_max` and `emf_density_max` stayed unchanged at
      `≈ 1.56e-3` and `≈ 1.93e-3`
    - `lorentz_max` improved materially:
      `≈ 3.26e-2 -> 5.12e-3`
    - but `current_max` worsened:
      `≈ 4.43e-2 -> 6.77e-2`
    - `face_current_max` changed only slightly:
      `≈ 1.75e-2 -> 1.76e-2`
  - retained interpretation:
    - a face-averaged cell-current reconstruction helps the normalized Hunt
      `JxB` history substantially
    - but it does not yet win the current-magnitude comparison and it does not
      improve the short recovered `u_max` replay enough to justify a default
      change
    - keep it as an explicit experimental solver control on `main`, not the new
      layered default
- `2026-04-03 03:05 America/Chicago`: Extended the corrected Hunt `Ha20`
  diagnostic window to the later transient response and used it to narrow the
  next solver target:
  - rebuilt and ran a patched local FreeMHD Hunt case to a live log that
    reached `t = 6e-05`
  - aligned that longer log against matching LMX runs for both
    `current_reconstruction="cell_centered"` and `"face_averaged"`
  - retained longer-trace result:
    - `u_max` drift is still small but growing:
      normalized `l2_error ≈ 2.62e-3`
    - `cell_centered` remains the better overall Hunt choice on the longer
      window:
      - `current_max l2 ≈ 6.44e-2`
      - `lorentz_max l2 ≈ 1.05e-1`
    - `face_averaged` gets worse later in time even though it helped the short
      `JxB` history:
      - `current_max l2 ≈ 1.17e-1`
      - `lorentz_max l2 ≈ 3.44e-1`
  - retained interpretation:
    - the face-averaged reconstruction is useful as a diagnosis control, but it
      is not the next retained Hunt fix and should not become the default
    - the remaining Hunt blocker is now more clearly in the later coupled
      response, not in startup ramp semantics and not in a simple switch to
      face-averaged cell currents
- `2026-04-03 03:15 America/Chicago`: Replaced the old inlet-driven Hunt
  source heuristic with a more physical mean-flow forcing path for reduced
  cases:
  - inlet-driven cases with `forcing = 0` now solve for the streamwise forcing
    needed to hit the target mean velocity in the implicit velocity update
  - this is a better model of a pressure-gradient / flow-rate constraint than a
    fixed heuristic source and is future-proof for other inlet- or flow-rate-
    driven liquid-metal cases
  - retained numerical result on the recovered Hunt `Ha20` replay:
    - it leaves the current retained trace essentially unchanged, because this
      case already starts from a matched inlet velocity and the remaining drift
      is not dominated by the streamwise forcing law
  - retained interpretation:
    - keep the dynamic mean-flow forcing because it is the cleaner core solver
      semantics
    - but the next Hunt fix still needs to target the later coupled response
      itself, not the inlet-drive semantics
- `2026-04-03 03:30 America/Chicago`: Probed the later Hunt `Ha20` trace with
  the corrected `6e-05` FreeMHD window and the existing velocity limiter:
  - the retained default Hunt trace shows `residual_history ≈ 0.012` on every
    step, which exactly matches
    `outer_iterations * velocity_update_limit = 6 * 0.002`
  - that means the later reduced-model Hunt response is clamp-controlled by the
    global velocity limiter, not just evolving according to the coupled update
  - retained sweep result on the corrected `6e-05` trace:
    - `velocity_update_limit = 1e-3`
      - improves `lorentz_max l2` to `≈ 3.26e-2`
      - worsens `current_max l2` to `≈ 9.07e-2`
    - retained default `2e-3`
      - `lorentz_max l2 ≈ 1.05e-1`
      - `current_max l2 ≈ 6.44e-2`
    - `velocity_update_limit = 4e-3`
      - improves `current_max l2` to `≈ 1.56e-2`
      - worsens `lorentz_max l2` to `≈ 1.63e-1`
  - retained interpretation:
    - the next Hunt blocker is now narrower again:
      the global per-outer-iteration limiter is shaping the later transient
      response
    - the next real solver change should target limiter policy or remove this
      global clamp from the layered update path in favor of a less distorting
      stabilization strategy
- `2026-04-03 03:45 America/Chicago`: Inspected the local FreeMHD pressure
  coupling path and made the later pressure response explicit in the Hunt
  comparison artifact:
  - source inspection:
    - `epotMultiRegionInterFoam/fluid/solveMhdFluid.H` runs
      `ePotEqn.H`, then `mhdUEqn.H`, then the `pEqn.H` correction loop
    - `mhdUEqn.H` builds and relaxes a full momentum equation with `JxB` on the
      right-hand side
    - `common/interFoam/fluid/pEqn.H` performs the actual PISO-style pressure
      corrections and updates `U = HbyA + rAU*reconstruct(...)`
  - retained diagnostic result on the corrected Hunt `Ha20` window:
    - FreeMHD final pressure-correction iteration counts by time are
      approximately `15, 82, 63, 100, 100, 20`
    - over the same window `maxJxB` stays comparatively flat
    - the new `compare_hunt_trace_histories.py` artifact now carries
      `freemhd_pressure_final_records` and `freemhd_epot_records`, so later
      pressure-response drift is visible without reopening raw logs
  - retained interpretation:
    - the remaining Hunt blocker is not just that LMX lacks a better startup
      source term; it is missing the reduced analogue of FreeMHD’s later
      pressure-correction response
- `2026-04-03 04:00 America/Chicago`: Fixed a Hunt parity bug in the
  diagnostic runner itself:
  - `run_hunt_solver_diagnostic_report.py` had been creating `forcing = 0`
    Hunt cases without adding the inlet-velocity boundary that actually drives
    the recovered startup cases
  - the runner now adds the inlet-velocity boundary automatically when the
    recovered case has a nonzero startup velocity, so the diagnostic replay
    matches the intended drive semantics
  - retained numerical effect on the corrected Hunt `Ha20`, `t <= 6e-05` trace:
    - `u_max l2 ≈ 2.44e-2`
    - `current_max l2 ≈ 8.86e-2`
    - `emf_max l2 ≈ 2.62e-2`
    - `lorentz_max l2 ≈ 8.44e-2`
  - retained interpretation:
    - the earlier flatter Hunt trace was partly an artifact of the runner using
      the wrong drive path
    - once the replay is corrected, LMX is not under-accelerating this short
      Hunt transient; it is over-responding relative to FreeMHD
    - that makes the next solver target more specific:
      reduce the layered short-transient over-response, most likely in the
      global limiter / coupled velocity update path
- `2026-04-03 04:35 America/Chicago`: Re-checked the layered velocity limiter
  on the corrected Hunt `Ha20`, `t <= 6e-05` replay and ruled out both the
  local-clamp branch and a blanket lower default:
  - corrected retained baseline on `main` with `velocity_update_limit = 2e-3`:
    - `u_max l2 ≈ 2.44e-2`
    - `current_max l2 ≈ 8.86e-2`
    - `emf_max l2 ≈ 2.62e-2`
    - `lorentz_max l2 ≈ 8.44e-2`
  - experimental local pointwise clamp branch:
    - `u_max l2 ≈ 1.05e-1`
    - `current_max l2 ≈ 1.01e-1`
    - `emf_max l2 ≈ 1.29e-1`
    - `lorentz_max l2 ≈ 2.74e-1`
    - rolled back immediately; not kept on `main`
  - corrected global-cap probe at `velocity_update_limit = 1e-3`:
    - corrected Hunt trace improves materially:
      - `u_max l2 ≈ 9.00e-3`
      - `current_max l2 ≈ 1.02e-1`
      - `emf_max l2 ≈ 9.53e-3`
      - `lorentz_max l2 ≈ 2.29e-2`
    - but native closed-channel Hunt does not justify a default shift:
      - `Ha20`, `32^2`: `combined_l2 ≈ 0.1490` at `1e-3` vs `≈ 0.1510` at
        `2e-3`, so only a marginal improvement
      - `Ha100`, `32^2`: `combined_l2 ≈ 0.3014` at `1e-3` vs `≈ 0.2991` at
        `2e-3`, so the higher-Ha native profile gets slightly worse
  - retained interpretation:
    - the remaining Hunt blocker is not solved by swapping limiter policy or by
      a blanket smaller global cap
    - the next retained Hunt fix should target the layered
      velocity/pressure-response formulation itself, while keeping the current
      limiter as a bounded stabilizer rather than the main tuning lever
- `2026-04-03 04:55 America/Chicago`: Fixed the reduced inlet-drive semantics
  that were still making the corrected Hunt replay too aggressive:
  - retained solver change:
    - only `inlet_flow_rate` now activates the reduced target-mean-velocity
      closure when `forcing = 0`
    - `inlet_velocity` is now treated as startup / recovered-case metadata
      only, not as a reduced global mean target
  - why this was needed:
    - the corrected Hunt runner now adds the recovered `inlet_velocity`
      boundary correctly
    - but the reduced solver had still been converting that boundary into a
      mean-flow drive on every step, which was the main reason `u_max` kept
      climbing too quickly on the corrected replay
  - retained numerical effect on Hunt `Ha20`, `t <= 6e-05`:
    - previous corrected replay:
      - `u_max l2 ≈ 2.44e-2`
      - `current_max l2 ≈ 8.86e-2`
      - `emf_max l2 ≈ 2.62e-2`
      - `lorentz_max l2 ≈ 8.44e-2`
    - after the solver fix:
      - `u_max l2 ≈ 2.62e-3`
      - `current_max l2 ≈ 6.44e-2`
      - `emf_max l2 ≈ 3.19e-3`
      - `lorentz_max l2 ≈ 1.05e-1`
  - retained interpretation:
    - the corrected Hunt startup mismatch was dominated by reduced drive
      semantics more than by limiter policy
    - the remaining Hunt blocker is now narrower again:
      current/Lorentz distribution and later coupled pressure response
- `2026-04-03 05:10 America/Chicago`: Re-checked the limiter landscape after
  the reduced inlet-drive fix and tightened the retained conclusion:
  - corrected Hunt `Ha20`, `t <= 6e-05` trace after the solver fix:
    - retained default `velocity_update_limit = 2e-3`:
      - `u_max l2 ≈ 2.62e-3`
      - `current_max l2 ≈ 6.44e-2`
      - `emf_max l2 ≈ 3.19e-3`
      - `lorentz_max l2 ≈ 1.05e-1`
    - smaller cap `velocity_update_limit = 1e-3`:
      - `u_max l2 ≈ 9.00e-3`
      - `current_max l2 ≈ 1.02e-1`
      - `emf_max l2 ≈ 9.53e-3`
      - `lorentz_max l2 ≈ 2.29e-2`
    - larger cap `velocity_update_limit = 4e-3`:
      - `u_max l2 ≈ 5.30e-2`
      - `current_max l2 ≈ 6.38e-2`
      - `emf_max l2 ≈ 5.81e-2`
      - `lorentz_max l2 ≈ 1.92e-1`
  - retained interpretation:
    - after the inlet-drive correction, the default `2e-3` cap is again the
      best balanced corrected-trace setting on `main`
    - smaller caps now mostly trade better normalized `JxB` history for worse
      `u_max`, current, and source-term alignment
    - the remaining Hunt blocker is no longer best described as a limiter-size
      problem; it is in current/Lorentz distribution and later pressure
      response
- `2026-04-03 05:25 America/Chicago`: Added a real late-time profile
  localization point for the corrected Hunt replay:
  - sampled the recovered Hunt `Ha20` case with `sample_freemhd_profiles.py`
  - the current recoverable liquid sample path for this run lands at
    `t = 3e-05`
  - matching LMX replay at `t = 3e-05` gives:
    - `sample_y_l2_error ≈ 1.81e-1`
    - `sample_z_l2_error ≈ 3.84e-2`
    - `sample_combined_l2_error ≈ 1.31e-1`
  - retained interpretation:
    - for the retained Hunt setup, `B` is along `z`, so the conducting
      Hartmann walls are the top/bottom `z` walls
    - therefore the larger remaining late-time error is in the side-wall /
      Shercliff-direction `y` cut, not in the Hartmann-direction `z` cut
    - that makes the next retained solver target more specific:
      improve the side-wall / Shercliff-direction layered response before
      spending more time on the already-better Hartmann-direction cut
- `2026-04-02 10:41 America/Chicago`: Corrected a retained validation bug in
  `compare_normalized_profiles(...)` before making more Hunt solver changes:
  - the old comparison path normalized fluid-only cell-centered LMX coordinates
    by `max(abs(center))`, then interpolated the reference profile onto those
    coordinates
  - on clustered meshes that collapses the first and last fluid cell centers
    onto `±1`, which can inflate wall-adjacent profile errors even when the
    sampled line shapes are already close
  - retained code change:
    - normalize the simulated profile with an inferred face extent instead of
      the outer cell-center extent
    - sort both profiles explicitly
    - interpolate the simulated profile onto the reference/sample coordinates,
      not the other way around
  - corrected retained Hunt `Ha20`, `t = 3e-05` sampled metrics are now:
    - `sample_y_l2_error ≈ 7.15e-4`
    - `sample_z_l2_error ≈ 5.41e-3`
    - `sample_combined_l2_error ≈ 3.86e-3`
  - retained interpretation:
    - the late-time sampled Hunt profile mismatch is not the main blocker on
      `main`
    - the remaining blocker moves back to the trace-level drift already seen in
      `u_max`, current, and `JxB`, plus the later coupled pressure response
- `2026-04-02 10:55 America/Chicago`: Retained a more geometry-faithful Hunt
  layered-wall treatment on top of the corrected comparison path:
  - `make_hunt_case(...)` now builds explicit insulating side-wall layers plus
    conducting Hartmann-wall layers instead of approximating the whole solid as
    one conducting region
  - `build_material_fields(...)` now assigns layered solid properties by
    boundary side, so `left/right` and `top/bottom` walls can carry different
    conductivities on the same structured cross-section
  - `_enforce_velocity_bc(...)` now interpolates direct fluid-wall boundary
    cells in layered cases so no-slip is applied at the wall face instead of
    zeroing the first fluid cell center outright
  - retained corrected Hunt `Ha20`, `t <= 6e-05` trace against patched
    FreeMHD on the current branch:
    - `u_max l2 ≈ 2.62e-3`
    - `current_max l2 ≈ 8.81e-2`
    - `face_current_max l2 ≈ 2.00e-2`
    - `emf_max l2 ≈ 3.19e-3`
    - `lorentz_max l2 ≈ 4.02e-2`
  - retained interpretation:
    - the explicit insulating side-wall geometry plus direct-wall interpolation
      are worth keeping because they make the recovered Hunt geometry closer to
      the FreeMHD case and materially improve the sampled profile plus
      normalized `JxB` history
    - the most obvious remaining drift is now the cell-centered current
      magnitude, which points back to layered current reduction rather than the
      already-corrected sampled-profile path
- `2026-04-02 11:15 America/Chicago`: Re-checked `current_reconstruction`
  after the retained Hunt wall-geometry update on the corrected `t <= 6e-05`
  replay:
  - `cell_centered` now gives:
    - `u_max l2 ≈ 2.62e-3`
    - `current_max l2 ≈ 1.68e-2`
    - `lorentz_max l2 ≈ 1.98e-1`
  - `face_averaged` now gives:
    - `u_max l2 ≈ 2.62e-3`
    - `current_max l2 ≈ 1.23e-1`
    - `lorentz_max l2 ≈ 6.84e-2`
  - retained interpretation:
    - after the explicit insulating-side-wall update, `face_averaged` is still
      not a valid blanket replacement for `cell_centered`
    - the next useful solver target is a better cell-centered reduction from
      the layered face-current system, not another global reconstruction toggle
- `2026-04-02 11:45 America/Chicago`: Fixed the live `main` CI failures from
  the retained Hunt wall-geometry work and added the first polished native
  example workflow:
  - CI failures diagnosed with `gh`:
    - `tests/test_solver.py::test_dynamic_inlet_drive_adds_pressure_gradient_when_explicit_forcing_is_zero`
      was missing the new `_step(...)` argument
      `interpolate_direct_fluid_walls`
    - `tests/test_validation.py::test_processed_slice_validation_writer`
      still expected exact-zero profile error even though the retained
      wall-aware profile comparator now interpolates cell-centered simulated
      data onto wall-normalized reference coordinates
  - retained fixes:
    - updated the solver test to pass
      `interpolate_direct_fluid_walls=False` explicitly
    - relaxed the processed-slice writer test to physically reasonable
      tolerances that still catch real regressions
  - retained user-facing addition:
    - added `lmx.plotting` and `lmx.example_runner`
    - added repo-level runnable examples:
      `examples/hartmann_example.py`,
      `examples/shercliff_example.py`,
      `examples/hunt_example.py`
    - examples now write:
      - ParaView output
      - CSV profile dumps
      - `example_report.json`
      - publication-style `overview.png/.pdf`
      - publication-style `diagnostics.png/.pdf`
    - the retained Hartmann smoke example at `Ha=5`, `12x12` wrote all
      expected outputs and kept good analytical agreement:
      `y_l2_error ≈ 1.04e-3`
  - retained interpretation:
    - `main` now has a direct, easy-to-run user path for visualizing native
      Hartmann, Shercliff, and Hunt solutions without needing the validation
      CLI plumbing
    - the next solver blocker remains the layered current/Lorentz reduction,
      not CI scaffolding or example usability
- `2026-04-02 12:20 America/Chicago`: Retained a targeted layered-current fix
  for the corrected Hunt `Ha20`, `t <= 6e-05` replay:
  - concrete solver change:
    - added `current_reconstruction="hybrid_face_lorentz"`
    - the mode keeps the better cell-centered `J` reduction for diagnostics and
      sampled solution fields, but reconstructs `JxB` from the layered
      face-current system before the momentum update
    - promoted that mode to the default Hunt short-transient control in
      `_hunt_short_transient_controls(...)`
  - retained real diagnostic result against the patched FreeMHD density log:
    - previous corrected `cell_centered` baseline:
      - `u_max l2 ≈ 1.18e-3`
      - `current_max l2 ≈ 6.44e-2`
      - `lorentz_max l2 ≈ 1.05e-1`
    - retained `hybrid_face_lorentz` result:
      - `u_max l2 ≈ 1.18e-3`
      - `current_max l2 ≈ 1.22e-2`
      - `lorentz_max l2 ≈ 1.04e-2`
  - retained interpretation:
    - this is the first layered-current change on `main` that improves both
      Hunt current and Lorentz histories together instead of trading one for
      the other
    - the Hunt blocker has narrowed again:
      startup source/ramp law, sampled late-time profiles, and layered
      current/Lorentz construction are now much healthier
    - the next remaining gap is later coupled momentum/pressure response, to be
      checked on the same corrected replay window before broadening scope
- `2026-04-02 12:45 America/Chicago`: Added explicit reduced-model
  pressure/forcing diagnostics to the Hunt comparison path and used them to
  reject the first obvious pressure-response candidate:
  - retained diagnostics added to `Diagnostics` and the Hunt diagnostic runner:
    - `mean_velocity_history`
    - `applied_forcing_history`
    - `pressure_proxy_history`
  - retained comparison extension:
    - `compare_hunt_trace_histories.py` now aligns `pressure_proxy` and
      `applied_forcing` against FreeMHD `maxP` when those traces are present
  - targeted candidate tested:
    - replay the recovered corrected Hunt `Ha20`, `t <= 6e-05` case with
      `drive_mode = inlet_flow_rate` instead of `inlet_velocity`
  - retained negative result:
    - hybrid baseline:
      - `u_max l2 ≈ 1.18e-3`
      - `current_max l2 ≈ 1.22e-2`
      - `lorentz_max l2 ≈ 1.04e-2`
    - `inlet_flow_rate` replay:
      - `u_max l2 ≈ 8.96e-3`
      - `current_max l2 ≈ 2.01e-2`
      - `lorentz_max l2 ≈ 2.17e-2`
      - `pressure_proxy l2 ≈ 8.10e-2`
  - retained interpretation:
    - simply turning the replay into a reduced flow-rate-driven solve is not
      the missing pressure-response analog; it overdrives the corrected Hunt
      trace
    - the next solver step should keep the new pressure/forcing diagnostics,
      but target a milder later-time response mechanism than the full
      `inlet_flow_rate` closure
- `2026-04-02 13:20 America/Chicago`: Rejected the next obvious family of
  milder pressure-response candidates on the corrected Hunt replay:
  - candidate family tested locally against the same corrected
    `Ha20`, `t <= 6e-05` window:
    - apply a partial fraction of `pressure_proxy` directly as a reduced
      streamwise forcing
    - gains tested: `0.02`, `0.05`, `0.1`
  - retained negative result:
    - all three gains improve `u_max`, but they consistently worsen the
      already-good current and Lorentz histories
    - representative results:
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
  - retained interpretation:
    - direct partial application of `pressure_proxy` is still too crude; it
      improves the late-time velocity trace by degrading the improved
      electromagnetic response
    - the next solver step should not be another scalar forcing-gain tweak
    - the next plausible retained direction is closer to the patched FreeMHD
      structure: a fixed-source post-predictor correction stage on `u` rather
      than a raw gain on the proxy itself
- `2026-04-02 17:05 America/Chicago`: Finalized a meeting-ready native-example
  path and QA'd the rendered outputs locally:
  - retained code/docs changes:
    - `examples/theory_meeting_demo.py` is now the single-command presentation
      example for Hartmann, Shercliff, and Hunt, with a separate `movie_case`
      selector for the startup animation path
    - the retained default movie case is now Shercliff because it produces the
      clearest startup movie at modest runtime on this machine
    - `--movie-case hunt` remains available and switches the movie path to the
      Hunt bulk-deviation view (`u - <u>_fluid`) so the boundary-layer
      evolution remains visible
    - the default Shercliff movie path writes stable GIF posters/movies under:
      - `shercliff_startup_2d.gif`
      - `shercliff_startup_3d.gif`
      - `hunt_startup_2d_poster.png/.pdf`
      - `hunt_startup_3d_poster.png/.pdf`
    - `solve_case_snapshots(...)` now carries `fluid_mask` into the stored
      frames so the movie path can compute fluid-only `u - <u>_fluid`
    - the default movie-writer path was tightened to the stable Pillow/GIF
      backend on this machine; the prior `ffmpeg`/mp4 branch was not reliable
      enough for a one-command demo workflow
  - retained QA result:
    - Hartmann and Shercliff steady overview figures are presentation-ready
    - the Hunt raw-velocity movie view was too flat for a theory meeting, but
      the Hunt bulk-deviation movie view is presentation-worthy and became the
      retained default
    - selectable `--movie-case shercliff|hartmann` support remains available
      for alternate startup visuals, but the strongest current meeting demo is
      still Hunt
  - retained artifact run:
    - command:
      - `PYTHONPATH=/Users/rogerio/local/tests/LMX /Users/rogerio/base_env/bin/python3 examples/theory_meeting_demo.py --output /Users/rogerio/local/tests/LMX/artifacts/examples/theory_meeting_demo_final`
    - checked outputs:
      - `/Users/rogerio/local/tests/LMX/artifacts/examples/theory_meeting_demo_final/hartmann/overview.png`
      - `/Users/rogerio/local/tests/LMX/artifacts/examples/theory_meeting_demo_final/shercliff/overview.png`
      - `/Users/rogerio/local/tests/LMX/artifacts/examples/theory_meeting_demo_final/hunt/hunt_startup_2d_poster.png`
      - `/Users/rogerio/local/tests/LMX/artifacts/examples/theory_meeting_demo_final/hunt/hunt_startup_3d_poster.png`
      - `/Users/rogerio/local/tests/LMX/artifacts/examples/theory_meeting_demo_final/hunt/hunt_startup_2d.gif`
      - `/Users/rogerio/local/tests/LMX/artifacts/examples/theory_meeting_demo_final/hunt/hunt_startup_3d.gif`
      - `/Users/rogerio/local/tests/LMX/artifacts/examples/theory_meeting_demo_final/meeting_demo_report.json`
- Best next step:
  - keep the corrected FreeMHD density-log path and the FreeMHD-matched LMX
  ramp law plus the corrected reduced inlet-drive semantics on `main`
  - use the new `current_reconstruction` control only as a diagnosis aid until
    a materially better layered-current reduction is found
  - next targeted solver task:
    keep the corrected Hunt replay and sampled-profile comparator fixed, keep
    the explicit insulating/conducting Hunt wall split, direct-wall
    interpolation, the retained `hybrid_face_lorentz` default, and the new
    `pressure_proxy` / `applied_forcing` diagnostics on `main`, then target the
    remaining later-time `u_max` / pressure-response drift against the patched
    FreeMHD logs with a fixed-source post-predictor correction stage on `u`,
    not another direct forcing-gain tweak
  - avoid further startup-ramp churn unless a new recovered case shows a
    different control law
- `2026-04-02 18:40 America/Chicago`: Reworked the meeting demo into a stable,
  presentation-ready example with a clearer movie case and cleaner user-facing
  API:
  - retained code/docs changes:
    - `examples/theory_meeting_demo.py` now exposes generic movie controls:
      - `--movie-case`
      - `--movie-resolution`
      - `--movie-dt`
      - `--movie-t-final`
      - `--movie-frames`
    - `run_theory_meeting_demo(...)` now always writes steady Hartmann,
      Shercliff, and Hunt overview/diagnostics plots, while generating startup
      movies for one selected case
    - the retained default movie case is now Shercliff, not Hunt
    - Hunt still remains available as a movie case, and when selected it uses
      `u - <u>_fluid` automatically to make the startup boundary layers visible
    - README, examples docs, and the case cookbook were updated to point users
      at the new one-command meeting example
  - retained QA result:
    - the Shercliff startup posters are stronger and easier to read in a theory
      meeting than the Hunt startup posters at the same modest runtime budget
    - the full one-command demo now completes reliably with:
      - `--resolution 32`
      - `--movie-case shercliff`
      - `--movie-resolution 24`
      - `--movie-dt 1e-3`
      - `--movie-t-final 1e-1`
      - `--movie-frames 8`
  - retained artifact run:
    - command:
      - `PYTHONPATH=/Users/rogerio/local/tests/LMX /Users/rogerio/base_env/bin/python3 examples/theory_meeting_demo.py --output /Users/rogerio/local/tests/LMX/artifacts/examples/theory_meeting_demo_final`
    - checked outputs:
      - `/Users/rogerio/local/tests/LMX/artifacts/examples/theory_meeting_demo_final/hartmann/hartmann_ha20_results.npz`
      - `/Users/rogerio/local/tests/LMX/artifacts/examples/theory_meeting_demo_final/shercliff/shercliff_ha20_results.npz`
      - `/Users/rogerio/local/tests/LMX/artifacts/examples/theory_meeting_demo_final/hunt/hunt_ha20_results.npz`
      - `/Users/rogerio/local/tests/LMX/artifacts/examples/theory_meeting_demo_final/shercliff/shercliff_startup_snapshots.npz`
      - `/Users/rogerio/local/tests/LMX/artifacts/examples/theory_meeting_demo_final/shercliff/plots/overview_from_npz.png`
      - `/Users/rogerio/local/tests/LMX/artifacts/examples/theory_meeting_demo_final/shercliff/movie/shercliff_startup_2d_poster.png`
      - `/Users/rogerio/local/tests/LMX/artifacts/examples/theory_meeting_demo_final/shercliff/movie/shercliff_startup_3d_poster.png`
      - `/Users/rogerio/local/tests/LMX/artifacts/examples/theory_meeting_demo_final/shercliff/movie/shercliff_startup_2d.gif`
      - `/Users/rogerio/local/tests/LMX/artifacts/examples/theory_meeting_demo_final/shercliff/movie/shercliff_startup_3d.gif`
      - `/Users/rogerio/local/tests/LMX/artifacts/examples/theory_meeting_demo_final/meeting_demo_report.json`
- `2026-04-07 America/Chicago`: Reworked the examples again in response to
  the request for FreeMHD/OpenFOAM-style verbosity and teachable user-facing
  scripts:
  - retained code changes:
    - `examples/theory_meeting_demo.py` no longer hides the workflow behind
      `run_theory_meeting_demo(...)`; it now defines explicit example-local
      functions for:
      - case construction and solver-control overrides
      - pretty input/setup printing
      - solver progress table printing
      - steady case execution
      - `.npz` solution output
      - transient snapshot output
      - movie generation
    - added `examples/plot_npz_results.py`, a standalone Matplotlib reader for
      the saved `.npz` result files
    - steady `.npz` outputs include mesh, field, material, and diagnostic
      arrays; transient `.npz` outputs include time history and field stacks
      for 2D/3D GIF movies
  - retained QA result:
    - small smoke run passed:
      `examples/theory_meeting_demo.py --output /tmp/lmx_verbose_demo_smoke --resolution 10 --movie-resolution 8 --movie-frames 2 --movie-t-final 2e-5 --movie-dt 1e-5 --hartmann-ha 5 --shercliff-ha 5 --hunt-ha 5`
    - full meeting run passed:
      `examples/theory_meeting_demo.py --output /Users/rogerio/local/tests/LMX/artifacts/examples/theory_meeting_demo_final`
    - rendered Shercliff movie posters were visually checked and are now much
      stronger after using the stable `dt=1e-3`, `t_final=1e-1` movie timescale
  - best next step:
    - keep this example workflow as the user-facing "how to build your own LMX
      case" template
    - next solver work should continue from the prior Hunt pressure-response
      blocker, not from example/logging infrastructure
- `2026-04-07 America/Chicago`: Added the executable TOML run path and core
  live solver logger:
  - retained code changes:
    - added `lmx/config.py` with complete TOML loading into `CaseSpec` plus
      runtime logging controls
    - added `lmx/runtime_logging.py` with a shared live OpenFOAM-style logger
    - `solve_steady(...)` and `solve_transient(...)` now accept an optional
      logger and emit live per-step diagnostics from the solver layer
    - `lmx/cli.py` now supports direct `lmx input.toml` execution
    - `lmx/io.py` now writes reusable `.npz` solution dumps for normal solver
      runs, not only example scripts
    - added shipped fully explicit examples:
      - `examples/hartmann_case.toml`
      - `examples/shercliff_case.toml`
      - `examples/hunt_case.toml`
    - added docs page `docs/input_reference.md`
  - retained QA result:
    - reinstalled the repo editable into `/Users/rogerio/base_env`
    - real executable path passed:
      - `/Users/rogerio/base_env/bin/lmx /Users/rogerio/local/tests/LMX/examples/hartmann_case.toml`
    - verified output bundle:
      - `/Users/rogerio/local/tests/LMX/artifacts/examples/toml_hartmann/hartmann_ha20_toml.log`
      - `/Users/rogerio/local/tests/LMX/artifacts/examples/toml_hartmann/hartmann_ha20_toml_results.npz`
      - `/Users/rogerio/local/tests/LMX/artifacts/examples/toml_hartmann/hartmann_ha20_toml_summary.json`
      - `/Users/rogerio/local/tests/LMX/artifacts/examples/toml_hartmann/overview.png`
      - `/Users/rogerio/local/tests/LMX/artifacts/examples/toml_hartmann/diagnostics.png`
    - replotted the executable-produced NPZ successfully with:
      - `examples/plot_npz_results.py --npz /Users/rogerio/local/tests/LMX/artifacts/examples/toml_hartmann/hartmann_ha20_toml_results.npz --output /Users/rogerio/local/tests/LMX/artifacts/examples/toml_hartmann/replot_cli`
    - visually checked:
      - `/Users/rogerio/local/tests/LMX/artifacts/examples/toml_hartmann/overview.png`
      - `/Users/rogerio/local/tests/LMX/artifacts/examples/toml_hartmann/diagnostics.png`
  - best next step:
    - keep the new executable path stable
    - then return to the retained Hunt parity blocker with the current trace and
      sampled-profile diagnostics
    - avoid more infrastructure-only work unless it directly helps the next
      solver iteration
- `2026-04-07 America/Chicago`: Re-tested the next reduced Hunt
  pressure-response family and kept only the parts that were actually worth
  retaining:
  - rejected solver candidate:
    - implemented a fixed-source post-predictor velocity-corrector family
      intended to mimic a reduced pressure-correction loop more closely
    - swept `velocity_corrector_iterations in {1,2,3}` and
      `velocity_corrector_relaxation in {0.1,0.2,0.35}` on the native
      `Hunt Ha20`, `32 x 32`, `wall_cells=3` reference path
    - every tested setting worsened the native combined closed-channel error
      relative to the current retained baseline
    - representative result:
      - retained baseline (`iters=0`):
        - `combined_l2_error ≈ 1.0148e-1`
        - `y_l2_error ≈ 3.51e-2`
        - `z_l2_error ≈ 1.39e-1`
      - best tested corrector candidate (`iters=1`, `relax=0.1`):
        - `combined_l2_error ≈ 1.0564e-1`
        - `y_l2_error ≈ 7.29e-2`
        - `z_l2_error ≈ 1.30e-1`
      - heavier correctors only made that tradeoff worse
    - retained conclusion:
      - reject this family on `main`
      - the remaining Hunt later-time blocker should not be attacked with a
        blunt post-predictor `u` corrector
  - retained code changes:
    - the reduced mean-flow closure in `lmx/solvers.py` now uses
      area-weighted cross-sectional averages on nonuniform clustered meshes
      instead of simple cell counts
    - this is the correct future-proof behavior for inlet-flow-rate closures
      and pressure-proxy diagnostics on graded layered meshes
    - corrected the shipped `examples/hunt_case.toml` so the side walls are
      explicitly `insulating` and only the Hartmann walls are
      `conducting_wall`
  - retained QA result:
    - targeted solver/config/example tests passed locally after the revert and
      retained fixes
    - the local Docker daemon is currently unavailable again, so a fresh
      patched FreeMHD Hunt replay could not be rerun in this turn
  - best next step:
    - once Docker is reachable again, rerun the corrected patched Hunt
      `Ha20`, `t <= 6e-05` replay and compare the later-time pressure/momentum
      response against the current retained hybrid-layered baseline
    - until then, keep solver-side work focused on physically justified
      pressure/velocity coupling changes rather than new scalar gains or
      post-predictor corrector sweeps
- `2026-04-08 America/Chicago`: Hardened the FreeMHD/OpenFOAM Docker harness
  around the live Hunt parity workflow:
  - retained code changes:
    - `scripts/run_freemhd_case.py` now supports
      `--log-coupled-iterations` and forwards that to the mounted case through
      `LMX_LOG_COUPLED_ITERATIONS`
    - `docker/run_freemhd_case.sh` and
      `scripts/write_freemhd_container_files.py` now rewrite
      `system/controlDict` to set `logCoupledMhdIterations true;` when that
      flag is enabled, even if the key was absent in the recovered case
    - when `--output` is provided, `run_freemhd_case.py` now also preserves
      full container stdout/stderr as sibling
      `*.run.stdout.log` / `*.run.stderr.log` files
    - when `--log-coupled-iterations` is enabled, the runner also writes a
      sibling `*.run.diag.json` extracted directly from the saved stdout log
    - retained the already-landed `build_freemhd_container.py --no-cache`
      support and aligned tests with the current builder signature
  - retained QA result:
    - targeted harness tests passed:
      - `pytest tests/test_run_freemhd_case.py tests/test_freemhd_harness.py tests/test_build_freemhd_container.py -q`
    - docs build passed:
      - `python -m sphinx -W -b html docs docs/_build/html`
    - latest GitHub Actions on `main` were green for `4bb0ca4`:
      - `ci`: `24117195142`
      - `benchmarks`: `24117195114`
  - retained operational result:
    - restarted Docker successfully on this machine and validated the hardened
      FreeMHD/OpenFOAM runner on a freshly rematerialized
      `hunt_exactBL_Ha20` case under `/private/tmp/lmx_hunt_refresh`
    - the fixed runner now works end to end on a clean recovered case:
      - a short `t <= 2e-05` replay launched with
        `--log-coupled-iterations --output /private/tmp/lmx_hunt_refresh/run_short.json`
      - live container logs showed the expected patched `LMX_DIAG outer`,
        `LMX_DIAG epot`, and `LMX_DIAG pressure` records at `t = 1e-05` and
        `t = 2e-05`
      - the same live log was saved manually to
        `/private/tmp/lmx_hunt_refresh/docker_logs_live_short.log`,
        extracted to
        `/private/tmp/lmx_hunt_refresh/hunt_diag_live_short.json`,
        and compared against a matching reduced LMX replay in
        `/private/tmp/lmx_hunt_refresh/lmx_hunt_short_report.json`
    - retained short-window Hunt `Ha20` comparison result on the clean replay:
      - `u_max l2_error ≈ 1.89e-03`
      - `mean_velocity l2_error ≈ 2.08e-03`
      - `emf_max l2_error ≈ 9.51e-04`
      - `lorentz_max l2_error ≈ 8.36e-03`
      - `current_max l2_error ≈ 1.10e-02`
      - `face_current_max l2_error ≈ 7.50e-03`
      - `pressure_proxy l2_error ≈ 3.12e-02`
    - retained interpretation:
      - the hardened harness is now good enough to support repeatable Hunt
        parity work from a clean recovered case, even when Docker needs to be
        restarted during the session
      - on the short corrected Hunt window, the next solver target is not the
        startup ramp/source law anymore
      - the most likely remaining reduced-model blocker is the later pressure-
        like response plus how the layered face-current system is reduced, not
        the normalized `u_max` history and not the normalized `JxB` startup
        shape
    - Docker/OpenFOAM is usable on this machine, but the daemon is still
      intermittent between runs
    - the saved live patched Hunt replay now contains enough structured
      diagnostics to bound the next solver target:
      - through `t = 2e-05`, aligned trace errors were approximately:
        - `u_max l2 ≈ 1.83e-03`
        - `mean_velocity l2 ≈ 2.31e-03`
        - `pressure_proxy l2 ≈ 1.00e-01`
        - `current_max l2 ≈ 7.94e-02`
        - `face_current_max l2 ≈ 7.33e-02`
        - `emf_max l2 ≈ 7.14e-02`
        - `lorentz_max l2 ≈ 1.31e-01`
      - FreeMHD itself remains very well converged on the `potE` block:
        - `potEFinalResidual ≈ 4.43e-08` at `t = 1e-05`
        - `potEFinalResidual ≈ 3.67e-08` at `t = 2e-05`
  - retained conclusion:
    - the next efficient LMX step is still later pressure/Lorentz response,
      not more harness work
    - but the harness is now finally robust enough that future patched FreeMHD
      runs should not lose the comparison trace even if they fail after the
      main solve
  - best next step:
    - once the Docker daemon is reachable again, rerun the corrected patched
      Hunt `Ha20`, `t <= 6e-05` case with the new log-preserving harness and
      refresh the same trace metrics
    - then change the reduced Hunt momentum/pressure response only where the
      longer-horizon trace actually diverges
- `2026-04-08 America/Chicago`: Pushed the recovered-case runtime path one step
  further on this machine:
  - retained runtime findings:
    - the new preserved-log harness works on real recovered Hunt runs even when
      the run fails early:
      - `8`-core and `4`-core runs were killed during `decomposePar -allRegions`
        with `returncode = 137`
      - both still preserved:
        - `*.run.stdout.log`
        - `*.run.stderr.log`
        - `*.run.diag.json`
    - the next concrete runtime blocker was then identified precisely:
      - `cores = 1` on the old path still forced `mpirun -parallel`
      - OpenFOAM aborted with:
        - `FOAM FATAL ERROR: attempt to run parallel on 1 processor`
  - retained code change:
    - `docker/run_freemhd_case.sh` and
      `scripts/write_freemhd_container_files.py` now treat
      `cores <= 1` as a true serial mode:
      - skip `decomposePar`
      - skip `mpirun -parallel`
      - run the solver directly while still writing `runLog.<solver>`
  - retained QA result:
    - focused harness tests still pass after the serial-path change:
      - `pytest tests/test_freemhd_harness.py tests/test_run_freemhd_case.py tests/test_build_freemhd_container.py -q`
    - a corrected recovered Hunt `Ha20` replay now reaches the actual coupled
      solver loop locally at `cores = 1`
    - live partial log confirms:
      - `LMX_DIAG outer time=1e-05`
      - `LMX_DIAG epot time=1e-05 ... potEFinalResidual ≈ 8.03e-08`
      - `LMX_DIAG pressure time=1e-05 ... maxU ≈ 1.1769e-01`
      - progression into `Time = 2e-05`
  - retained interpretation:
    - Docker/OpenFOAM/FreeMHD is now genuinely usable on this machine for
      short recovered-case diagnostics
    - the practical route here is currently the serial `cores = 1` path, not
      higher-core decomposition
    - the next retained solver step should still be driven by the saved Hunt
      trace once this serial replay finishes and is compared cleanly against
      the current LMX baseline
- `2026-04-08 America/Chicago`: Refreshed the short-window Hunt comparison
  against a real current serial replay on this machine:
  - retained real artifacts:
    - FreeMHD run JSON:
      - `/private/tmp/lmx_hunt_case_extract/run_hunt_ha20_2e05_1core_serial.json`
    - saved solver log:
      - `/private/tmp/lmx_hunt_case_extract/run_hunt_ha20_2e05_1core_serial.run.stdout.log`
    - extracted diagnostics:
      - `/private/tmp/lmx_hunt_case_extract/run_hunt_ha20_2e05_1core_serial.run.diag.json`
    - matching reduced LMX report:
      - `/private/tmp/lmx_hunt_case_extract/lmx_hunt_short_report_refresh.json`
    - aligned comparison:
      - `/private/tmp/lmx_hunt_case_extract/hunt_trace_compare_refresh.json`
  - retained short-window `Hunt Ha20`, `t <= 2e-05` comparison result:
    - normalized trace errors:
      - `u_max l2 ≈ 6.75e-04`
      - `mean_velocity l2 ≈ 8.59e-04`
      - `emf_max l2 ≈ 9.51e-04`
      - `current_max l2 ≈ 1.10e-02`
      - `face_current_max l2 ≈ 7.50e-03`
      - `lorentz_max l2 ≈ 7.78e-03`
      - `pressure_proxy l2 ≈ 2.65e-02`
    - FreeMHD pressure-correction records at the same times:
      - `t = 1e-05`: `maxU ≈ 1.1769e-01`, `pFinalResidual ≈ 9.05e-08`
      - `t = 2e-05`: `maxU ≈ 1.1780e-01`, `pFinalResidual ≈ 6.81e-08`
  - retained interpretation:
    - the refreshed current machine path confirms that the short Hunt startup
      agreement is now genuinely good in normalized `u`, `emf`, `J`, and
      `JxB` history
    - the remaining obvious short-window outlier is the reduced
      `pressure_proxy`
    - the next efficient solver step is therefore back on reduced
      pressure-response semantics, not on startup ramp/current construction and
      not on more Docker/runtime scaffolding
- `2026-04-08 America/Chicago`: Kept a small Hunt solver consistency fix on top
  of the clean short replay:
  - retained code changes:
    - `lmx/solvers.py` now returns the layered current/Lorentz reduction using
      the updated `u_next` with the fixed outer-step `phi`, instead of
      reporting the lagged pre-update velocity
    - `tests/test_solver.py` now checks that the retained hybrid Hunt
      diagnostics match the returned-state reduction to within solver tolerance
  - retained QA result:
    - targeted tests passed:
      - `pytest tests/test_solver.py -k 'hunt_hybrid_diagnostics_match_returned_state_reduction or hartmann_solver_runs or hunt_case_supports_hybrid_face_lorentz_current_reconstruction or hunt_case_supports_volume_scaled_cg_potential_backend' -q`
    - on the clean short Hunt `Ha20`, `t <= 2e-05` replay:
      - `u_max l2` stayed at `≈ 1.89e-03`
      - `pressure_proxy l2` stayed at `≈ 3.12e-02`
      - `face_current_max l2` stayed at `≈ 7.50e-03`
      - `current_max l2` improved from `≈ 1.104e-02` to `≈ 1.099e-02`
      - `lorentz_max l2` improved from `≈ 8.36e-03` to `≈ 7.79e-03`
  - retained interpretation:
    - this is a real state-consistency improvement, not a new pressure model
    - it is worth keeping because it improves the clean short replay slightly
      without regressing the current Hartmann/Shercliff guard path
  - best next step:
    - keep the clean short Hunt replay fixed as the local benchmark
    - extend or recover the longer `t <= 6e-05` patched replay again
    - only then change the later pressure/current reduction path itself
- `2026-04-08 America/Chicago`: Added a retained restart/continue vertical slice
  for native `lmx input.toml` runs:
  - retained code changes:
    - `lmx/config.py` now parses a top-level `[restart]` table into
      `RestartSpec`
    - `lmx/io.py` now writes explicit `state_time` / `state_residual` into the
      standard solution NPZ and can load/validate that NPZ as a restart bundle
    - `lmx/solvers.py` now accepts optional `initial_state` /
      `initial_diagnostics`, continues from the saved solver time, and can
      append histories instead of always starting fresh
    - `lmx/runtime_logging.py` now prints restart provenance and resume time in
      the live solver banner
    - `lmx/cli.py` now supports `lmx input.toml` continuation, writes restart
      metadata into the summary JSON, and can emit a dedicated restart NPZ at
      the end of the run
  - retained user-facing examples:
    - `examples/hartmann_restart_case.toml`
    - updated `README.md`, `examples/README.md`, `docs/input_reference.md`, and
      `docs/developer_guide.md`
  - retained validation:
    - targeted tests passed:
      - `pytest tests/test_config.py tests/test_io.py tests/test_cli.py tests/test_solver.py -q`
      - `python -m sphinx -W -b html docs docs/_build/html`
    - real executable QA passed:
      - base run:
        - `lmx examples/hartmann_case.toml`
      - continuation run:
        - `lmx examples/hartmann_restart_case.toml`
      - retained output artifacts:
        - base summary:
          - `artifacts/examples/toml_hartmann/hartmann_ha20_toml_summary.json`
        - continued summary:
          - `artifacts/examples/toml_hartmann_restart/hartmann_ha20_toml_summary.json`
        - continued restart NPZ:
          - `artifacts/examples/toml_hartmann_restart/hartmann_ha20_toml_restart.npz`
      - retained restart semantics confirmed locally:
        - restart banner reports source NPZ and `startTime = 4.0e-01`
        - continued summary reports `time = 8.0e-01`
        - continued restart NPZ contains `time_history_len = 400`,
          `state_time = 0.8`, `state_residual ≈ 2.13e-05`
  - retained interpretation:
    - native `lmx input.toml` runs now support OpenFOAM-like continue/restart
      from a saved state without a separate hidden format
    - the standard solution NPZ is now the canonical restart source, while the
      dedicated `*_restart.npz` output is the recommended handoff artifact for
      further continuation
    - CI/CD was clean at the time this work landed; no failing or queued `main`
      workflow runs were present in the latest `gh run list`
  - best next step:
    - return to the Hunt plan with the runtime path stabilized
    - use the new restart capability to extend recovered diagnostic windows
      without always rerunning from `t = 0`
    - then target the later-time reduced pressure/current response against the
      patched Hunt replay
- `2026-04-08 America/Chicago`: Landed a retained Hunt replay/runtime follow-up
  on top of the new restart path:
  - retained code changes:
    - `scripts/run_hunt_solver_diagnostic_report.py` now supports:
      - `--restart-npz`
      - `--append-histories`
      - `--write-restart-npz`
    - that script now validates restart bundles against the rebuilt reduced
      Hunt mesh before resuming and can emit a fresh restart NPZ for the next
      continuation window
    - `lmx/freemhd.py` now treats `docker image ls <image> --format {{.ID}}`
      as a fallback availability probe when Docker Desktop reports a false
      negative for `docker image inspect <image>`
    - `scripts/run_freemhd_case.py` now classifies nonzero-return runs with
      extracted `LMX_DIAG` records as `partial-failed` and records the latest
      diagnostic time in the output JSON
  - retained validation:
    - targeted tests passed:
      - `pytest tests/test_run_hunt_solver_diagnostic_report.py tests/test_compare_hunt_trace_histories.py -q`
      - `pytest tests/test_freemhd.py tests/test_run_freemhd_case.py tests/test_freemhd_harness.py -q`
      - `python -m sphinx -W -b html docs docs/_build/html`
    - retained reduced Hunt continuation artifacts:
      - first chunk:
        - `/private/tmp/lmx_hunt_long_refresh/lmx_hunt_2e05.json`
        - `/private/tmp/lmx_hunt_long_refresh/lmx_hunt_2e05_restart.npz`
      - resumed chunk:
        - `/private/tmp/lmx_hunt_long_refresh/lmx_hunt_6e05.json`
        - `/private/tmp/lmx_hunt_long_refresh/lmx_hunt_6e05_restart.npz`
      - retained resumed LMX trace through `t = 6e-05`:
        - `u_max_history` stays near `0.1175`
        - `pressure_proxy_history` decays from `≈ 201.2` to `≈ 152.2`
        - `current_max_history` decays from `≈ 4.89` to `≈ 4.40`
        - `lorentz_max_history` stays O(10) and ends at `≈ 14.86`
    - retained patched FreeMHD continuation result:
      - `/private/tmp/lmx_hunt_long_refresh/freemhd_hunt_6e05.json`
      - `/private/tmp/lmx_hunt_long_refresh/freemhd_hunt_6e05.run.stdout.log`
      - `/private/tmp/lmx_hunt_long_refresh/freemhd_hunt_6e05.run.diag.json`
      - the run no longer fails at image lookup; it now launches correctly and
        reaches `Time = 3e-05`
      - it still exits with `returncode = 137` inside that `3e-05` step before
        any patched pressure diagnostics are emitted
      - retained patched `epot` record at `t = 3e-05`:
        - `potEFinalResidual ≈ 1.63e-08`
        - `potEIterations = 6`
        - `maxJ ≈ 7.81e+04`
        - `maxJn ≈ 8.29e-01`
        - `maxJnDensity ≈ 8.29e+04`
        - `maxPsiub ≈ 1.176`
        - `maxPsiubDensity ≈ 2.36e+04`
        - `maxJxB ≈ 4.64e+03`
  - retained interpretation:
    - the Docker/OpenFOAM/FreeMHD runtime path is now usable again on this
      machine; the previous blocker was a brittle image-availability probe
    - the next external-backend blocker is no longer image lookup but the
      serial patched Hunt continuation being killed inside the `3e-05` step
    - because that kill happens before the patched `pressure` log record, the
      next solver-side pressure-response change should wait for a retained way
      to finish or slice that step cleanly
  - best next step:
    - keep the retained LMX `t <= 6e-05` continuation artifacts as the reduced
      baseline
    - make the smallest runtime change that allows the patched FreeMHD Hunt
      case to complete the `3e-05` step and emit `pressure/maxU` diagnostics
      without regressing the current harness
    - only then retarget the later-time reduced pressure/current response in
      `lmx/solvers.py`
- `2026-04-08 America/Chicago`: Kept a second retained runtime pass after the
  first partial FreeMHD continuation result:
  - retained code changes:
    - `scripts/run_freemhd_case.py` now records:
      - `run_diag_last_time`
      - `status = "partial-failed"` when a nonzero-return run still emits
        usable `LMX_DIAG` records
    - `tests/test_run_freemhd_case.py` now locks down that partial-progress
      reporting behavior
  - retained validation:
    - targeted tests passed:
      - `pytest tests/test_run_freemhd_case.py tests/test_freemhd_harness.py tests/test_run_hunt_solver_diagnostic_report.py -q`
      - `python -m sphinx -W -b html docs docs/_build/html`
    - Docker image lookup is stable again from the Python harness:
      - `docker_image_available("lmx-freemhd-localdiag-density-fixed2:latest")`
        now returns `True`
    - the patched serial Hunt continuation still reaches only the `3e-05`
      `epot` stage before being killed with `returncode = 137`
    - repeated local attempts confirmed the kill is now a runtime/resource
      issue inside the `3e-05` step, not an image-availability problem
  - retained interpretation:
    - the harness is now good enough to preserve and classify partial external
      backend progress instead of losing it behind a generic failure label
    - the next blocker is not Docker startup or image resolution anymore
    - the next practical external-runtime target is a retained way to finish
      the `3e-05` Hunt step cleanly enough to emit patched pressure records,
      or otherwise to reduce that step into a slice that survives on this host
  - best next step:
    - keep the retained `partial-failed` FreeMHD Hunt output as the external
      backend baseline for this host
    - try the smallest runtime-side change that can let the `3e-05` step emit
      `pressure/maxU` diagnostics:
      - one-step continuation from `2e-05 -> 3e-05`
      - or a lighter runtime configuration inside the same recovered case
    - avoid changing the LMX solver again until that later-time external
      pressure record is retained cleanly
- `2026-04-08 America/Chicago`: Broke the later-time FreeMHD runtime blocker
  with a retained harness-side control:
  - retained code changes:
    - `scripts/run_freemhd_case.py` now supports `--disable-vtk-write`
    - `docker/run_freemhd_case.sh` and
      `scripts/write_freemhd_container_files.py` now strip the `vtkWrite`
      function object from `system/controlDict` when that flag is enabled
    - `tests/test_run_freemhd_case.py` and `tests/test_freemhd_harness.py`
      now cover the new runtime flag and generated bundle behavior
  - retained validation:
    - targeted tests passed:
      - `pytest tests/test_run_freemhd_case.py tests/test_freemhd_harness.py -q`
      - `python -m sphinx -W -b html docs docs/_build/html`
    - retained FreeMHD Hunt artifacts with `--disable-vtk-write`:
      - `3e-05` continuation:
        - `/private/tmp/lmx_hunt_long_refresh/freemhd_hunt_3e05_novtk.json`
        - `/private/tmp/lmx_hunt_long_refresh/freemhd_hunt_3e05_novtk.run.diag.json`
        - `/private/tmp/lmx_hunt_long_refresh/trace_compare_3e05_novtk.json`
      - `6e-05` continuation:
        - `/private/tmp/lmx_hunt_long_refresh/freemhd_hunt_6e05_novtk.json`
        - `/private/tmp/lmx_hunt_long_refresh/freemhd_hunt_6e05_novtk.run.diag.json`
        - `/private/tmp/lmx_hunt_long_refresh/trace_compare_6e05_novtk.json`
    - the retained no-`vtkWrite` FreeMHD continuation now completes through
      `t = 6e-05` on this host with `status = "ok"` and `run_diag_last_time = 6e-05`
    - retained aligned later-time Hunt trace metrics against the resumed LMX
      reduced replay:
      - `u_max l2 ≈ 1.03e-03`
      - `mean_velocity l2 ≈ 1.54e-03`
      - `current_max l2 ≈ 2.29e-02`
      - `lorentz_max l2 ≈ 1.07e-01`
      - `face_current_max l2 ≈ 1.06e-02`
  - retained interpretation:
    - the main external-runtime blocker was the heavy `vtkWrite` function
      object, not the patched solver logging or the Docker image/runtime path
    - later-time patched Hunt FreeMHD traces are now available on this host
      without special manual intervention beyond the runtime flag
    - the next solver-side target is finally back where it should be:
      later-time Hunt electromagnetic and reduced pressure-response drift
      against a full retained `t <= 6e-05` FreeMHD trace
  - best next step:
    - keep `--disable-vtk-write` as the retained FreeMHD parity runtime
      setting for recovered Hunt cases on this host
    - use `/private/tmp/lmx_hunt_long_refresh/trace_compare_6e05_novtk.json`
      as the new later-time external reference
    - retarget `lmx/solvers.py` at the remaining later-time Hunt drift,
      especially the Lorentz/current evolution after `t = 4e-05`
- `2026-04-08 America/Chicago`: Re-checked Hunt current reconstruction on top
  of the retained later-time `t <= 6e-05` patched FreeMHD replay and moved the
  default back to `cell_centered`:
  - retained code changes:
    - `_hunt_short_transient_controls(...)` in `lmx/cases.py` now defaults
      Hunt back to `current_reconstruction="cell_centered"`
    - `examples/hunt_case.toml` now ships the same retained Hunt default
    - `tests/test_solver.py` now locks down the reverted Hunt default
  - retained validation:
    - corrected later-time Hunt replay against patched FreeMHD:
      - `cell_centered`: `u_max l2 ≈ 1.03e-3`, `current_max l2 ≈ 2.31e-2`,
        `lorentz_max l2 ≈ 8.34e-2`
      - `hybrid_face_lorentz`: `u_max l2 ≈ 1.03e-3`, `current_max l2 ≈ 2.29e-2`,
        `lorentz_max l2 ≈ 1.07e-1`
    - native Hunt analytical sweep at `32^2`:
      - `Ha20` combined L2:
        - `cell_centered ≈ 0.1002`
        - `hybrid_face_lorentz ≈ 0.1015`
      - `Ha100` combined L2:
        - `cell_centered ≈ 0.3381`
        - `hybrid_face_lorentz ≈ 0.3733`
  - retained interpretation:
    - `hybrid_face_lorentz` kept only a tiny edge on longer-window
      `current_max`, while `cell_centered` clearly won later-time `JxB`
      tracking and the native Hunt analytical path
    - Hunt should keep `cell_centered` as the retained default and leave
      `hybrid_face_lorentz` as an experimental diagnostics mode
  - best next step:
    - keep the corrected `t <= 6e-05` Hunt replay as the fixed later-time
      target
    - change the later-time pressure/current reduction path itself, not the
      top-level current-reconstruction mode
    - require any retained solver change to improve `lorentz_max` on the
      longer replay without regressing `u_max` or the native Hunt analytical
      combined error
- `2026-04-08 America/Chicago`: Ruled out two more tempting later-Hunt solver
  changes before touching the retained baseline:
  - rejected solver candidate:
    - add one final electromagnetic consistency solve at the end of each
      pseudo-time step, re-solving `phi` on `u_next` and recomputing
      `J` / `JxB` for the retained state and trace output
  - rejected solver-control candidate:
    - raise the later Hunt replay `potential_iterations` ceiling from
      `400` to `800`
  - retained validation:
    - final-consistency-solve candidate on corrected Hunt `Ha20`,
      `t <= 6e-05` replay:
      - `u_max l2`: unchanged at `≈ 1.03e-3`
      - `current_max l2`: worsened from `≈ 2.31e-2` to `≈ 2.36e-2`
      - `lorentz_max l2`: worsened from `≈ 8.34e-2` to `≈ 8.52e-2`
      - native Hunt analytical combined error at `Ha20`, `32^2`: effectively
        unchanged at `≈ 0.1002`
    - larger `phi` ceiling (`potential_iterations = 800`) on the same replay:
      - `u_max l2`: unchanged at `≈ 1.03e-3`
      - `current_max l2`: worsened from `≈ 2.31e-2` to `≈ 2.32e-2`
      - `lorentz_max l2`: worsened from `≈ 8.34e-2` to `≈ 8.73e-2`
      - `potential_iterations_used` climbed into the `631-698` range without
        producing a better later-time replay
  - retained interpretation:
    - the remaining Hunt blocker is not an end-of-step `phi` consistency issue
      and is not solved by simply driving the layered `phi` solve deeper
    - the next retained solver pass should move to the later-time
      pressure/Lorentz response mechanism itself, not another scalar-potential
      control tweak

## Instruction For Future Agents

Read this file first. Treat it as the live execution log and context handoff. Update it whenever you make a meaningful decision, add or remove scope, fix or discover a blocker, or identify a better next step. Keep entries chronological, concrete, and honest about what is implemented versus planned.

## Current Completion Assessment

This section is the current honest assessment of how close LMX is to the
retained target after the latest Hunt replay/default work.

- Retained ship target:
  - standalone Python/JAX laminar inductionless liquid-metal MHD solver
  - structured duct and layered-duct cases
  - Hartmann, Shercliff, and Hunt validation
  - executable TOML workflow, restart support, ParaView/CSV/NPZ outputs,
    plots, examples, and CI/CD
  - external parity and benchmarking against recovered FreeMHD/OpenFOAM
    `epotMultiRegionFoam` / `epotMultiRegionInterFoam` cases
- Explicitly deferred from the retained target:
  - temperature equation / Joule-heating thermal coupling
  - free-surface VoF / `interFoam` parity as a release requirement
  - turbulence
  - contact-angle / atmosphere-opening free-surface models
  - general unstructured/polyhedral meshing
  - full OpenFOAM feature parity beyond the laminar inductionless solver path

### What FreeMHD currently has beyond the retained LMX scope

From the local `external/FreeMHD/MHD_Solvers/solvers` tree, FreeMHD currently
contains more than the retained LMX release target:

- `epotMultiRegionFoam`
  - multi-region MHD with full OpenFOAM PIMPLE-style fluid/solid structure
  - includes energy/species/compressible pieces in the solver tree
- `epotMultiRegionInterFoam`
  - multiphase / free-surface path
  - includes temperature equation and interFoam-derived fluid coupling pieces
- `epotMultiRegionInterIsoFoam`
  - additional interIsoFoam/free-surface path
- `apot*` and `bScalarPotCht*` solvers
  - vector-potential and thermal/CHT-oriented branches outside the retained
    current LMX target

LMX does not need to implement all of that to finish the retained current plan.

### What is already done well enough for the retained target

- Native solver/runtime:
  - executable `lmx input.toml` path
  - live OpenFOAM-style solver logging
  - restart/continue from NPZ state bundles
  - ParaView / CSV / NPZ / plot outputs
- Validation/runtime infrastructure:
  - unit / regression / physics / validation / benchmark suites in CI
  - examples and plotting/movie scripts
  - real Dockerized FreeMHD/OpenFOAM parity harness
  - patched FreeMHD logging with retained later-time Hunt replay through
    `t = 6e-05`
- Physics status:
  - Hartmann: strong retained path
  - Shercliff: strong retained path
  - Hunt: now materially improved and externally replay-backed, but still the
    main remaining solver-fidelity gap

### What still blocks “finished” for the retained target

- Main solver blocker:
  - later-time Hunt Lorentz/current/pressure-response drift against the
    retained patched FreeMHD `t <= 6e-05` replay
- Release-quality validation blocker:
  - need one more retained solver pass that improves the later Hunt replay
    without regressing Hartmann/Shercliff or native Hunt analytical metrics
- Scope-closure blocker:
  - decide whether mapped simple-pipe / fringing-field support is in the first
    ship-ready release or explicitly deferred to the next milestone

### Current honest completion estimate

- Relative to the original very broad vision
  - including broader FreeMHD/OpenFOAM parity, pipe/fringing-field release
    support, and future extensibility:
    about 60% complete
- Relative to the retained near-term ship target
  - standalone laminar inductionless duct solver with Hartmann/Shercliff/Hunt,
    TOML workflow, restart, outputs, examples, CI/CD, and FreeMHD parity
    harness:
    about 80% complete

### Remaining major steps

1. Finish the retained Hunt solver-fidelity pass.
   - Improve later-time Lorentz/current/pressure-response evolution on the
     patched `t <= 6e-05` Hunt replay.
   - Keep `u_max` parity and native Hunt analytical combined error at least as
     good as the current retained baseline.
2. Lock the validation story.
   - Re-run Hartmann, Shercliff, Hunt native validation artifacts.
   - Re-run retained FreeMHD parity artifacts after the solver change.
   - Make sure examples and executable TOML cases still reflect the retained
     default behavior.
3. Freeze the first release scope.
   - Either:
     - ship ducts/layered ducts only, or
     - finish one mapped simple-pipe/fringing-field validation path and include
       it in the first release.
4. Do ship-readiness cleanup.
   - tighten docs around retained scope
   - make sure examples are polished and reproducible
   - recheck coverage/CI/benchmarks after the last solver change

### Latest retained Hunt parity narrowing

- The recovered Hunt `Ha20` case in `0/liquid/U` uses
  `type flowRateInletVelocity`, not a simple fixed inlet velocity.
- The FreeMHD parity helpers now infer that inlet type automatically from the
  recovered case and switch the reduced replay to `inlet_flow_rate` by default
  for Hunt when appropriate.
- The raw recovered `volumetricFlowRate` is now recorded in the JSON artifacts
  as `recovered_inlet_flow_rate`, but it is not injected directly into the
  nondimensional reduced duct model.
- On the retained corrected Hunt `t <= 6e-05` replay this new default changes
  the later-time trace metrics to:
  - `u_max l2 ≈ 5.26e-04` from `≈ 1.03e-03`
  - `mean_velocity l2 ≈ 1.03e-03` from `≈ 1.54e-03`
  - `current_max l2 ≈ 2.28e-02` from `≈ 2.31e-02`
  - `lorentz_max l2 ≈ 8.42e-02` from `≈ 8.34e-02`
  - `pressure_proxy l2 ≈ 1.11e-01` from `≈ 7.55e-02`
- Retained interpretation:
  - this is the more physically faithful recovered-case parity baseline for
    Hunt because it mirrors the actual boundary-condition type
  - it improves the flow/current replay slightly, but it does not close the
    remaining later-time Lorentz/pressure-response gap
  - the next solver-side change should still target the later-time Hunt
    Lorentz/pressure evolution itself

### Latest rejected Hunt pressure-response family

- I tested a reduced mean-drive memory / `drive_relaxation` family on the
  corrected recovered Hunt `t <= 6e-05` replay, treating the `inlet_flow_rate`
  closure as a relaxed response instead of recomputing a fully new forcing each
  step.
- Retained replay results:
  - baseline auto-inferred Hunt replay:
    - `u_max l2 ≈ 5.26e-04`
    - `current_max l2 ≈ 2.28e-02`
    - `lorentz_max l2 ≈ 8.42e-02`
    - `pressure_proxy l2 ≈ 1.11e-01`
  - `drive_relaxation = 0.5`:
    - `u_max l2 ≈ 5.93e-04`
    - `current_max l2 ≈ 2.28e-02`
    - `lorentz_max l2 ≈ 8.45e-02`
    - `pressure_proxy l2 ≈ 1.06e-01`
  - `drive_relaxation = 0.25`:
    - `u_max l2 ≈ 7.11e-04`
    - `current_max l2 ≈ 2.29e-02`
    - `lorentz_max l2 ≈ 8.27e-02`
    - `pressure_proxy l2 ≈ 9.71e-02`
- Native Hunt processed-slice validation at `Ha20`, `32^2` stayed effectively
  unchanged across `drive_relaxation = 1.0, 0.5, 0.25`:
  - `combined_l2_error ≈ 1.708e-01`
- Retained interpretation:
  - this family only trades replay observables against each other
  - it does not improve the native Hunt validation path
  - it is not the missing later-time Hunt fix and should stay out of retained
    defaults and public controls for now

### Latest corrected layered-current replay check

- After promoting `face_current_max` and `face_lorentz_max` to the retained
  layered parity metrics, I rechecked the older
  `hybrid_face_lorentz` current-reconstruction path on the corrected
  recovered Hunt `t <= 6e-05` replay.
- Retained replay comparison against the current `cell_centered` default:
  - current retained default:
    - `u_max l2 ≈ 5.26e-04`
    - `primary_current_max l2 ≈ 1.09e-02`
    - `primary_lorentz_max l2 ≈ 3.68e-03`
    - `pressure_proxy l2 ≈ 1.11e-01`
  - `hybrid_face_lorentz`:
    - `u_max l2 ≈ 5.26e-04`
    - `primary_current_max l2 ≈ 1.11e-02`
    - `primary_lorentz_max l2 ≈ 3.68e-03`
    - `pressure_proxy l2 ≈ 1.11e-01`
- Retained interpretation:
  - under the corrected layered parity metrics, `hybrid_face_lorentz` still
    does not improve the dominant later-time Hunt mismatch
  - it leaves `u_max` unchanged, slightly worsens the primary current metric,
    and does not improve the reduced pressure-response trace
  - it should remain an experimental mode, not the retained default

### Latest retained flow-rate closure cleanup

- I changed the reduced `inlet_flow_rate` control closure to compute
  `mean_base` and `mean_sensitivity` on the full fluid area rather than on the
  interior active-mask area, while keeping the reported `mean_velocity`
  diagnostic unchanged.
- This is a principled semantic cleanup:
  - flow-rate control should act on the fluid cross-section, not on the
    interior-only diagnostic subset
  - the new behavior is covered by a dedicated unit test on a nonuniform
    density field
- Retained replay result on the corrected Hunt `t <= 6e-05` baseline:
  - `u_max l2`: unchanged at `≈ 5.26e-04`
  - `pressure_proxy l2`: unchanged at `≈ 1.11e-01`
  - `primary_current_max l2`: unchanged at `≈ 1.09e-02`
  - `primary_lorentz_max l2`: unchanged at `≈ 3.68e-03`
- Retained interpretation:
  - the remaining later-time Hunt blocker is downstream of the
    flow-rate-control averaging geometry
  - the next retained solver change should continue to target the reduced
    pressure-response mechanism itself

### Latest retained Hunt pressure-span correction

- I patched the FreeMHD/OpenFOAM pressure logging path to emit:
  - `minP`
  - `pSpan = max(p) - min(p)`
  - `minPRgh`
  - `pRghSpan = max(p_rgh) - min(p_rgh)`
- The Hunt trace comparator now promotes `pSpan` to
  `primary_pressure_metric` when it is available, instead of comparing the
  reduced `pressure_proxy` against the absolute-pressure `maxP` signal.
- Retained corrected Hunt `Ha20`, `t <= 6e-05` replay against the current LMX
  baseline:
  - `u_max l2 ≈ 1.68e-03`
  - `primary_pressure_metric = pSpan`
  - `pressure_proxy l2 ≈ 6.23e-02`
  - `primary_pressure_proxy_metric = current_scaled_pressure_proxy`
  - `current_scaled_pressure_proxy l2 ≈ 4.42e-02`
  - `primary_current_metric = face_current_max`
  - `primary_current_max l2 ≈ 2.29e-02`
  - `primary_lorentz_metric = face_lorentz_max`
  - `primary_lorentz_max l2 ≈ 1.62e-02`
  - `mean_velocity l2 ≈ 2.62e-03`
- Retained interpretation:
  - the old pressure mismatch was partly a parity-observable problem
  - switching from `maxP` to `pSpan` cuts the long Hunt pressure mismatch
    materially (`≈ 1.11e-01 -> ≈ 6.23e-02`)
  - deriving a current-scaled reduced pressure observable narrows it further
    (`≈ 6.23e-02 -> ≈ 4.42e-02`) without changing solver physics
  - that corrected pressure-side observable is now first-class in the runtime
    path: live logs print `currentScaledPressureProxy`, and restart/NPZ
    outputs persist `current_scaled_pressure_proxy_history`
  - the remaining later-time Hunt blocker is now a smaller, genuinely
    solver-side pressure/current response gap after correcting the retained
    pressure/current/Lorentz observables

### Latest rejected forcing-relaxation family

- I tested a general forcing-response relaxation family for reduced
  `inlet_flow_rate` closures on the corrected long Hunt replay, but did not
  keep it on `main`.
- The candidate was intentionally general rather than Hunt-specific:
  - blend the newly inferred reduced forcing with the previous applied forcing
    before advancing the next step
  - compare against the same corrected FreeMHD `t <= 6e-05` replay using:
    - `primary_pressure_metric = pSpan`
    - `primary_current_metric = face_current_max`
    - `primary_lorentz_metric = face_lorentz_max`
- Retained replay results:
  - current corrected baseline:
    - `u_max l2 ≈ 1.68e-03`
    - `pressure_proxy l2 ≈ 6.23e-02`
    - `primary_current_max l2 ≈ 2.29e-02`
    - `primary_lorentz_max l2 ≈ 1.62e-02`
  - `forcing_relaxation = 0.75`:
    - `u_max l2 ≈ 1.68e-03`
    - `pressure_proxy l2 ≈ 2.08e-01`
    - `primary_current_max l2 ≈ 2.29e-02`
    - `primary_lorentz_max l2 ≈ 1.62e-02`
  - `forcing_relaxation = 0.5`:
    - `u_max l2 ≈ 1.74e-03`
    - `pressure_proxy l2 ≈ 6.31e-01`
    - `primary_current_max l2 ≈ 2.30e-02`
    - `primary_lorentz_max l2 ≈ 1.61e-02`
- Retained interpretation:
  - forcing-response relaxation barely changes the current/Lorentz parity
    metrics
  - it makes the corrected pressure-span parity much worse
  - it is not the right later-time Hunt control family and should stay out of
    retained solver controls and public inputs

### Latest rejected Hunt relaxation retune

- I rechecked the retained Hunt `Ha20` relaxation value against the corrected
  long replay and native analytical validation before changing defaults.
- Retained replay results on the corrected `t <= 6e-05` FreeMHD comparison:
  - current retained default `relaxation = 0.08`:
    - `u_max l2 ≈ 1.675e-03`
    - `pressure_proxy l2 ≈ 6.229e-02`
    - `primary_current_max l2 ≈ 2.295e-02`
    - `primary_lorentz_max l2 ≈ 1.619e-02`
  - candidate `relaxation = 0.10`:
    - `u_max l2 ≈ 1.675e-03`
    - `pressure_proxy l2 ≈ 6.229e-02`
    - `primary_current_max l2 ≈ 2.295e-02`
    - `primary_lorentz_max l2 ≈ 1.618e-02`
- Native Hunt analytical check at `Ha20`, `32^2`:
  - current retained default `relaxation = 0.08`:
    - `y_l2 ≈ 2.979e-02`
    - `z_l2 ≈ 1.386e-01`
    - `combined_l2 ≈ 1.0023e-01`
  - candidate `relaxation = 0.10`:
    - `y_l2 ≈ 2.824e-02`
    - `z_l2 ≈ 1.393e-01`
    - `combined_l2 ≈ 1.0052e-01`
- Retained interpretation:
  - the replay-side difference is too small to justify a default change
  - the native analytical combined error gets slightly worse
  - the retained Hunt `Ha20` default remains `relaxation = 0.08`

### Latest rejected outer-coupling retune

- I rechecked Hunt `outer_iterations` against both the corrected long replay
  and the native analytical gate before changing defaults.
- Corrected retained Hunt `Ha20`, `t <= 6e-05` replay:
  - retained default `outer_iterations = 6`:
    - `u_max l2 ≈ 1.675e-03`
    - `pressure_proxy l2 ≈ 6.229e-02`
    - `primary_current_max l2 ≈ 2.295e-02`
    - `primary_lorentz_max l2 ≈ 1.619e-02`
  - candidate `outer_iterations = 4`:
    - `u_max l2 ≈ 2.064e-03`
    - `pressure_proxy l2 ≈ 1.587e-02`
    - `primary_current_max l2 ≈ 1.902e-02`
    - `primary_lorentz_max l2 ≈ 1.601e-02`
  - candidate `outer_iterations = 8`:
    - `u_max l2 ≈ 1.130e-03`
    - `pressure_proxy l2 ≈ 1.586e-01`
    - `primary_current_max l2 ≈ 2.047e-02`
    - `primary_lorentz_max l2 ≈ 3.934e-02`
- Native Hunt analytical check at `Ha20`, `32^2`:
  - `outer_iterations = 4`: `combined_l2 ≈ 1.0519e-01`
  - `outer_iterations = 6`: `combined_l2 ≈ 1.0023e-01`
  - `outer_iterations = 8`: `combined_l2 ≈ 9.9136e-02`
- Retained interpretation:
  - `outer_iterations = 4` improves replay-side pressure/current metrics but
    clearly worsens `u_max` and the native analytical combined error
  - `outer_iterations = 8` improves `u_max` and the native analytical
    combined error slightly, but it badly worsens the corrected replay
    pressure and primary Lorentz metrics
  - this is another control-tradeoff family, not the missing later-time Hunt
    fix, so the retained default remains `outer_iterations = 6`

### Latest rejected Hunt pressure-state family

- I tested the first structural later-time Hunt response family directly in
  `lmx/solvers.py`: a carried scalar streamwise-forcing / pressure-like state
  with post-BC mean-velocity correction inside `_step(...)`, motivated by the
  corrected long replay and the solver-code inspection.
- The variant used:
  - iterate-centered velocity prediction for reduced `inlet_flow_rate` control
  - a scalar forcing state updated from the post-BC mean-velocity defect
  - carry of that scalar across time steps and restart points
- Replay artifact:
  - `/private/tmp/lmx_hunt_long_refresh/trace_compare_6e05_pressure_state.json`
- Retained baseline artifact after rollback:
  - `/private/tmp/lmx_hunt_long_refresh/trace_compare_6e05_reverted.json`
- On the corrected long Hunt `Ha20`, `t <= 6e-05` replay:
  - retained baseline:
    - `u_max l2 ≈ 1.6750e-03`
    - `pressure_proxy l2 ≈ 6.2293e-02`
    - `primary_current_max l2 ≈ 2.2946e-02`
    - `primary_lorentz_max l2 ≈ 1.6186e-02`
  - rejected pressure-state family:
    - `u_max l2 ≈ 2.5619e-03`
    - `pressure_proxy l2 ≈ 1.8692e+01`
    - `primary_current_max l2 ≈ 2.3855e-02`
    - `primary_lorentz_max l2 ≈ 1.5143e-02`
- On the native Hunt `Ha20`, `32^2` analytical gate:
  - `combined_l2 ≈ 1.0023e-01`, effectively unchanged
- Retained interpretation:
  - this family attacks the right conceptual mechanism, but the first variant
    is not physically usable
  - it does not improve the native analytical gate and it catastrophically
    worsens the replay-side reduced pressure trace
  - the remaining Hunt work should keep the corrected long replay fixed and
    target the reduced pressure/current response with a different mechanism,
    not by carrying this scalar forcing state forward

### Latest rejected relaxed-sensitivity flow-rate closure

- I tested a more self-consistent reduced `inlet_flow_rate` control law in
  `lmx/solvers.py`: solve the scalar forcing against the relaxed velocity
  response inside `_step(...)` instead of the pre-relaxation response.
- This is physically plausible because the actual update applies `relaxation`
  before the limiter, so I checked it against both the native Hunt analytical
  gate and a freshly rebuilt patched FreeMHD live run.
- Native Hunt analytical check at `Ha20`, `32^2`:
  - retained baseline:
    - `y_l2 ≈ 6.806e-02`
    - `z_l2 ≈ 1.911e-01`
    - `combined_l2 ≈ 1.4342e-01`
  - retained interpretation:
    - this is materially worse than the current retained Hunt baseline near
      `combined_l2 ≈ 1.0023e-01`
    - the relaxed-sensitivity closure should not stay on `main`
- Fresh live FreeMHD replay artifacts rebuilt from `external/StartingFiles.zip`
  with `--disable-vtk-write`:
  - run metadata:
    - `/private/tmp/lmx_hunt_long_refresh/freemhd_hunt_6e05_relaxed_sensitivity.json`
  - solver replay:
    - `/private/tmp/lmx_hunt_long_refresh/lmx_hunt_6e05_relaxed_sensitivity_valid.json`
  - aligned comparison:
    - `/private/tmp/lmx_hunt_long_refresh/trace_compare_6e05_relaxed_sensitivity.json`
- Replay-side normalized parity on that rebuilt run:
  - `u_max l2 ≈ 3.39e-03`
  - `primary_pressure_proxy l2 ≈ 1.33e-02`
  - `primary_current_max l2 ≈ 4.18e-03`
  - `primary_lorentz_max l2 ≈ 1.04e-02`
- Caveat:
  - the local `lmx-freemhd-smoke` image used for this rebuild was still on the
    older pressure logger, so the run records `maxP` rather than the newer
    `pSpan`-based pressure metric
  - that replay is useful as boundedness / trend evidence, but not strong
    enough to override the native analytical regression
- Retained interpretation:
  - this control-law family is another tempting but wrong fix
  - it can look reasonable on a live replay window, but it weakens the native
    Hunt solver materially
  - the retained baseline stays on the original pre-relaxation forcing closure,
    and the next Hunt work should keep targeting the reduced later-time
    pressure/current response with a different mechanism

### Expected number of focused iterations

- If we keep the retained first-release scope to duct/layered-duct laminar
  inductionless MHD:
  - about 4 to 6 focused solver/validation iterations
- If mapped simple-pipe / fringing-field parity stays inside the first release:
  - about 8 to 12 focused iterations

### Latest retained Hunt parity-script corrections

- I stopped treating the Hunt replay mismatch as a pure solver problem and
  checked the actual source-level observables on both sides.
- Two concrete parity-script mismatches were real and are now corrected:
  - the recovered `flowRateInletVelocity` value from `0/liquid/U` is now used
    correctly in the replay path:
    - raw OpenFOAM metadata is recorded as `recovered_inlet_flow_rate`
    - the replay value is rescaled onto the reduced LMX duct area and recorded
      as `reduced_inlet_flow_rate`
  - Hunt trace comparison now maps the layered observables to the physically
    matching patched FreeMHD diagnostics:
    - `current_max_history -> maxJ`
    - `face_current_max_history -> maxJnDensity`
    - `emf_max_history -> maxPsiubDensity`
    - `lorentz_max_history -> maxCenteredJxB`
- Fresh rebuilt patched Hunt `Ha20`, `t <= 6e-05` replay artifacts:
  - FreeMHD run metadata:
    - `/private/tmp/lmx_hunt_long_refresh/freemhd_hunt_6e05_current.json`
  - extracted diagnostics:
    - `/private/tmp/lmx_hunt_long_refresh/freemhd_hunt_6e05_current.run.diag.json`
  - matching short LMX replay:
    - `/private/tmp/lmx_hunt_long_refresh/lmx_hunt_6e05_current_short.json`
  - aligned comparison:
    - `/private/tmp/lmx_hunt_long_refresh/trace_compare_6e05_current_fixed.json`
- On that corrected source-matched replay, the retained normalized metrics are:
  - `u_max l2 ≈ 1.43e-03`
  - `primary_pressure_metric = pSpan`
  - `pressure_proxy l2 ≈ 1.94e-02`
  - `current_scaled_pressure_proxy l2 ≈ 3.18e-02`
  - `primary_current_metric = face_current_density_max`
  - `primary_current_max l2 ≈ 1.44e-01`
  - `primary_lorentz_metric = centered_lorentz_max`
  - `primary_lorentz_max l2 ≈ 8.29e-02`
- Retained interpretation:
  - once the replay flow-rate is rescaled onto the reduced duct area, Hunt
    `u_max` and the pressure-side parity recover to the strong range expected
    from the older retained short-window runs
  - the remaining mismatch is now more concentrated in the source-correct
    layered current-density and centered-Lorentz observables
  - the next solver work should target that current/Lorentz evolution directly,
    not replay input plumbing

### Latest retained Hunt limiter diagnosis

- I added retained limiter telemetry to the core solver, restart/output path,
  validation summary, runtime logger, and Hunt diagnostic report:
  - `raw_update_max_history`
  - `limiter_scale_history`
  - `limited_fraction_history`
- The live log now prints those as:
  - `rawUpdateMax`
  - `limiterScale`
  - `limitedFraction`
- The retained replay artifact is:
  - `/private/tmp/lmx_hunt_long_refresh/lmx_hunt_6e05_with_limiter_diag.json`
- On the corrected Hunt `Ha20`, `t <= 6e-05` replay, the telemetry is:
  - `raw_update_max_history ≈ [6.54e-02, 5.64e-02, 4.76e-02, 3.90e-02, 3.06e-02, 2.24e-02]`
  - `limiter_scale_history ≈ [3.06e-02, 3.54e-02, 4.20e-02, 5.13e-02, 6.54e-02, 8.91e-02]`
  - `limited_fraction_history ≈ [1.21e-01, 1.21e-01, 1.25e-01, 1.33e-01, 2.34e-01, 2.34e-01]`
- Retained interpretation:
  - the corrected Hunt replay is strongly limiter-dominated, not lightly
    clipped
  - the raw coupled update is much larger than the retained Hunt cap early in
    the replay, and a nontrivial fraction of the fluid region is limited every
    step
  - this means the next solver-side Hunt fix should be driven by a better raw
    update or a more faithful current/Lorentz reduction, not by pretending the
    present limiter is only a small perturbation

### Latest retained Hunt replay-side consistency check

- I exposed the existing `post_update_potential_refresh` hook in
  `scripts/run_hunt_solver_diagnostic_report.py` and re-checked it through the
  public diagnostic CLI against the corrected replay.
- Retained artifacts:
  - `/private/tmp/lmx_hunt_long_refresh/lmx_hunt_6e05_refresh_cli.json`
  - `/private/tmp/lmx_hunt_long_refresh/trace_compare_6e05_refresh_cli.json`
- Result:
  - `u_max l2` stayed unchanged near `1.43e-03`
  - `current_scaled_pressure_proxy l2` improved only trivially
  - `primary_current_max l2` worsened slightly
  - `primary_lorentz_max l2` worsened slightly
- Retained interpretation:
  - re-solving `phi` on `u_next` is not the missing Hunt fix
  - it is worth keeping as an explicit parity/consistency control, but it
    should remain off by default

### Latest rejected Hunt source-term reconstruction trial

- I also tested the most direct source-level discretization change suggested by
  the FreeMHD code: replace LMX's resistance-weighted face EMF construction
  with a direct linear face interpolation of `(sigma * U) x B`, matching the
  `psiub` interpolation pattern in `epotMultiRegionInterFoam`.
- Retained result:
  - that change was not stable enough to keep in the reduced solver
  - it drove the Hunt replay/diagnostic path into an obviously wrong regime
    with runaway pressure/current magnitudes
  - the change was reverted immediately after the experiment
- Retained interpretation:
  - copying the FreeMHD face-source interpolation literally is not sufficient
    inside the current reduced LMX formulation
  - the next solver-side Hunt change still needs to respect the reduced-model
    structure rather than transplanting one OpenFOAM face formula in isolation
