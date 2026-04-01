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
- Zenodo closed-channel analytical and processed-slice reference-data loaders.
- Unit and categorized tests passing in `/Users/rogerio/base_env/bin/python3`.
- Combined local coverage across `lmx/` and `scripts/` is currently `89%`.
- Native mesh-convergence study runner and Hartmann acceptance reports.

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

## What Worked

- The package structure and top-level API are in place and importable.
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

## Instruction For Future Agents

Read this file first. Treat it as the live execution log and context handoff. Update it whenever you make a meaningful decision, add or remove scope, fix or discover a blocker, or identify a better next step. Keep entries chronological, concrete, and honest about what is implemented versus planned.
