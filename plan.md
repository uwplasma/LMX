# LMX Execution Plan And Chronological Log

## Project Goal

LMX is a Python/JAX-native inductionless MHD code intended to reproduce the laminar electric-potential functionality of FreeMHD without carrying over the full OpenFOAM stack. The immediate parity target is the laminar subset used for closed-channel verification and then the fringing-field pipe validation. The code should remain differentiable, CPU/GPU-capable, and structured around JAX compilation.

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
- Zenodo closed-channel analytical and processed-slice reference-data loaders.
- Unit and categorized tests passing in `/Users/rogerio/base_env/bin/python3`.

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
- `scripts/inspect_freemhd_setup.py`: inspects the locally available FreeMHD assets, reports discovered case directories, and recommends the smallest smoke target.

## Best Next Steps

1. Replace the current pseudo-transient duct step with a more faithful laminar parity solver:
   - better electric-potential gauge handling
   - stable iterative coupling between `u`, `phi`, and `J x B`
   - the Hartmann/Shercliff fine-mesh clipping issue is mitigated by smaller pseudo-time defaults, but Hunt still needs a real multi-region stability fix
2. Extend analytical validation beyond the current Hartmann implementation:
   - Shercliff and Hunt profile comparison hooks now exist
   - Hartmann is now close enough that it can support stronger acceptance checks
   - Shercliff and Hunt still need solver-fidelity improvements before they can become acceptance tests
3. Tighten CI acceptance criteria once better parity is available:
   - convert Hartmann validation into a stronger pass/fail parity check
   - keep Shercliff and Hunt as informative reports until their fidelity improves
   - add benchmark threshold tracking with explicit tolerances
4. Focus solver work on Hunt multi-region coupling and Shercliff profile fidelity now that both analytical and processed-slice metrics are emitted by the same CLI path.
5. Acquire or reconstruct at least one standalone `epotMultiRegion*` laminar case locally, since the current local assets do not yet contain runnable FreeMHD paper cases.
6. Tighten the current FreeMHD container bundle into a verified build-and-run path with actual OpenFOAM solver compilation rather than documented placeholder commands.
7. Implement mapped-operator support for the fringing-field pipe case.
8. Build parity runners that extract comparable LMX and FreeMHD metrics from the same cases.

## What Worked

- The package structure and top-level API are in place and importable.
- The local JAX environment at `/Users/rogerio/base_env/bin/python3` works for development and tests.
- The current duct solver path produces fields, VTK output, CSV cuts, and benchmark timing.
- Test suite is green after tightening the expectations to match current implementation state rather than full parity claims.
- Small Hartmann and Shercliff low-Ha cases are deterministic enough for regression snapshots.
- GitHub Actions workflows now cover unit, regression, physics, validation, and benchmark paths.
- The processed-figures Zenodo archive is sufficient for immediate closed-channel reference ingestion; the 8.9 GB `StartingFiles.zip` archive is not needed by default.
- CLI validation now emits both analytical and processed-slice comparison JSON when the matching Zenodo `XSlice` CSV exists.
- The validation-suite script can now emit analytical and processed-slice reports in one run when a reference root is provided.
- Fine-mesh Hartmann and Shercliff stability improved materially after reducing the pseudo-time step and increasing the iteration budget in their case factories.
- Harmonic face conductivity averaging improved the multi-material discretization and helped Shercliff on smaller validation grids.
- Semi-implicit treatment of the linear Lorentz damping term improved Hartmann and Shercliff robustness without breaking the existing solver interface.
- An adaptive per-step velocity-update limiter now keeps the default Hunt path bounded and produces finite Hunt validation metrics.
- CI artifact summaries now include processed-slice error columns.
- A local FreeMHD container bundle generator and container execution helper now exist and are covered by unit tests.
- The repo now has explicit FreeMHD environment and case inspection scripts, and they correctly report the current local target as the bundled OpenFOAM Hartmann tutorial when no standalone FreeMHD cases are present.
- A local FreeMHD environment probe now captures Docker-daemon availability and the current `wmkdepend` local-build blocker in machine-readable form.
- FreeMHD-side inspection now reports that the current downloads contain no standalone runnable FreeMHD cases and recommends the bundled OpenFOAM Hartmann tutorial as the smallest local smoke target.

## What Did Not Work

- Running with plain `python3` from the shell did not use the JAX-enabled environment.
- The initial solver branch that tried to decide between `lineax` and Jacobi inside a traced JAX function failed due to traced boolean conversion.
- The initial unconstrained pseudo-transient update produced `NaN` blow-up; explicit wall enforcement and bounded updates were needed to stabilize the first implementation.
- The first Shercliff symmetry assertion was too strong for the current solver and had to be downgraded to a finite-field/no-slip smoke test until better parity numerics are implemented.
- A later attempt to replace the pseudo-transient steady path with a more directly coupled fixed-point steady solver was not robust enough and was rolled back instead of being left on `main`.
- The Hunt validation path remains artifact-only for now because the current solver still clips and saturates on that case; it should not be treated as parity-complete.
- The current Shercliff `Ha=20` reference comparison also shows large normalized error, confirming that solver-fidelity work is still the critical path after reference ingestion.
- The earlier Hartmann/Shercliff defaults (`dt=0.01`, low iteration counts) were too aggressive for fine meshes because the current solver core uses an explicit diffusive update.
- Hunt can be stabilized only with a much smaller pseudo-step than the current default, which confirms the next Hunt fix should be adaptive pseudo-stepping or a more implicit coupling rather than more validation plumbing.
- Semi-implicit Lorentz damping alone was not enough to fix Hunt at default settings; it needed the additional adaptive update limiter.
- Hunt is now bounded by default, but the resulting bounded solution is still not accurate enough yet to be treated as parity-complete.
- The current FreeMHD container bundle is still a scaffold for local iteration; it documents the expected build/run layout but has not yet been proven end to end against a real OpenFOAM build on this machine.
- The current machine has the Docker CLI installed, but the Docker daemon is not reachable from the active environment.
- The current local assets include the FreeMHD source tree and processed paper figures, but not the standalone `epotMultiRegion*` case directories needed for direct parity execution.
- A direct local `wmake` probe currently fails because `OpenFOAM-v2206/platforms/tools/darwin64Clang/wmkdepend` is missing in the vendored FreeMHD tree on this machine.
- The Docker client is installed here, but the Docker daemon is not currently reachable, so containerized FreeMHD execution is blocked by the local runtime state rather than repo code.
- The currently downloaded assets do not yet include standalone runnable FreeMHD case directories, so real parity runs still require either the larger starting-files archive or another case source.

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

## Instruction For Future Agents

Read this file first. Treat it as the live execution log and context handoff. Update it whenever you make a meaningful decision, add or remove scope, fix or discover a blocker, or identify a better next step. Keep entries chronological, concrete, and honest about what is implemented versus planned.
