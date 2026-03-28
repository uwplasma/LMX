# Developer Guide

## Package structure

- `lmx.mesh`: grid builders.
- `lmx.specs`: immutable case/config dataclasses.
- `lmx.physics`: conductivity and magnetic-field field construction.
- `lmx.operators`: mesh-aware finite-volume style kernels.
- `lmx.solvers`: laminar inductionless solver entrypoints.
- `lmx.io`: ParaView XML and CSV outputs.
- `lmx.validation`: analytical helpers and FreeMHD harness.
- `lmx.reference_data`: Zenodo closed-channel analytical and processed-slice loaders.

## Array layout

- Cross-section fields use shape `(ny, nz)`.
- The solver currently models the streamwise velocity `u(y, z)` plus electric potential `phi(y, z)`.
- Conductivity and masks are stored on the same cell-centered layout.

## JAX strategy

- Solver stepping uses `jax.lax.scan` for stable fixed-step execution.
- Linear solves use `lineax` when available and otherwise fall back to a pure-JAX Jacobi solver.
- Keep shapes static when adding new operators or diagnostics.
- Hartmann and Shercliff case factories now use smaller pseudo-time steps and a larger iteration budget because the current solver core is explicit in the diffusive update and otherwise clips on fine meshes.
- The velocity update now treats the linear `-sigma |B|^2 u` portion of the Lorentz force semi-implicitly, which improves stability for Hartmann and Shercliff without changing the overall solver structure.
- The solver also applies a global limiter on the per-step velocity increment. At the current project stage this acts as an adaptive pseudo-step controller and is what prevents the Hunt default path from blowing up while the multi-region update remains explicit.
- `CaseSpec.initial_velocity` now exists because real FreeMHD cases do not necessarily start from rest; the current Shercliff `Ha20` smoke parity path uses this to match the recovered case initialization.

## Extension points

- Add full pressure-velocity coupling in `lmx.solvers`.
- Add mapped-grid differential operators for the fringing-field pipe case.
- Replace the current pseudo-transient laminar step with a fuller implicit Newton/Krylov formulation when parity demands it.

## Test and CI workflow

- `pytest -m unit`: mesh, operators, I/O, and benchmark report helpers.
- `pytest -m regression`: deterministic Hartmann and Shercliff golden centerline checks.
- `pytest -m physics`: solver smoke tests and low-Ha physical-invariant checks.
- `pytest -m validation`: analytical and FreeMHD-harness metadata checks.
- `python scripts/run_validation_suite.py --output artifacts/validation`: writes validation CSV, JSON, and VTK artifacts for Hartmann, Shercliff, and Hunt cases.
- `python scripts/run_benchmark_suite.py --output artifacts/benchmarks/benchmark.json`: writes the current benchmark report.
- `python -m lmx.cli validate shercliff --ha 20 --reference-root ./external/FreeMHDPaperAllFigures/FreeMHDPaperAllFigures/ClosedChannel --output ./artifacts/reference_compare`: compares the current solution against the ingested Zenodo analytical file.
- `python scripts/run_freemhd_case.py --image lmx-freemhd-smoke --platform linux/amd64 --case-dir ... --cores 8 --delta-t 1e-5 --end-time 1e-4 --write-interval 1e-4`: executes the current short FreeMHD smoke/parity run on this Apple-silicon host.
- `python scripts/sample_freemhd_profiles.py --case-dir ... --image microfluidica/openfoam:2206 --time 0.0001 --dict-name lmxSampleDict`: writes a sampling functionObject into `system/`, runs `postProcess`, and emits the sampled FreeMHD line cuts used by the newer parity metrics.

GitHub Actions now mirrors these entrypoints:

- `.github/workflows/ci.yml`: matrix over `unit`, `regression`, `physics`, and `validation`, plus a validation-artifact job.
- `.github/workflows/benchmarks.yml`: benchmark run with uploaded JSON artifacts.

## Reference data policy

- `scripts/fetch_freemhd_assets.py` now downloads the FreeMHD repo and the processed-figures archive by default.
- The much larger `StartingFiles.zip` archive is opt-in through `--include-starting-files` because it is roughly 8.9 GB and not needed for the current analytical ingestion path.
