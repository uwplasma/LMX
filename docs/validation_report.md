# Validation Report

## Implemented now

- Structured duct meshes with optional conducting wall layers.
- Laminar inductionless field solve with `u`, `phi`, `J`, and `JxB`.
- Hartmann analytical profile helper.
- Hartmann validation JSON generation from the CLI and the validation suite runner.
- CSV and ParaView outputs for centerline and field inspection.
- FreeMHD comparison harness metadata and expected container command generation.
- GitHub Actions validation artifacts for Hartmann, Shercliff, and Hunt runs.
- GitHub Actions benchmark artifacts for the current Hartmann timing path.
- Zenodo closed-channel analytical text ingestion for Shercliff and Hunt.
- Zenodo processed closed-channel slice ingestion for future parity checks against the paper figures.

## Planned next

- Shercliff and Hunt analytical-profile ingestion from the paper and Zenodo assets.
- Fringing-field mapped-pipe operators.
- Automated parity runners against local FreeMHD container executions.
- Stronger acceptance thresholds that can fail CI on parity regressions rather than only emitting artifacts.

## Current limitation

- Hartmann `Ha=20` is now stable on the default finer mesh and shows a low normalized error in the current analytical comparison.
- Shercliff `Ha=20` no longer clips on the default finer mesh, but its normalized error is still too large to count as parity-complete.
- Hunt remains clipping at the current default pseudo-time settings and is still the highest-priority solver-fidelity gap.
- Hunt can be stabilized only with a much smaller pseudo-step, which confirms the remaining issue is solver stability rather than missing reference-data plumbing.
