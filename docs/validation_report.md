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

- Shercliff and Hunt reference comparisons are now wired, but the present solver still shows large error and clipping at `Ha=20`, so these reports are informative artifacts rather than passing parity gates.
