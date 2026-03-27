# Validation Report

## Implemented now

- Structured duct meshes with optional conducting wall layers.
- Laminar inductionless field solve with `u`, `phi`, `J`, and `JxB`.
- Hartmann analytical profile helper.
- Hartmann validation JSON generation from the CLI.
- CSV and ParaView outputs for centerline and field inspection.
- FreeMHD comparison harness metadata and expected container command generation.

## Planned next

- Shercliff and Hunt analytical-profile ingestion from the paper and Zenodo assets.
- Fringing-field mapped-pipe operators.
- Automated parity runners against local FreeMHD container executions.
- Runtime and convergence benchmark dashboards.
