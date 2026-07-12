# Changelog

LMX follows semantic versioning for the stable public API. Research-stage APIs
may change between minor releases, but every such API is labeled in the feature
manifest and documentation.

## Unreleased

### Added

- Canonical Benchmark A specifications, compact acceptance records, provenance,
  and all eight Samper/Bühler high-Hartmann-number rows.
- A deterministic full-suite runner with a ten-minute hard limit and 95% branch-
  coverage floor.
- Machine-checked architecture, feature, benchmark, workflow-disposition, and
  release-asset manifests.
- One-release warnings for the legacy root namespace and a migration guide.

### Changed

- Reduced the stable root API to 30 deliberate exports.
- Curated `examples/` to 11 first-run journeys; moved research/evidence work to
  `campaigns/` and reusable configurations to `cases/`.
- Moved 65 large generated artifacts to the checksummed
  `lmx-research-assets-v1` GitHub release, reducing the source checkout below
  10 MiB.

### Fixed

- High-Ha Hunt and Shercliff solver robustness, exact insulating topology,
  scale-invariant PCG stopping, strict coupled convergence certification, and
  conservative current/power diagnostics.

## 1.1.3 — 2026-05-01

- Current published package baseline before the Benchmark A and M2 hardening
  work listed under Unreleased.
