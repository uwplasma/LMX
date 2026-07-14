# Changelog

LMX follows semantic versioning for the stable public API. Research-stage APIs
may change between minor releases, but every such API is labeled in the
documentation.

## Unreleased

### Added

- Canonical Benchmark A specifications, compact acceptance records, provenance,
  and all eight Samper/Bühler high-Hartmann-number rows.
- A deterministic full-suite runner with a ten-minute hard limit and 95% branch-
  coverage floor.
- Machine-checked architecture budgets, benchmark provenance, and release-asset
  index.
- Axial JAX sharding with measured two-GPU B2 strong scaling and restart-safe
  checkpointing.
- A migration guide from former root aliases to their owning modules.

### Changed

- Reduced the stable root API to 30 deliberate exports.
- Curated `examples/` to 11 first-run journeys and grouped reusable inputs under
  `examples/cases/`; generated research work now stays under ignored
  `artifacts/`.
- Consolidated the portable suite to 34 files and 767 passing tests while
  retaining 95.28% branch coverage and a 240.3-second six-worker gate.
- Accepted compatible SOLVAX 0.8 releases below 1.0 instead of pinning one
  patch version.
- Moved 65 large generated artifacts to the checksummed
  `lmx-research-assets-v1` GitHub release; six compressed documentation
  derivatives remain in Git and the tracked checkout stays below 4 MiB.

### Fixed

- Skip the second B2 electric PCG refinement when the first solve already
  satisfies the local current-balance residual.
- High-Ha Hunt and Shercliff solver robustness, exact insulating topology,
  scale-invariant PCG stopping, strict coupled convergence certification, and
  conservative current/power diagnostics.

## 1.1.3 — 2026-05-01

- Current published package baseline before the Benchmark A and M2 hardening
  work listed under Unreleased.
