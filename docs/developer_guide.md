# Developer Guide

## Package structure

- `lmx.mesh`: grid builders.
- `lmx.specs`: immutable case/config dataclasses.
- `lmx.physics`: conductivity and magnetic-field field construction.
- `lmx.operators`: mesh-aware finite-volume style kernels.
- `lmx.solvers`: laminar inductionless solver entrypoints.
- `lmx.io`: ParaView XML and CSV outputs.
- `lmx.validation`: analytical helpers and FreeMHD harness.

## Array layout

- Cross-section fields use shape `(ny, nz)`.
- The solver currently models the streamwise velocity `u(y, z)` plus electric potential `phi(y, z)`.
- Conductivity and masks are stored on the same cell-centered layout.

## JAX strategy

- Solver stepping uses `jax.lax.scan` for stable fixed-step execution.
- Linear solves use `lineax` when available and otherwise fall back to a pure-JAX Jacobi solver.
- Keep shapes static when adding new operators or diagnostics.

## Extension points

- Add full pressure-velocity coupling in `lmx.solvers`.
- Add mapped-grid differential operators for the fringing-field pipe case.
- Replace the current pseudo-transient laminar step with a fuller implicit Newton/Krylov formulation when parity demands it.
