# 0003 — Explicit precision and bounded-memory derivatives

Status: accepted under plan D3–D5.

Configure precision before constructing arrays or tracing. Fully developed and
validation defaults are float64; float32 requires measured error bounds, not a
global accuracy promise. Reuse SOLVAX implicit tangent/transpose solves for
converged algebraic states; checkpoint transient steps. Do not materialize dense
production Jacobians. Verify derivatives by duality and independent differences.

Basis: JAX's [dtype contract](https://docs.jax.dev/en/latest/101/default_dtypes.html)
and [custom linear solve](https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_linear_solve.html).
