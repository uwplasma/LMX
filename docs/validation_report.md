# Validation report

This page summarizes current claims. Machine-readable records in `benchmarks/`
are authoritative when prose and results differ.

## Accepted

- Hartmann, Shercliff, and Hunt fully developed inductionless duct cases have
  analytical profile, mesh-convergence, current-closure, and power-balance
  tests.
- All eight frozen Samper Table I Shercliff/Hunt rows at
  `Ha = 500, 5000, 10000, 15000` pass.
- Audited `85 x 63` FreeMHD closed-channel Shercliff and Hunt observables pass
  the 1% finite-grid gate.
- SOLVAX 0.8 integration passes primal, implicit-gradient, independent
  transpose, CPU/GPU, and bounded end-to-end gates.
- The portable package gate passes 807 tests with 8 expected external-data
  skips and 95.30% branch coverage in 321.0 seconds on the reference Mac.

## Research-stage

- ALEX B1 conducting-pipe fringing flow: conservation and restart machinery are
  present; large-grid steady pressure convergence and experimental pressure
  agreement remain open.
- ALEX B2 square-duct fringing flow: conservation and two-GPU numerical
  equivalence pass; experimental validation remains open.
- Q2D turbulence, magnetic-obstacle, mapped blanket, and Dean-vortex workflows
  provide model or adapter checks but do not yet support quantitative claims.

## Not implemented or claimed

- full magnetic induction;
- validated 3D turbulence;
- coupled heat transfer and buoyancy;
- free surfaces;
- complete FreeMHD feature parity.

## Evidence hierarchy

1. Analytical/manufactured solutions verify equations and operators.
2. Mesh/time convergence and conservation quantify numerical error.
3. Independent-code comparison detects implementation disagreement.
4. Experimental comparison assesses model validity.

Each level answers a different question. Agreement with one code cannot replace
convergence or experimental validation.

## Reproduce

Portable source gate:

```bash
uv run --locked --extra dev python scripts/run_full_test_suite.py
```

Benchmark A aggregate:

```bash
python scripts/run_samper_table_i.py
python scripts/build_benchmark_a_acceptance.py
```

External and accelerator runs are explicit because they require separately
installed software, reference data, or hardware. See [External benchmarks](external_benchmarks.md),
[Performance](performance.md), and the [Benchmark matrix](benchmark_matrix.md).
