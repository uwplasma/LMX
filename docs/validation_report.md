# Validation report

This page summarizes current claims. Machine-readable records in `benchmarks/`
are authoritative when prose and results differ.

<p align="center">
  <img src="_static/analytic_velocity_profiles.webp" alt="Analytical and LMX duct profiles" width="47%">
  <img src="_static/freemhd_closed_channel_observable_parity.webp" alt="LMX and FreeMHD closed-channel observable parity" width="47%">
</p>

<p align="center">
  <a href="_static/readme_hunt_startup_2d.mp4"><img src="_static/readme_hunt_startup_2d_poster.webp" alt="Seven-second Hunt startup loop" width="55%"></a>
</p>

## Accepted

- Hartmann, Shercliff, and Hunt fully developed inductionless duct cases have
  analytical profile, mesh-convergence, current-closure, and power-balance
  tests.
- All eight frozen Samper Table I Shercliff/Hunt rows at
  `Ha = 500, 5000, 10000, 15000` pass.
- Audited `85 x 63` FreeMHD closed-channel Shercliff and Hunt observables pass
  the 1% finite-grid gate.
- SOLVAX integration, including symmetric additive-line composition,
  passes primal, implicit-gradient, independent transpose, CPU/GPU, and bounded
  end-to-end gates.
- The portable package gate passes 784 tests with 8 expected external-data
  skips and 95.28% branch coverage in 149.9 seconds on the reference Mac.

## Research-stage

- ALEX B1 conducting-pipe fringing flow: conservation, restart, and bounded
  large-grid pressure convergence pass; experimental pressure agreement and
  the final mesh/observable acceptance record remain open.
- ALEX B2 square-duct fringing flow: the earlier no-inertia, stationwise-flow
  formulation has conservation, two-GPU numerical equivalence, a 1.87x
  fine-checkpoint solver promotion, and fine numerical gates. These are
  diagnostics, not validation of the canonical finite-inertia formulation.
  That formulation and its independently observed LMX/FreeMHD tiny inputs are
  implemented; executing the exact two-update smoke remains open. The fine
  pressure curve misses every frozen ALEX literature-error limit. A checksummed
  Maxwell-consistent coarse-field diagnostic improves peak underprediction from
  15.6% to 8.2% but worsens the plateau-sensitive aggregate error, so it is not
  validation evidence; exact matched-field FreeMHD and three-mesh gates remain
  required.
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
.venv/bin/python scripts/run_full_test_suite.py
```

Benchmark A aggregate:

```bash
python scripts/run_samper_table_i.py
python scripts/freeze_samper_table_i.py \
  artifacts/samper/table-i-summary.json \
  benchmarks/results/samper-table-i-accepted.json
python scripts/build_benchmark_a_acceptance.py
```

External and accelerator runs are explicit because they require separately
installed software, reference data, or hardware. See [External benchmarks](external_benchmarks.md),
[Performance](performance.md), and the [Benchmark matrix](benchmark_matrix.md).
