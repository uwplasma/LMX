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
- The portable package gate passes 817 tests with 8 expected external-data
  skips and 95.02% combined line/branch coverage in 149.3 seconds on the
  reference Mac.
- The reduced B2 same-state ladder supports a 64x larger pseudo-time cap while
  preserving the electromagnetic-scale map defect within 0.192%; eight warm
  updates are monotone and restart bitwise. A direct momentum-defect threshold
  and the ALEX pressure-hole observation operator remain open.

## Research-stage

- ALEX B1 conducting-pipe fringing flow: conservation, restart, and bounded
  large-grid pressure convergence pass; experimental pressure agreement and
  the final mesh/observable acceptance record remain open.
- ALEX B2 square-duct fringing flow: the earlier no-inertia, stationwise-flow
  formulation has conservation, two-GPU numerical equivalence, a 1.87x
  fine-checkpoint solver promotion, and fine numerical gates. These are
  diagnostics, not validation of the canonical finite-inertia formulation.
  That formulation and its independently observed LMX/FreeMHD tiny inputs are
  implemented. Its exact two-update LMX/FreeMHD harness passes every frozen
  smoke gate, and its current production path has exact restart plus equivalent
  observables on 1/2/4 CPU devices. The prior 1/2 deterministic-GPU ladder
  predates the terminal-restart fix and awaits refresh. These tiny runs establish
  orchestration and sharding correctness. A historical `128 x 67 x 67`
  calibration has equivalent observables and bounded face-flux replay; diagonal
  momentum preconditioning cuts its GPU runtime by 3.75–6.84x, but shared-host
  variance leaves that fixed-grid ratio open. A stable doubled-axial rung
  reaches only 1.125x, below the scaling-promotion threshold. A three-update
  trajectory preserves all primary fields exactly. This is not production
  parity or steady scaling. The first fresh current-formulation coarse
  trajectory and one exact continuation pass conservation and all 256 pressure
  solves, but stop at step 256 with residual `7.1081e-4`, above both its
  precommitted continuation gate and the `5e-5` steady criterion. It is not
  promoted, and more stepping, larger, or independence runs remain blocked.
  The legacy fine pressure curve misses every frozen ALEX
  literature-error limit. A checksummed
  Maxwell-consistent coarse-field diagnostic improves peak underprediction from
  15.6% to 8.2% but worsens the plateau-sensitive aggregate error, so it is not
  validation evidence; production-mesh matched-field FreeMHD and three-mesh
  gates remain required.
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
python scripts/run_samper_table_i.py \
  --freeze-summary artifacts/samper/table-i-summary.json \
  --output benchmarks/results/samper-table-i-accepted.json
python scripts/build_benchmark_a_acceptance.py
```

External and accelerator runs are explicit because they require separately
installed software, reference data, or hardware. See [External benchmarks](external_benchmarks.md),
[Performance](performance.md), and the [Benchmark matrix](benchmark_matrix.md).
