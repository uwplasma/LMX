# Testing and validation

LMX uses one portable gate for source quality and separate explicit lanes for
external solvers, long physics campaigns, and hardware scaling.

## Complete portable gate

```bash
.venv/bin/python scripts/run_full_test_suite.py
```

The driver runs the full suite with branch coverage, a minimum coverage of 95%,
and a hard ten-minute timeout. The last committed Apple M4 record is 823 passed,
8 expected external-data skips, 95.00% combined line/branch coverage, and 161.3
seconds with six workers. The default warning threshold is five minutes,
leaving another five minutes before the hard failure. The record keeps the ten
slowest node IDs for critical-path review.

The eight skips represent unavailable independent datasets, not disabled source
paths. The gate must stay below ten minutes as the code grows; its engineering
target is five minutes to preserve CI margin.

## Focused development checks

```bash
python -m pytest tests/test_config.py
python -m pytest tests/test_freemhd.py -k matched_b2_lmx_input
python -m pytest tests/test_fringing.py::test_b2_steady_gate_requires_three_consecutive_passing_updates
```

Prefer direct node IDs or narrow `-k` expressions while developing. The current
module-level `unit` marker includes some expensive fringing and autodiff tests,
so it is not yet a guaranteed fast lane. Markers describe cost or external
requirements, not correctness importance. Avoid adding a new test file when a
compact test belongs naturally in an existing module-level family.

## Capability-to-evidence map

Exact benchmark node IDs and source hashes remain in the
[benchmark matrix](benchmark_matrix.md) and `benchmarks/provenance.json`.

| Capability | Portable test owner | Physics or external evidence | Status |
|---|---|---|---|
| CLI, config, restart, and I/O | CLI, config, I/O, and example-runner tests | portable workflow catalog | stable |
| Operators, mesh, and conservation | operator, mesh, physics, and convergence tests | manufactured solutions and observed order | stable |
| Ducts and high Ha | solver, physics, and validation tests | A1/A2/A3 analytical and FreeMHD records | accepted |
| Linear algebra and SOLVAX | linear plus focused solver/fringing tests | frozen SOLVAX CPU/GPU acceptance records | accepted |
| B1/B2 fringing | benchmark, fringing, and independence-runner tests | exact B2 smoke plus B1/B2 ALEX evidence | production FreeMHD parity open |
| Fields, geometry, walls, and blanket models | field, mesh, wall, and blanket tests | limiting cases and convergence | scoped external status |
| Differentiability | autodiff and gradient-focused solver tests | finite-difference and transpose evidence | stable paths accepted |
| Q2D and external adapters | Q2D and external-validation tests | independent-data readiness | quantitative parity open |
| Sharding and scaling | scaling and placement tests | exact tiny CPU/GPU equivalence records | larger accepted-case scaling open |
| Plots, examples, and packaging | plotting, showcase, repository, and reporting tests | media, docs, wheel, and provenance gates | stable |

Coverage is a floor, not a validation claim. Physics acceptance requires
quantitative reference and conservation gates even when code coverage is 100%.

## Test design rules

- Test public behavior and invariants rather than implementation choreography.
- Parametrize related cases instead of copying test functions.
- Share expensive compiled fixtures within a worker when isolation is safe.
- Keep grids at the smallest size that exposes the numerical property.
- Mark a genuinely expensive test with a measured local timeout.
- Never weaken a physics tolerance merely to shorten CI.
- Put multi-hour campaigns outside pytest and retain compact accepted outputs.

## Manual research lanes

These commands are intentionally not part of the portable gate:

```bash
python scripts/run_freemhd_parity_suite.py --help
python scripts/run_convergence_suite.py --help
python scripts/run_convergence_suite.py --mode time --help
python scripts/run_strong_scaling_worker.py --help
```

Each lane must emit a compact JSON or CSV summary with input, source, dependency,
and hardware fingerprints. Large raw fields belong in a versioned release.

## CI contract

Pull requests run the full test battery on both supported Python endpoints,
measuring branch coverage once on Python 3.13 and running Python 3.10 without
instrumentation overhead. Documentation builds with warnings as errors. Release
jobs repeat the covered gate, build the docs and distributions, and smoke-test
the wheel. CI exercises the minimum and newest compatible SOLVAX endpoints;
release records preserve the resolved environment and versions as evidence.

## Adding functionality

A new stable feature is complete only when it has:

1. concise public API documentation and docstrings;
2. unit and failure-path coverage;
3. a numerics or physics validation appropriate to its claim;
4. one bounded example if it introduces a distinct user workflow;
5. no regression of the ten-minute gate or 95% combined line/branch floor.

See [Benchmark matrix](benchmark_matrix.md) for research acceptance and
[Developer guide](developer_guide.md) for contribution mechanics.
