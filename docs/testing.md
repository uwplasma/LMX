# Testing and validation

LMX uses one portable gate for source quality and separate explicit lanes for
external solvers, long physics campaigns, and hardware scaling.

## Complete portable gate

```bash
uv run --locked --extra dev python scripts/run_full_test_suite.py
```

The driver runs the full suite with branch coverage, a minimum coverage of 95%,
and a hard ten-minute timeout. The current Apple M4 result is 804 passed, 8
expected external-data skips, 95.30% branch coverage, and 229.5 seconds with six
workers.

The eight skips represent unavailable independent datasets, not disabled source
paths. The gate must stay below ten minutes as the code grows; its engineering
target is six minutes to preserve CI margin.

## Focused development checks

```bash
python -m pytest -m unit
python -m pytest tests/test_solvers.py
python -m pytest tests/test_fringing.py
python -m pytest tests/test_validation.py
```

Markers describe cost or external requirements, not correctness importance.
Avoid adding a new test file when a compact test belongs naturally in an
existing module-level family.

## What is tested

| Layer | Examples |
|---|---|
| API and configuration | constructors, TOML, CLI, errors, restart metadata |
| numerics | operators, linear solves, manufactured solutions, observed order |
| physics | analytical profiles, charge closure, wall currents, power balance |
| solver integration | steady/transient paths, diagnostics, backend equivalence |
| differentiability | finite differences, implicit gradients, transpose solves |
| parallelism | shard placement, one/two-device equivalence, scaling summaries |
| outputs | compact plots, tables, NPZ/JSON round trips |
| examples | every curated portable workflow |

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
python scripts/run_time_convergence_suite.py --help
python scripts/run_strong_scaling_worker.py --help
```

Each lane must emit a compact JSON or CSV summary with input, source, dependency,
and hardware fingerprints. Large raw fields belong in a versioned release.

## CI contract

Pull requests run the full gate on supported Python endpoints and build the
documentation with warnings as errors. Release jobs repeat the gate, build the
docs, build wheel and source distributions, and smoke-test the wheel. Dependency
compatibility is expressed as a supported range; the lockfile supplies CI
reproducibility.

## Adding functionality

A new stable feature is complete only when it has:

1. concise public API documentation and docstrings;
2. unit and failure-path coverage;
3. a numerics or physics validation appropriate to its claim;
4. one bounded example if it introduces a distinct user workflow;
5. no regression of the ten-minute gate or 95% branch-coverage floor.

See [Benchmark matrix](benchmark_matrix.md) for research acceptance and
[Developer guide](developer_guide.md) for contribution mechanics.
