# Testing and Validation

LMX uses a split validation strategy so that the default release gate remains
fast while deeper research checks stay reproducible.

## Fast ship gate

The routine gate is the fast `unit` / `validation` lane:

```bash
python -m pytest -m 'unit or validation' -q
python -m sphinx -W -b html docs docs/_build/html
```

This lane is intentionally kept below five minutes. It covers:

- configuration parsing
- CLI dispatch
- runtime logging
- output writing and restart loading
- plotting/reporting contracts
- low-cost solver checks
- manufactured/direct-kernel tests

## Manual research lanes

Heavier validation remains available for manual or release-time execution:

- regression snapshots
- heavier physics suites
- benchmark artifact generation
- extended coverage collection

This split avoids exhausting CI runtime on every routine push while still
preserving reproducible research checks.

## Test families

### Unit tests

Unit tests validate:

- dataclass/config behavior
- operator kernels
- linear-solver helpers
- output schema and logging contracts
- example orchestration paths

### Validation tests

Validation tests focus on:

- analytical benchmark metrics
- report generation
- convergence bookkeeping
- benchmark summary writing

### Regression tests

Regression tests freeze representative outputs for:

- solver behavior
- examples
- reporting formats
- selected benchmark artifacts

### Physics-heavy checks

The heavier manual lane is where we continue to harden:

- Hartmann analytical acceptance
- Shercliff analytical acceptance
- Hunt benchmark acceptance
- deeper solver-family hardening on larger meshes and longer windows

## Coverage strategy

LMX intentionally avoids expensive end-to-end solves when a smaller synthetic
or monkeypatched test can validate the same contract. The current coverage
strategy uses:

- direct kernel tests for `lmx/linear.py` and `lmx/operators.py`
- monkeypatched orchestration tests for examples and reporting
- small-mesh real solves for selected solver paths
- focused autodiff tests on reduced iteration counts

This is how the project maintains the routine five-minute budget while keeping
coverage near the release target.

## Source map

- `tests/test_solver.py`
  - solver-family and control-path tests
- `tests/test_validation.py`
  - benchmark/report summaries
- `tests/test_linear.py`
  - linear-kernel coverage
- `tests/test_operators.py`
  - operator-kernel coverage
- `tests/test_example_runner.py`
  - example orchestration and artifacts
- `tests/test_autodiff.py`
  - differentiable lane checks
- `tests/test_fringing.py`
  - fringing research slice
