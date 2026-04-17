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
- small bundled-reference physics regressions for Hartmann, Shercliff, and Hunt

## Manual research lanes

Heavier validation remains available for manual or release-time execution:

- regression snapshots
- heavier physics suites
- benchmark artifact generation
- extended coverage collection

The main post-`1.0` manual entry point is:

```bash
python scripts/run_manual_solver_family_validation.py \
  --output artifacts/manual_validation/solver_family_summary.json \
  --ha-values 10,20 \
  --resolutions 16,24 \
  --include-fringing \
  --write-csv \
  --write-plot \
  --max-steps 20 \
  --potential-iterations 80 \
  --coupling-iterations 8
```

That command keeps the fast lane clean while still exercising the heavier
Hartmann / Shercliff / Hunt acceptance path together with the bounded
fringing slices. With `--write-csv` and `--write-plot`, the same run produces
a machine-readable campaign table and a convergence figure
for the fringing conservation metrics.

The manual lane can also enforce conservation thresholds directly:

```bash
python scripts/run_manual_solver_family_validation.py \
  --output artifacts/manual_validation/solver_family_summary.json \
  --ha-values 10,20 \
  --resolution 24 \
  --include-fringing \
  --fringing-geometries rect_duct,layered_duct,pipe_ogrid \
  --max-charge-balance 8e-1 \
  --max-interface-current 2.5e-1 \
  --max-fringing-wall-current-leakage 1e-1 \
  --max-fringing-boundary-current 1e-5 \
  --fail-on-threshold
```

This is the intended post-1.0 release-validation mode for conservation
hardening.

The current combined Benchmark A/B exercise is:

```bash
python scripts/run_full_validation_exercise.py \
  --output artifacts/validation/full_validation_exercise \
  --ha-values 10,20 \
  --resolution 12 \
  --fringing-resolutions 8,12 \
  --skip-paraview \
  --write-plot
```

That driver produces Benchmark A artifacts, Benchmark B gate summaries, and a
combined JSON/CSV/Markdown report for the current documented thresholds.
When run from the source tree it uses the bundled closed-channel reference
dataset by default, so `--reference-root` is only needed for an alternate
comparison set.

The bundled-reference physics regressions sit between the fast gate and the
manual campaign: they use the real steady solver on small grids, compare
against the bundled analytical/reference datasets, and keep low-resolution
profile drift visible without forcing the full A/B exercise into routine CI.

## Physics gates

The heavier validation lane should only be considered passing when all of the
following are satisfied on the selected dataset:

- fully developed ducts
  - Hartmann/Shercliff/Hunt profile errors remain bounded under refinement
  - flow-rate and forcing/pressure-proxy trends stabilize under refinement
  - charge-balance and interface-current residuals stay below configured
    thresholds
- fringing 3D cases
  - charge-balance, wall-current leakage, and boundary-current residuals stay
    below configured thresholds
  - throughput variation outside the field-ramp region remains bounded
  - field/mean-velocity correlation has the expected negative sign
  - pressure span grows through the magnet region and relaxes downstream

## Quality gates

Routine and release validation together should maintain:

- fast lane under five minutes
- strict docs build
- deterministic restart continuation on the executable TOML/CLI path
- stable JSON/CSV/NPZ output schemas
- example workflows that regenerate the committed docs media and figures
- branch coverage concentrated away from dead or historical code paths

Current larger dataset:

- `Ha = 10, 20`
- `resolution = 10`
- bounded fringing lane on `rect_duct`, `layered_duct`, and `pipe_ogrid`
- hard thresholds:
  - `charge_balance <= 8e-1`
  - `interface_current <= 2.5e-1`
  - `wall_current_leakage <= 1e-1`
  - `boundary_current <= 1e-5`

The layered 3D case joined that validation gate after the multi-region electric
subproblem was upgraded from a bounded iterative solve to a sparse direct solve
of the conservative variable-coefficient potential operator.

That gate now passes on the current tree for rectangular ducts,
layered ducts, and mapped pipes.

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
