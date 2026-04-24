# Testing and Validation

LMX uses a split validation strategy so that the default release gate remains
fast while deeper research checks stay reproducible.

## Research-grade test contract

The test suite is expected to protect scientific behavior, not only exercise
lines of Python. Every new paper-facing capability should have all of the
following before it is described as validated:

- a direct unit test for setup, helper, and fallback logic
- a numerical verification or invariant test
- a physics or literature-facing validation test
- an artifact-producing example that writes a figure and summary JSON
- a docs entry that states whether the case is verification, validation, or a
  capability demonstration

This contract follows the benchmark hierarchy in Samper et al. and the current
FreeMHD comparison surface: analytical solutions and conservation checks are
the primary acceptance gate, while external solver comparisons add independent
evidence through matched observables such as velocity, potential, current,
Lorentz force, pressure proxy, and runtime.

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
- Q2D decay, forced, and wall-bounded reduced-model artifacts with modal
  energy-budget gates
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
coverage at the release target.

The broad local coverage gate now clears `95%` over `lmx/` and `scripts/`
without moving long benchmark runs into routine CI. The current preferred order
for further coverage work is:

- delete genuinely dead helper branches instead of testing historical behavior
- add cheap direct tests for `lmx/solvers.py` branch helpers, restart/initial
  condition paths, flow-rate/forcing modes, current reconstruction, and
  diagnostic guards
- keep manufactured-solution and invariant tests close to `lmx/operators.py`
  and `lmx/linear.py`
- keep paper-facing examples in `examples/`, then test their summary JSON
  schema and governing observables rather than comparing image pixels
- leave heavy FreeMHD reruns, high-Ha mesh ladders, and long scaling campaigns
  in manual or release workflows

The manual coverage workflow now enforces the `95%` gate with
`--cov-fail-under=95`. The default push/PR lane remains the sub-five-minute
`unit or validation` suite so routine commits do not run the full coverage
campaign.

## Publication artifact rule

Numerical and physics tests that demonstrate publishable behavior should have a
matching artifact-producing workflow. The expected pattern is:

- the test asserts the invariant, convergence rate, profile error, or
  conservation metric cheaply
- the example writes a publication-ready PNG/PDF and a machine-readable JSON
  summary
- the docs link the figure and state the literature anchor
- CI checks the JSON summary and docs build; manual workflows regenerate the
  heavier figures when needed

This is the route for the straight-duct overlays, 3D fringing-field summaries,
Q2D panels, localized-field response panels, WHAM sensitivity figures,
bent-pipe overview, and future heat-transfer plots.

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

## Literature anchors

The testing strategy is intentionally tied to published benchmark ladders and
verification/validation practice, not only to internal regression history.

- [Samper et al., *An approach to verification and validation of MHD codes for fusion applications*](https://www.sciencedirect.com/science/article/pii/S0920379614003263)
  - governs the A/B/C/D/E benchmark ladder used in the plan and benchmark docs
- [FreeMHD V&V paper, arXiv:2409.08950](https://arxiv.org/abs/2409.08950)
  - provides straight-duct, fringing, and free-surface comparison targets and
    a useful reference implementation baseline
- [Quasi-two dimensional perturbations in duct flows under transverse magnetic field](https://arxiv.org/abs/2006.03993)
  - anchors the current Q2D Hartmann-friction validation direction
  - the current decay, forced, and wall-bounded Q2D summaries now include the
    modal energy balance `dE/dt = P - 2 lambda E` as a reduced-model closure
    gate before turbulent spectra are claimed
  - the current wall-bounded Q2D summary now emits energy, enstrophy,
    dissipation, and shell-spectrum observables, but the turbulence gate remains
    open until those quantities are compared with published turbulent Q2D data
- [On the flow past a magnetic obstacle](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/on-the-flow-past-a-magnetic-obstacle/F4185BE5315273DBA9D1C53DD49990AA)
- [Constrained flow around a magnetic obstacle](https://www.cambridge.org/core/journals/journal-of-fluid-mechanics/article/constrained-flow-around-a-magnetic-obstacle/DFD706B066E0B0C7E8598544E1783BC0)
  - anchor the wake-deficit / recovery / distortion observables needed before
    the magnetic-obstacle lane can be called externally validated
- [Validation and verification of a robust 3-D MHD code](https://www.sciencedirect.com/science/article/pii/S0920379618300358)
  - supports the broader validation roadmap for curved ducts, magnetic
    obstacles, and 3D liquid-metal benchmark structure
- [A research framework for writing differentiable PDE discretizations in JAX](https://arxiv.org/abs/2111.05218)
  - anchors the autodiff verification philosophy for gradient, optimization,
    and differentiable-operator tests

## Artifact-producing verification examples

When a numerical or physics verification family is strong enough to support
the docs or manuscript, it should not remain only as a unit test. The intended
pattern is:

- keep the cheap direct test in `tests/`
- add a companion standalone example under `examples/`
- write a publication-ready figure plus a summary JSON
- validate the summary schema and governing observables in tests

This keeps the fast lane bounded while making the same verification evidence
available for documentation and later manuscript figures.

The first explicit examples following this pattern are:
- `examples/operator_verification_demo.py` for smooth-grid manufactured
  operator convergence
- `examples/operator_clustered_verification_demo.py` for clustered
  boundary-layer operator convergence
- `examples/interface_conductivity_verification_demo.py` for aligned
  coefficient-jump verification
- `examples/straight_duct_profile_comparison.py` for the Hartmann /
  Shercliff / Hunt literature-facing straight-duct panel, including no-slip
  wall reconstruction when comparing cell-centered profiles against the
  analytical wall-to-wall curves; the current release target is
  `L2 <= 1.2e-2` on the retained cuts, and the current bounded `37 × 37`
  artifact meets it for Hartmann, Shercliff, and Hunt
- `examples/hartmann_validation_ladder.py` for the bounded Hartmann multi-`Ha`
  literature ladder, with the same stable summary-JSON pattern used by the
  manuscript-facing straight-duct figures
- `examples/straight_duct_validation_ladder.py` for the bounded Shercliff /
  Hunt multi-Ha literature ladder
- `examples/freemhd_closed_channel_parity.py` for fresh LMX versus FreeMHD
  transient straight-duct parity and runtime comparison on the same host
