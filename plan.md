# LMX 1.0 Plan

## Goal

Ship a research-grade `1.0` inductionless MHD code with:

- a stable fully developed duct solver
- benchmark-quality Hartmann, Shercliff, and Hunt validation
- explicit restart, logging, plotting, and CLI workflows
- a documented differentiable core
- a clear roadmap to the 3D fringing-field solver family

## Scope for 1.0

### In scope

- `fully_developed_inductionless`
- Hartmann, Shercliff, Hunt
- `rect_duct` and `layered_duct`
- TOML and Python-driver workflows
- restartable `.npz` outputs
- validation, convergence, and benchmark scripts
- user and developer documentation

### Out of scope

- turbulence
- heat transfer
- free surfaces
- full induction
- production-ready 3D fringing-field solver

## Open items

1. Keep hardening the default fully developed solver family in the manual
   release-validation lane.
2. Expand benchmark and physics depth for publication datasets.
3. Stage `extruded_inductionless` interfaces and benchmark manifests for the
   next paper.
4. Extend the differentiable lane beyond the shipped Hartmann example.

## Current status

- The default duct solver family is now `fully_developed_inductionless`.
- The historical reduced solver is retained only for regression.
- Runtime diagnostics now include linear residuals, flow rate, current,
  Lorentz power, and conservation signals.
- The logging surface now has a documented boolean `verbose` alias and explicit
  `verbosity = quiet|normal|detailed|debug` controls in TOML, CLI, and Python
  driver usage.
- Public docs and examples now present a clean `1.0` release surface.
- External executable comparisons are documented only as secondary benchmark
  evidence, not as implementation guidance.
- Public JSON/example/report outputs have been scrubbed to avoid leaking
  workstation-specific absolute paths.
- The remaining `1.0` gate is now dominated by solver-heavy physics and
  validation tests rather than benchmark/example/I/O harness overhead.
- Benchmark, I/O, and example tests have been rewritten to use synthetic or
  monkeypatched orchestration paths where full solves were unnecessary.
- Validation report tests now stub solver execution where they are asserting
  report/schema behavior rather than analytical acceptance.
- CI coverage no longer forces `JAX_DISABLE_JIT=1`, because that setting was
  inflating runtime on solver-heavy tests without improving release confidence.
- Default CI is now being narrowed to a fast ship gate, with benchmark and
  validation-artifact workflows moved to manual `workflow_dispatch` runs so
  routine pushes do not consume research-artifact runtime on every change.
- The default push/PR gate now excludes the heaviest `physics` marker tests;
  those remain available in a manual workflow-dispatch lane together with
  benchmark, artifact, and extended coverage runs.
- The default push/PR gate also excludes the heavier `regression` marker
  tests, which are now part of the manual release-validation lane together
  with physics.
- The full local fast suite now completes in about `35 s`, and the full
  local coverage lane completes in about `45 s`, both within the hard
  five-minute limit for routine validation.
- Current combined coverage for `lmx/` and `scripts/` is `90%`.
- Budgeted CLI and restart smokes now pass on the shipped Hartmann TOML path;
  the release gate uses short-budget generated TOMLs rather than full
  long-horizon example runs so the interface is verified without violating the
  five-minute rule.
- CPU and GPU strong-scaling artifacts now exist for the dominant stencil kernel,
  with a committed publication figure under `docs/_static/generated/strong_scaling.png`.
- The differentiable Hartmann example now has a committed publication figure under
  `docs/_static/generated/autodiff_summary.png`, showing both Hartmann-number
  sensitivity and inverse recovery of a forcing parameter.
- Manufactured-solution and direct-kernel tests now cover the low-cost
  numerical core well: `lmx/linear.py` is about `99%`, and
  `lmx/operators.py` is about `98%`.
- A direct branch-coverage pass now closes most of the cheap remaining misses
  in `lmx/physics.py` and `lmx/plotting.py`; the remaining release-coverage
  gap is even more concentrated in `lmx/solvers.py`.
- The biggest remaining coverage gap is now overwhelmingly
  `lmx/solvers.py`, so post-1.0 test work is targeted solver-family branch
  coverage rather than suite-runtime cleanup.
- The fast ship gate, docs build, budgeted CLI/restart smokes, performance
  figures, autodiff figures, and release coverage threshold are all now in
  place for `1.0`.

## Release checklist

- [x] Hartmann analytical acceptance locked
- [x] Shercliff analytical acceptance locked
- [x] Hunt benchmark acceptance locked
- [x] docs build clean
- [x] CLI examples clean
- [x] restart examples clean
- [x] fast test-runtime budget enforced
- [x] coverage and QA pass reviewed
- [x] performance and differentiability notes documented
