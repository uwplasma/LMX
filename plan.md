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

1. Finish solver-family hardening for the default fully developed path.
2. Push benchmark and physics tests toward release quality.
3. Complete the documentation and API cleanup for the `1.0` public surface.
4. Audit performance and preserve the differentiable core path.
5. Stage `extruded_inductionless` interfaces and benchmark manifests for the
   next paper.

## Current status

- The default duct solver family is now `fully_developed_inductionless`.
- The historical reduced solver is retained only for regression.
- Runtime diagnostics now include linear residuals, flow rate, current,
  Lorentz power, and conservation signals.
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

## Release checklist

- [ ] Hartmann analytical acceptance locked
- [ ] Shercliff analytical acceptance locked
- [ ] Hunt benchmark acceptance locked
- [x] docs build clean
- [ ] CLI examples clean
- [ ] restart examples clean
- [ ] coverage and QA pass reviewed
- [ ] performance and differentiability notes documented
