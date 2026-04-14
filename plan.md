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
3. Harden and extend the first true `extruded_inductionless` solver slice into
   a broader production 3D family.
4. Extend the differentiable lane beyond the shipped Hartmann example set.

## Conservation hardening lane

The next research-grade solver work must keep conservation properties explicit,
not implicit. The main implementation targets are:

- conservative face-current construction in every solver family
- compatibility projection of potential right-hand sides onto a zero-net-source
  subspace before every Poisson-like electric solve
- interface-current continuity diagnostics across fluid/solid material jumps
- explicit boundary-current audits:
  - wall-normal current leakage on insulating or externally closed boundaries
  - inlet/outlet axial current imbalance
  - net boundary-flux residual over the full 3D control volume
- pressure-velocity updates that reduce both `div(u)` and charge/current
  imbalance together on the extruded 3D lane

The fully developed lane already carries compatibility projection and
charge-balance diagnostics. The fringing 3D slice now also carries axial
current and wall-leakage diagnostics, and the manual validation lane now
supports turning those metrics into hard pass/fail gates.

The current retained larger-dataset hard gate is:

- Hartmann / Shercliff / Hunt at `Ha = 10, 20`, resolution `10`
- fringing `rect_duct` and `pipe_ogrid` at the same `Ha` values with
  `nx_stations = 5`
- hard thresholds:
  - `charge_balance <= 8e-1`
  - `interface_current <= 7e-2`
  - `wall_current_leakage <= 1e-1`
  - `boundary_current <= 1e-5`

That retained gate now passes. Layered 3D fringing remains outside the
retained passing gate and stays on the open hardening list.

## Current status

- The default duct solver family is now `fully_developed_inductionless`.
- The historical reduced solver is retained only for regression.
- Runtime diagnostics now include linear residuals, flow rate, current,
  Lorentz power, and conservation signals.
- The logging surface now has a documented boolean `verbose` alias and explicit
  `verbosity = quiet|normal|detailed|debug` controls in TOML, CLI, and Python
  driver usage.
- Python `3.10` support is now explicit through the `tomli` fallback and the
  broader JAX dependency acceptance in packaging.
- Time integration now uses bounded step-count logic so `t_final` is not
  rounded up spuriously on fractional `dt` ratios.
- Public docs and examples now present a clean `1.0` release surface.
- The documentation surface has now been expanded into a fuller standalone
  manual: landing page, getting-started guide, numerics page, geometry and
  mesh workflows, testing strategy, richer theory notes, and Python-native
  variable-field/custom-geometry examples are all part of the shipped tree.
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
- A second autodiff example now validates Hartmann and forcing sensitivities
  against finite differences for a publication-friendly derivative-verification
  figure.
- A third autodiff example now performs full-profile inverse design over both
  forcing and Hartmann number, broadening the differentiable lane from scalar
  matching to small field-level inverse problems.
- A fourth autodiff example now performs fringing-history inverse design over
  peak Hartmann number and axial field-profile shape parameters, which is the
  first retained bridge from Hartmann-only autodiff into fringing-oriented
  research objectives.
- A fifth autodiff example now performs fringing multi-observable inverse
  design against both axial mean-velocity and current-proxy histories, which
  is the current lightweight bridge from scalar fringing objectives toward
  richer multi-observable fringing calibration.
- Rectangular fringing cases now have a first true low-Re
  `extruded_inductionless` 3D projection slice, exposing `u`, `v`, `w`, `p`,
  `phi`, current, Lorentz, and charge-balance fields through the explicit
  `ExtrudedInductionlessProblem -> ExtrudedInductionlessSolution` Python API.
- Layered fringing ducts now use the same first low-Re 3D projection slice.
- Layered fringing ducts now also use the same conservative face-current audit
  and closed-current axial boundary treatment as the rectangular 3D slice.
- Mapped `pipe_ogrid` fringing cases now also use the same explicit low-Re
  3D projection slice, so the remaining `extruded_inductionless` work is
  concentrated on production hardening, broader validation, and stronger
  front-end support rather than on geometry-family coverage gaps.
- The first executable TOML/CLI front-end for `extruded_inductionless` now
  exists through a dedicated `[fringing]` block plus shipped rectangular and
  mapped-pipe fringing input files.
- The executable surface now also includes direct `lmx run fringing_rect`,
  `lmx run fringing_pipe`, and `lmx run fringing_layered` shortcuts for quick
  3D/fringing launches without authoring TOML first.
- The heavy manual validation lane can now include a bounded fringing summary
  through `scripts/run_manual_solver_family_validation.py --include-fringing`,
  so solver-family hardening is no longer limited to fully developed Hartmann /
  Shercliff / Hunt datasets.
- The fully developed potential solve now projects its right-hand side onto a
  charge-neutral compatibility space and tracks an explicit
  `charge_balance_residual` diagnostic alongside `max|div J|`.
- An executable fringing-field benchmark scaffold now exists in `lmx/fringing.py`
  and `examples/fringing_benchmark_demo.py`, so axial field profiles and
  stationwise response metrics are now part of the post-1.0 research lane
  before the full `extruded_inductionless` solver lands.
- That fringing path now writes retained axial field bundles for
  `u(x, y, z)`, `v(x, y, z)`, `w(x, y, z)`, `p(x, y, z)`, `phi(x, y, z)`,
  `J(x, y, z)`, and Lorentz force, together with charge-balance residuals.
- That fringing path now also audits conservation with stationwise axial
  current histories, wall-current leakage, and a global net boundary-current
  residual so inlet/outlet and external-boundary behavior can be hardened
  explicitly during the post-`1.0` solver-family work.
- `solve_extruded_inductionless(...)` now runs a first true low-Re 3D
  pressure-velocity-potential iteration for rectangular ducts, layered ducts,
  and mapped-pipe fringing slices.
- That fringing path remains wrapped in an explicit
  `ExtrudedInductionlessProblem -> ExtrudedInductionlessSolution` public entry
  point, with validation metrics for residual size, charge balance, and
  field/response correlation. The remaining gap is now production hardening and
  geometry/scope expansion, not the total absence of a 3D slice.
- The manual solver-family validation lane can now treat conservation metrics
  as hard gates through explicit threshold controls for charge balance,
  interface current continuity, wall-current leakage, and net boundary-current
  residual.
- The autodiff lane now includes a bridge from the retained 3D slice back into
  the differentiable fringing model: `extruded_inductionless` histories can be
  used directly as inverse-design targets for axial field-shape recovery.
- The default rectangular extruded-target inverse-design path now goes one
  step further: it uses a direct differentiable rectangular
  `extruded_inductionless` response model instead of the older lightweight
  fringing-response surrogate.
- That direct rectangular differentiable 3D response now uses the same
  conservative electric source assembly and boundary-current residual target as
  the retained executable rectangular 3D fringing slice.
- The retained rectangular/layered 3D conservation audit is now aligned with
  the discrete electric operator: the source term, `div J` check, and boundary
  current residual are all assembled from conservative face fluxes instead of
  mixed cell-gradient diagnostics.
- Geometry preview tooling now exists for `rect_duct`, `layered_duct`, and
  mapped `pipe_ogrid` meshes, together with a user-facing example and docs for
  preprocessing/postprocessing geometry inspection.
- A separate variable-field/custom-geometry example now shows how to drive
  geometry changes and analytic magnetic fields directly from Python without
  relying on TOML-only workflows.
- The geometry preview example now defaults to a fast preview-only mode and
  exposes an explicit `--with-post-run` flag for short follow-on solves, so
  preprocessing visualization does not accidentally become a long-running task.
- Runtime logs now expose both initial and final residuals for the velocity and
  potential linear solves, which makes the CLI output closer to a long-form
  research solver log.
- The solver/runtime/IO/validation path now carries `charge_balance_residual`
  end to end: live logs, validation summaries, CLI JSON summaries, and
  restartable `.npz` bundles all expose it.
- The CPU and remote-GPU scaling workflow has now been revalidated on the live
  `office` host after the post-`1.0` compatibility changes, including Python
  `3.10` and a different installed JAX version.
- A manual solver-family hardening script now exists for the heavier
  post-1.0 release-validation lane so larger Hartmann/Shercliff/Hunt cases can
  be rerun without polluting the fast ship gate.
- The retained hard-gate dataset now passes for fully developed Hartmann /
  Shercliff / Hunt plus fringing `rect_duct` and `pipe_ogrid`; layered 3D
  fringing remains explicitly tracked but still outside the retained passing
  gate because its local charge-balance residual is not yet tight enough on the
  bounded larger-dataset runs.
- A fresh matched one-device smoke comparison now confirms the expected device
  direction of travel on the current tree: local CPU `512 x 512`, `32`
  iterations gives `warm_seconds ≈ 4.31e-3`, while a single office GPU gives
  `warm_seconds ≈ 6.65e-4`.
- Small two-GPU smoke runs on `256 x 256` remain overhead-dominated, so the
  strong-scaling narrative continues to rely on the larger committed artifact
  rather than those tiny validation points.
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
