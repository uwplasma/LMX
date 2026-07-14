# LMX authoritative development plan

This is the single project plan. It replaces historical task logs and campaign
ledgers. Evidence belongs in compact benchmark records, tests, documentation,
or versioned release assets—not in new top-level project trees.

## Product target

LMX will be a lightweight JAX code for accurate, end-to-end differentiable
inductionless liquid-metal MHD on CPUs and GPUs. FreeMHD and published
solutions are independent references. Stable claims require reproducible
physics, numerical, gradient, and performance evidence.

The public repository has four working surfaces:

- `lmx/`: source code and the `lmx` command
- `tests/`: unit, numerical, physics, regression, and workflow tests
- `examples/`: bounded first-run programs and reusable TOML inputs
- `docs/`: theory, inputs, validation, performance, and development guidance

Root files are limited to packaging, licensing, citation, contribution,
changelog, README, and this plan. Large outputs, raw external-code runs,
movies, meshes, and manuscript bundles belong in GitHub or Zenodo releases.

## Non-negotiable gates

Every promoted change must satisfy all applicable gates:

1. The complete portable suite finishes within 600 seconds on the reference
   Mac and in CI.
2. Branch coverage of `lmx/` remains at least 95%.
3. All public functionality has a bounded test; physics claims additionally
   have conservation, convergence, manufactured-solution, analytical, or
   independent-reference evidence.
4. Differentiable paths compare gradients with finite differences or an
   independent transpose/adjoint and control primal and transpose residuals.
5. CPU and GPU performance reports separate compilation from warm runtime,
   verify equivalent solutions, report memory, and state device placement.
6. Multi-device claims use real sharding and include one-device baselines,
   speedup, efficiency, and numerical parity.
7. The wheel contains no benchmark archives or generated media.
8. Dependencies use compatible release ranges unless an upstream regression
   is documented with an issue and a temporary bounded pin.

## Validation ladder

| Level | Required evidence | Current state |
|---|---|---|
| A: fully developed ducts | Hartmann, Shercliff, Hunt profiles; current and power balance; mesh convergence | verified for documented bounded cases |
| A-high: high-Ha ducts | all Samper Table I rows under one solver policy | accepted baseline |
| FreeMHD parity | matched equations, geometry, mesh ladder, observables, checksums | closed-channel bounded parity accepted |
| B1: fringing pipe | ALEX specification, conservative current, pressure projection, steady response | solver path improved; final external acceptance open |
| B2: fringing square duct | published profile/pressure observables and mesh ladder | specification frozen; acceptance open |
| Q2D and magnetic obstacle | model invariants first, independent turbulent/experimental reference second | research-stage |
| Blanket and wall models | unit audits, limiting cases, mesh convergence, current closure | research-stage |

Results that fail a strict gate remain useful diagnostic evidence but are never
presented as validation.

## Work sequence

### 1. Repository consolidation

- Remove internal campaign, study, external-code Docker, and status-dashboard
  trees from the source repository.
- Keep reusable inputs under `examples/cases/`.
- Retain only compact benchmark specifications, reference observables, and
  accepted summaries needed by tests.
- Reduce maintenance scripts to a small set of package/release commands;
  migrate reusable logic into `lmx/` and call it through the CLI.
- Collapse historical closure documents into current validation and
  development pages.
- Keep a small compressed image set in docs; serve movies and full result
  bundles from releases with hashes.
- Consolidate accepted work onto `main`, then delete superseded local and
  remote branches.

Exit: a new user can understand the repository from README, `examples/`, and
the docs index without learning internal campaign terminology.

### 2. Solver architecture slimming

- Split oversized implementation modules only at stable conceptual boundaries;
  remove compatibility wrappers and duplicate diagnostic/plotting paths first.
- Prefer pure array kernels, immutable configuration, explicit diagnostics, and
  short public functions with type hints and concise docstrings.
- Delegate generic linear algebra to current SOLVAX releases where parity,
  differentiation, compilation, and performance gates pass.
- Keep one implementation per numerical operation; retain alternatives only as
  tested backends with a documented purpose.
- Establish file, line, import-time, and wheel-size budgets in the architecture
  test.

Exit: public APIs are documented, implementation ownership is obvious, and
source lines decrease without reducing tested functionality.

### 3. Fast portable test architecture

- Merge tests by subsystem and behavior, not by historical task.
- Mark tests as unit, numerical, physics, regression, validation, or external.
- Run bounded representative grids in the full gate; keep expensive refinement
  and external-solver campaigns manual or scheduled.
- Cache JAX compilation where safe, group same-shape tests, and use process
  parallelism only where it reduces wall time and memory remains bounded.
- Record test count, skips, branch coverage, wall time, and slowest tests.

Exit: the full suite is under ten minutes, over 95% branch coverage, and each
functionality/physics claim maps to at least one test.

### 4. B1/B2 fringing acceptance

- The separated retained-mode B1 pressure path is promoted after
  small/medium/large parity and restart gates.
- The large-grid physical-convergence pilot reduces pressure work while
  preserving conservation and projection residuals.
- Run matched ALEX B1/B2 meshes and observables against digitized literature
  data and FreeMHD where formulations overlap.
- Assemble three source-identical mesh campaigns with the tested frozen
  uncertainty, refinement, wall/tolerance, and checksummed FreeMHD evaluator.
- Publish compact tables and compressed plots; place full fields and movies in
  a release.

Exit: both benchmarks have frozen inputs, reference provenance, mesh evidence,
observable tolerances, and reproducible accepted records.

### 5. Parallel CPU/GPU execution

- Make the production state explicitly shardable and remove host-staged
  collectives from timed regions.
- Validate single-device CPU/GPU equivalence before scaling.
- Measure fixed-size strong scaling on Mac CPU cores and the two office GPUs,
  including compilation, warm solve time, memory, speedup, and efficiency.
- Use multi-process GPU assignment only for independent cases; use JAX sharding
  for one distributed solve.
- Add performance regression thresholds only for stable, low-variance kernels.

Exit: at least one accepted production benchmark demonstrates useful
multi-device scaling with identical physics; unsupported paths say so clearly.

### 6. Research release

- Build docs with warnings as errors, build and inspect wheel/sdist, and install
  the wheel in a clean environment.
- Run the portable quality gate, selected external validations, gradient gates,
  and accepted CPU/GPU scaling protocol.
- Publish package artifacts plus checksummed research assets and cite exact
  commits, environments, references, tolerances, and known limitations.

Exit: the release is installable, reproducible, honestly scoped, and suitable
for external research use.

## Current checkpoint

- Portable gate: 789 passed, 8 expected external-data skips, 95.33% branch
  coverage, 181.2 seconds on the reference Mac.
- SOLVAX: compatible `>=0.8,<1`; latest tested package is 0.8.1.
- B1 retained modes: separated real `m=0` and complex `m=1..4` block factors
  pass factor parity and reduce medium restart time from 24.12 to 10.63 seconds.
- Large B1 pressure gate: a one-cycle physical-convergence pilot reduces the
  `21 x 24 x 64` solve-plus-restart ceiling from 768 to 669 Krylov iterations;
  all four projections, divergence, fixed-flow, and charge gates pass.
- B1 promotion: small factor parity, medium and large field/pressure-observable
  parity, and large restart gates pass; the compatible retained-modal solver is
  now the sole frozen B1 pressure path.
- GPU: accepted B2 checkpoint scales from 36.96 seconds on one A4000 to 22.23
  seconds on two (1.66x, 83.1% efficiency).
- Repository consolidation is active; historical campaigns and large studies
  are being moved out of the source tree before further solver slimming.
- Architecture budgets now cap module count, source/core lines, tracked bytes,
  lazy import time, curated examples, root exports, and release-wheel contents.

## Decision rule

Optimize the accepted physics path, not a proxy kernel. When two approaches are
equivalent, choose the one with fewer concepts, files, and lines. When evidence
is incomplete, label the feature research-stage and keep the stable surface
small.
