# LMX authoritative development plan

Status: 2026-07-15, through implementation commit `a97eeaf`. This is the single active
plan. It records accepted baselines, active gates, and stop/go criteria—not
campaign history. Completed campaign details belong in checksummed result
records and the validation or performance documentation.

## Product outcome

LMX will be a lightweight JAX code for accurate inductionless liquid-metal MHD
on CPUs and GPUs. It will be end-to-end differentiable for explicitly supported
objectives, research-grade for explicitly accepted physics, and honest about
research-stage paths.

The public repository has four working surfaces:

- `lmx/`: source and the `lmx` command;
- `tests/`: bounded unit, numerical, physics, regression, and workflow checks;
- `examples/`: small runnable Python and TOML workflows;
- `docs/`: theory, inputs, validation, performance, and development guidance.

A production claim requires analytical or manufactured verification,
conservation and convergence evidence, a literature or independent-code
comparison, reproducible inputs and outputs, and an uncertainty-aware
acceptance threshold. Coverage, a plot, or agreement between two copied
dictionaries is not physics validation.

## Operating contract

### Work small first

Every experiment declares one hypothesis, frozen metrics, a wall-time ceiling,
a stop rule, and a go/no-go threshold before launch. Escalate in this order:

1. static, analytical, parser, plotting-only, or tiny-grid check;
2. bounded one-device smoke;
3. medium confirmation only after the smoke passes;
4. fine, external-code, or multi-hour work only after all earlier gates pass,
   with restart and interim stop/go checks.

A failed bounded probe stops. It does not trigger an open-ended parameter
search. Reuse a checksummed accepted record whenever it answers the question.

### Test only what changed

- During development, run direct node IDs and static checks; target under two
  minutes.
- At a subsystem boundary, run affected modules and repository gates; target
  under five minutes.
- Once per coherent green tranche, and in CI/release, run the complete portable
  line-and-branch coverage gate; target 300 seconds and hard-limit 600 seconds.
- FreeMHD, accelerator, refinement, and large-data lanes remain manual or
  scheduled, with tiny portable representatives.

Combined line/branch coverage for `lmx/` stays at least 95%; 95.5% is the
engineering target before raising the enforced floor. Every public workflow
maps to one portable test and, where necessary, a numerical, physics, external,
gradient, or performance gate.

The current `unit` marker is not a reliable fast lane: expensive fringing and
autodiff nodes inherit it at module scope. Reclassify those nodes, or replace
the marker promise with an explicit fast manifest, before advertising
`pytest -m unit` as a quick check.

### Parallel work, evidence, and Git

Use subagents for disjoint literature, ownership, media, and test audits; one
integrator owns shared files. Run independent non-timing checks in parallel,
but measure performance alone on the target host. Commit and push each coherent
green tranche. Keep `main` authoritative and do not revive speculative branches
or worktrees after their findings are recorded.

Generated cases, logs, VTK, restarts, raw histories, source frames, large
meshes, and full-quality movies stay outside Git. Compact specifications,
evaluators, accepted summaries, and small presentation assets may be tracked;
large reusable artifacts go in checksummed releases.

## Authoritative baseline

| Surface | Accepted state | Open claim |
|---|---|---|
| Developed ducts | Hartmann, Shercliff, Hunt, and eight high-Ha rows pass analytical, conservation, and regression gates | preserve; do not generalize to arbitrary 3D flow |
| FreeMHD closed channels | bounded Shercliff/Hunt observables pass the frozen 1% finite-grid gate | this is not full FreeMHD parity |
| B1 ALEX pipe | retained-modal numerical evidence exists | exact-formulation parity remains open |
| B2 ALEX square duct | conservative momentum, mixed axial boundaries, explicit stress, compact corrected flux, exact restart/CFL/stopping state, and forced one-/two-CPU equivalence have bounded gates | canonical LMX/FreeMHD parity and scaling remain open |
| Matched B2 harness | deterministic LMX and FreeMHD inputs, pinned source evidence, independent observers, exact mismatch attribution, and OpenFOAM setup checks pass | execute the exact two-update smoke; production acceptance remains impossible for this role |
| Differentiation | selected objectives pass finite-difference or independent-transpose checks | no blanket end-to-end claim for every workflow |
| README/docs | concise feature-led README, corrected sourced comparison table, feature-specific visuals, and a seven-second Hunt loop | refresh B2/scaling panels only from accepted canonical records |
| SOLVAX | released 0.8.3 owns the generic algebra consumed by LMX | no further solver migration is required for the B2 smoke |

Current structure after the independent observer tranche at `a97eeaf`:

| Surface | Current | Active ratchet | CI hard ceiling |
|---|---:|---:|---:|
| package modules | 35 | no new module | 35 |
| package lines | 35,059 | below 34,850 after smoke cleanup | 35,100 |
| maintained-core lines | 8,015 | below 8,000 after smoke cleanup | 8,100 |
| test files / lines | 31 / 21,211 | no new file; below 21,000 after observer consolidation | 32 / 21,300 |
| maintenance scripts | 18 | 17 when the superseded SOLVAX acceptance freezer is retired | 18 |
| tracked checkout | 3,412,001 bytes | do not increase without a user-facing need | 4,194,304 bytes |

These ratchets must come from ownership deletion, shared helpers, or removal of
superseded behavior—not unreadable formatting or arbitrary test merging.

The portable-gate artifact keyed to `a97eeaf` records 788 passes, 8 expected
external-data skips, 95.04% combined line/branch coverage, and 208.2 seconds on
the reference Apple M4. It remains below the 300-second engineering target and
600-second hard limit. Coverage remains above the enforced floor but below the
95.5% engineering target.

## Priority 0: solver-free matched B2 harness — complete

Completed at `a97eeaf`:

- Schema 2 compares both observed contracts with a frozen canonical contract;
  two equal submitted dictionaries cannot self-certify.
- Artifact verification rejects escapes, aliases, links, overlaps, missing or
  empty inputs, special files, type mismatches, and changed content.
- At `a4c83ab`, the source materializer verifies FreeMHD commit
  `14b54a3e8e1a05b6ee4c98331995abaaae96e7a5` and scoped cleanliness. At
  `475789e`, the frozen evidence added the electric equation. The observer
  audit then added the vector-scheme macro and registration, for seven exact
  source files plus the manifest. The source-snapshot tree SHA-256 is
  `18c33bda110e92ad9d0e1872b776af373a18ba75075baffb88a9838432fcb333`.
- At `f033a4b`, the LMX materializer writes a deterministic strict JSON input;
  its loader reconstructs a real `CaseSpec`/`FringingProfile`, and its observer
  derives the contract without reading the expected contract.
- The compact FreeMHD input contains 392 cells across fluid and conducting-wall
  regions, direct block cell zones, exact B2 material/field/boundary facts, and
  fixed Euler controls for two updates. Generated meshes and fields stay out of
  Git.
- The FreeMHD observer independently parses input dictionaries and seven pinned
  source files plus their manifest. Both observers agree on the shared evaluator
  and normalized pressure contract.
- Ten one-sided mutations cover mesh, field, fluid, wall, velocity, pressure,
  electric boundary, scheme, iteration budget, and source content while proving
  that the opposite observation is unchanged.
- OpenFOAM setup utilities pass through mesh generation, region splitting,
  dictionary changes, field expression evaluation, and region listing. No
  FreeMHD or LMX solver process ran in this tranche.
- `harness-smoke` now reports schema, artifact, contract, and observation passes,
  while comparison and acceptance remain false. The role cannot promote itself
  to `b2-production`.

The LMX smoke realization is `L=U=rho=sigma_f=1`, `Ha=2900`, `N=540`,
`Re=Ha^2/N`, `Q=4`, wall thickness `0.02`, and wall conductivity `3.5`. It has
eight axial cells over `[-15,10]`, a `5x5` fluid cross-section plus one wall
cell on each side, `dt=1/540000`, and two updates ending at `step_limit`. These
are shared contract facts because the independent FreeMHD input reproduces them
exactly.

Exit met: deterministic inputs and source hashes, two independent observers,
one-sided mismatch attribution, the frozen non-accepting role, strict docs, and
one complete portable gate are green and pushed.

## Priority 1: freeze, then run one exact tiny B2 smoke

The current parity command handles Benchmark A and materializes B2 inputs, but
does not execute B2 or observe its outputs. The first solver-free checkpoint now
adds a real `--matched-b2-preflight` mode: the local bundle passes both
independent observers with contract SHA-256
`e30650045508cab8fce34a421e733591ff9f7503e322b54468dfdd300e11588a`.
Complete the remaining small solver-free tranche before launching either solver:

1. Extend the existing preflight with explicit B2 run and postprocess modes,
   without a new script or package module.
2. Define compact independent LMX and FreeMHD output observers for executed
   steps, effective `dt`, volume-mean and maximum Courant histories, stopping
   reason, mass/current closure, pressure gauge/orientation, restart identity,
   and the shared normalized pressure observable.
3. Add only the necessary FreeMHD pressure probes and mass/current surface
   sums. Bypass its demo plotter and VTK conversion; raw fields are not evidence
   for this gate.
4. Freeze smoke-specific tolerances before seeing solver output. Base exact
   controls and hashes on the input contract, restart tolerance on arithmetic
   precision, and closure/cross-code tolerances on the discrete schemes. Do not
   reuse the production ALEX experimental-error thresholds.
5. Rematerialize a clean external bundle, verify every hash and both input
   observations, and stop if any byte or contract fact differs.

The instrumentation checkpoint is now materialized and setup-valid in the
FreeMHD container. Sixteen pressure taps use the same adjacent-cell centers as
LMX; liquid mass/current sums use registered `rhoPhi`/`jn`. FreeMHD does not
register `jn` in the solid region, so outer and interface wall current are
derived as `-sigma_wall grad(potE) . Sf` through ordered native function
objects. Requesting a nonexistent solid `jn` is forbidden.

Then declare one ten-minute wall budget for both codes. Run LMX first as one
uninterrupted two-update path and as one update, restart, and one update; require
their compact outputs to agree. Run FreeMHD only if the LMX execution and output
contract pass, using the remaining budget. Invoke it directly with exact runtime
controls, a named container, deadline-aware termination, and unconditional
container cleanup; capture the effective controls. Stop at the first contract,
execution, restart, closure, or comparison failure.

The smoke keeps nonzero inertia, Lorentz force, diffusion, conducting-shell
topology, and the canonical axial conditions. It may reduce mesh and duration,
not equations. Commit only compact checksummed specifications, evaluator, and
summary; raw cases, logs, meshes, fields, and restarts remain external.

Implement this by consolidating the existing parity script: no new script or
package module, and no package-line increase. Use mocked child-process, timeout,
cleanup, and parser tests before the one real run.

Exit: the tiny run is contract-valid and numerically consistent under the
predeclared smoke gates, yet remains explicitly ineligible for production
acceptance. Failure returns to the first failed tiny gate; no medium run starts.

## Priority 2: unblock fast iteration and real strong scaling

After the accepted smoke, run these disjoint workstreams in parallel. Timing
measurements themselves run alone.

### CI critical path

The current serial-duration leaders are approximately 57.6 seconds for the
pipe weighted-modal physics node, 48.6 seconds for the reduced B2 path, 27.1
seconds for reduced B1, and 28.2 seconds for autodiff/nonrectangular work. Reduce their exact grids,
share safe compilation, or replace redundant integration work with independent
manufactured operators while preserving the same physics and tolerances.

Add the top ten node durations to the portable-gate record and warn on any
portable node above 45 seconds. Preserve the 300-second engineering target,
600-second hard limit, and at least 95% combined line/branch coverage. Prefer
parameterization and shared fixtures inside existing test files; do not create
another test file merely to move lines.

### Canonical sharding and performance

First prove one-/multi-device equivalence on the accepted smoke observables.
Then measure fixed-size warm strong scaling on:

- Mac CPU: 1, 2, and 4 JAX devices;
- office GPUs: 1 and 2 GPUs.

Separate compilation from repeated timings and report uncertainty, memory,
placement, speedup, and parallel efficiency. Independent-case multiprocessing
is throughput evidence, not strong scaling. Historical 1.66x two-GPU and CPU
surrogate results are diagnostic only and must not appear as canonical claims.

Production fields shard axially. Keep compact cell-shaped positive-face flux
plus one replicated inlet plane; exchange nonperiodic halos explicitly and do
not checkpoint duplicated `nx+1` arrays. Optimize only a profiled bottleneck on
the physics-valid path.

Exit: the full portable suite remains below ten minutes with no critical-path
surprise, and the accepted B2 path has equivalent observables plus useful
measured speedup on its target hardware.

## Priority 3: canonical B2 validation

The frozen numerical formulation is:

- transverse `B=(0,B_y(x),0)` for primary ALEX acceptance;
- conservative `fvm::div(rhoPhi,U)` inertia with Euler stepping, one
  `magSqr(U)`-derived `limitedLinear 1.0` vector limiter, and
  `cellLimited leastSquares 1.0` gradients;
- lagged explicit `div(mu*dev2(T(grad(U))))` stress on the momentum right-hand
  side;
- one inlet integral-flow condition, outlet zero-gradient velocity, inlet
  pressure Neumann, fixed outlet pressure, and zero normal current at both
  axial ends;
- the same oriented, pressure-corrected, area-integrated mass flux for
  projection and momentum convection;
- compact restart state containing velocity, pressure, face flux, previous
  scaled residual, relaxation, convergence streak, CFL/stopping state, and all
  required histories.

Advance exact coarse, medium, then fine meshes one level at a time. Each level
must pass literature/ALEX pressure, FreeMHD observable, conservation, restart,
wall-thickness, tolerance, steady-state, and mesh-change gates for the correct
reason. A Maxwell-consistent fringe field is a separately labelled sensitivity
study, not a replacement acceptance case.

Refresh README/docs B2 and scaling visuals only from compact accepted records.
Every image states validation status and provenance; no superseded result is
silently relabelled.

Exit: B2 has a three-mesh ladder, exact-source FreeMHD evidence, literature and
experimental comparison with uncertainty, reproducible environments, and a
stable accepted claim.

## Priority 4: B1 and remaining research functionality

Apply the proved harness to B1: exact tiny parity, coarse agreement, then medium
and one large confirmation only if required. Retained-modal results remain
numerical evidence, not exact-formulation evidence.

Magnetic obstacles, Q2D turbulence, blanket models, mapped/pipe geometries,
inverse design, and other research workflows retain bounded portable examples
and tests. Promote each independently only when its own analytical/numerical,
physics, gradient, and performance gates pass; do not let B2 acceptance
self-promote unrelated features.

## SOLVAX ownership

Keep `solvax>=0.8.3,<1`. PyPI and the latest GitHub tag are 0.8.3, and CI tests
both that minimum and the newest compatible release. SOLVAX `origin/main` is
prepared for 0.8.4 at `255d280`, while the untagged `release/0.8.4` worktree at
`4808695` contains the Anderson-weight API. Reconcile and publish those changes
through SOLVAX's separate release process before LMX consumes a 0.8.4-only API.
No tag or publication is authorized here.

LMX owns MHD equations, finite-volume stencils and limiters, geometry,
materials, interfaces, open-boundary and gauge semantics, corrected flux,
physical residuals, stopping/restart state, sharding policy, observables, and
acceptance. SOLVAX owns generic linear algebra after primal, residual,
transpose/gradient, JIT, placement, memory, and repeated timing gates pass.
Delete an LMX duplicate in the same tranche that adopts SOLVAX.

Near-term, released-0.8.3 candidates are deliberately small:

- replace manual pressure-preconditioner sums with
  `additive_preconditioner(..., weights=(1,...))`;
- consider `solvax.jacobi` for the five-point PCG preconditioner while
  preserving its tiny guard and `none` behavior.

Together these can save roughly 10–20 package lines; they do not justify
interrupting the matched harness. After 0.8.4 is published, use one
`anderson_weights` result for scaled fields and compact-flux histories, and use
`linear_solve(has_aux=True)` to retain momentum diagnostics without a final
extra matvec. Raise the minimum dependency in that same correctness tranche.

Benchmark-gated later, evaluate `block_thomas_factor_fn` for B1 modal setup and
`chunked_jacfwd` only if memory profiling identifies the retained-modal
Jacobian. Do not move finite-volume assembly into SOLVAX, and do not substitute
Fourier–Helmholtz, Newton–Krylov, generic multigrid, or affine fixed-point GMRES
into parity-critical paths without new topology, stopping, gradient, and timing
evidence. No credible SOLVAX-driven module deletion exists today.

## README, documentation, and media contract

Keep the README below 800 words: pitch, quickstart, a concise capability table,
and short visual sections for verified ducts, fields/geometries,
differentiation, research workflows, and scaling. There is no generic
“selected media” gallery; each plot or movie explains a specific feature.

Comparison-table cells describe the named native workflow, not what could be
implemented through arbitrary custom sources. Keep primary sources beside the
table. The current audit uses the FreeMHD paper/source, FreeMHD2 preprint, NekRS
26 documentation, NekRS MHD report, and NekRS GPU paper. Re-audit changing
capabilities before every release.

Aim for readable 6–8-second loops, stills below 100 KiB, tracked movies below
150 KiB where practical, and all tracked media below 1 MiB. Host source frames,
full-quality media, meshes, and raw outputs in checksummed releases. Put
provenance and acceptance status beside every asset.

## Release gate

Before release:

1. build docs with warnings as errors and verify links, media, and provenance;
2. run architecture, repository-layout, source-distribution, and wheel-content
   audits;
3. build wheel/sdist, inspect them, install the wheel in a clean environment,
   and run a smoke;
4. run the full portable coverage gate plus selected accepted external,
   gradient, restart, and scaling lanes;
5. publish exact commits, environments, references, tolerances, limitations,
   and release-asset hashes.

Hosted CI is a release blocker. Resolve the current GitHub Actions
billing/spending failure and require green Python 3.10 with minimum SOLVAX,
Python 3.13 with newest-compatible SOLVAX, strict docs, wheel, and release jobs.

Exit: LMX is small, installable, reproducible, honestly scoped, documented with
useful visuals, above 95% branch coverage, below the ten-minute portable-test
limit, and research-grade for every feature it labels accepted.
