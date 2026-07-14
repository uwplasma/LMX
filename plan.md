# LMX authoritative development plan

Status: 2026-07-14, through LMX commit `f033a4b`. This is the single active
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
| Matched B2 harness | pinned source snapshot and a real deterministic LMX input/loader/observer are complete | FreeMHD input/observer and cross-observer agreement remain |
| Differentiation | selected objectives pass finite-difference or independent-transpose checks | no blanket end-to-end claim for every workflow |
| README/docs | concise feature-led README, corrected sourced comparison table, feature-specific visuals, and a seven-second Hunt loop | refresh B2/scaling panels only from accepted canonical records |
| SOLVAX | released 0.8.3 owns the generic algebra consumed by LMX | no further solver migration is required for the B2 smoke |

Current structure at `f033a4b`:

| Surface | Current | Active ratchet | CI hard ceiling |
|---|---:|---:|---:|
| package modules | 35 | no new module | 35 |
| package lines | 34,838 | at most 34,850 while adding the FreeMHD observer; below 34,800 after smoke cleanup | 35,000 |
| maintained-core lines | 8,002 | below 8,000 after smoke cleanup | 8,100 |
| test files / lines | 31 / 21,023 | no new file; below 21,000 after observer consolidation | 32 / 21,300 |
| maintenance scripts | 18 | 17 when the superseded SOLVAX acceptance freezer is retired | 18 |
| tracked checkout | 3,378,869 bytes | do not increase without a user-facing need | 4,194,304 bytes |

These ratchets must come from ownership deletion, shared helpers, or removal of
superseded behavior—not unreadable formatting or arbitrary test merging.

The last committed portable-gate artifact records 784 passes, 8 expected
external-data skips, 95.28% combined line/branch coverage, and 149.9 seconds on
the reference Apple M4. It predates the current observer tranche. Refresh one
single authoritative result after the FreeMHD observer is complete, and update
the README/docs from that record only.

## Priority 0: finish the solver-free matched B2 harness

No solver run is authorized until this gate is committed and green.

Completed foundation:

- Schema 2 compares both observed contracts with a frozen canonical contract;
  two equal submitted dictionaries cannot self-certify.
- Artifact verification rejects escapes, aliases, links, overlaps, missing or
  empty inputs, special files, type mismatches, and changed content.
- At `a4c83ab`, the source materializer verifies FreeMHD commit
  `14b54a3e8e1a05b6ee4c98331995abaaae96e7a5`, scoped cleanliness, and exact
  hashes for four pinned source files. The generated five-file source tree hash
  is `366e3a6a9464192183db9383dead47f5e3c8719a065ce4982e7c2b4586306289`.
- At `f033a4b`, the LMX materializer writes a deterministic strict JSON input;
  the loader reconstructs a real `CaseSpec`/`FringingProfile`; and the observer
  derives the contract without reading the expected contract.

Next tranche, still without running either solver:

1. Materialize FreeMHD independently from the small multi-region `hunt_demo`
   dictionary skeleton. Do not copy generated fields or treat its original
   million-cell case as B2 evidence.
2. Rewrite the block mesh, all-conducting shell, ALEX field, material values,
   inlet-flow/outlet-pressure/electric boundaries, and fixed-step controls.
   Require `alpha=1`, zero gravity, constant temperature/properties, laminar
   flow, zero-duration field ramp, conservative current, no velocity clipping,
   and no adaptive step.
3. Preserve and independently parse the exact `Euler`, `limitedLinear 1.0`,
   and `cellLimited leastSquares 1.0` dictionaries.
4. Derive the FreeMHD contract from mesh, topology, materials, field,
   boundaries, phase/thermal, numerical, and control dictionaries. The observer
   must not read a manifest, expected contract, or LMX input.
5. Mutate one side at a time—mesh face, field sample, material, wall,
   inlet/outlet/electric boundary, step/corrector budget, and pinned source
   byte—and require the exact failed contract path while the other observer is
   unchanged.
6. Freeze `harness-smoke` only after both real observers agree. This role must
   report `contract_pass=True` and `acceptance_pass=False`; it can never promote
   itself to `b2-production`.

The LMX smoke realization is `L=U=rho=sigma_f=1`, `Ha=2900`, `N=540`,
`Re=Ha^2/N`, `Q=4`, wall thickness `0.02`, and wall conductivity `3.5`. It has
eight axial cells over `[-15,10]`, a `5x5` fluid cross-section plus one wall
cell on each side, `dt=1/540000`, and two updates ending at `step_limit`. These
become shared contract facts only when the independent FreeMHD input reproduces
them; otherwise fix the first materializer discrepancy rather than weakening
the contract.

Exit: deterministic FreeMHD input and source hashes, two independent observers,
one-sided mismatch attribution, frozen smoke role, strict docs, one complete
portable gate, and a pushed commit. No FreeMHD process has run.

## Priority 1: run one exact tiny B2 smoke

Use the existing parity command and the committed materialized inputs. Declare
a ten-minute wall ceiling for the entire LMX-plus-FreeMHD smoke and stop at the
first contract, execution, or closure failure.

Compare:

- independently observed contracts and artifact hashes;
- executed steps, effective `dt`, volume-mean and maximum Courant histories,
  convergence streak, and `step_limit` reason;
- mass and current closure, pressure gauge/orientation, and identical normalized
  ALEX pressure observable definitions;
- restart state where the two-step path crosses a checkpoint boundary.

The smoke keeps nonzero inertia, Lorentz force, diffusion, conducting-shell
topology, and the canonical axial conditions. It may reduce mesh and duration,
not equations. Commit only compact checksummed inputs and a summary; raw cases,
logs, fields, and restarts remain external.

Exit: the tiny run is contract-valid and numerically consistent, and remains
explicitly ineligible for production acceptance. Failure returns to the first
failed tiny gate; no medium run starts.

## Priority 2: unblock fast iteration and real strong scaling

After the accepted smoke, run these disjoint workstreams in parallel. Timing
measurements themselves run alone.

### CI critical path

The current serial-duration leaders are approximately 78.5 seconds for the
pipe weighted-modal physics node, 42.7 seconds for the reduced B2 path, 27.7
seconds for reduced B1, and 22.0 seconds for autodiff. Reduce their exact grids,
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
both that minimum and the newest compatible release. The unpublished 0.8.4
candidate is the dedicated `release/0.8.4` worktree at `4808695`; the ordinary
local `SOLVAX/main` checkout is dirty and divergent and is not release evidence.
Reconcile the SOLVAX branches and publish through its separate release process
before LMX consumes a 0.8.4-only API. No tag or publication is authorized here.

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
