# LMX authoritative development plan

Status: 2026-07-15. The current source/scaling baseline is `3e53d8b`; the latest
complete portable-gate artifact is keyed to `ad9b9b3`.
The exact two-update LMX/FreeMHD B2 smoke, one-/two-/four-CPU-device equivalence,
deterministic one-/two-GPU equivalence, and portable coverage gate are green.
The smoke closes bounded orchestration and comparison, not production B2
acceptance. The current `128 x 67 x 67` GPU rung closes deterministic sharding
and a bounded two-update calibration, not fast or production scaling. This
single active plan records accepted baselines, active gates, and
stop/go criteria—not campaign history. Completed campaign details belong in
checksummed result records and the validation or performance documentation.

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
| B2 ALEX square duct | conservative momentum, mixed axial boundaries, explicit stress, compact corrected flux, strict smoke replay, exact device equivalence, and current CPU/GPU two-update calibrations have bounded gates | production parity and steady-production scaling remain open |
| Matched B2 harness | deterministic inputs, pinned sources, independent observers, and native two-update LMX/FreeMHD execution pass every frozen schema-3 smoke gate | production acceptance and mesh convergence remain open |
| Differentiation | selected objectives pass finite-difference or independent-transpose checks | no blanket end-to-end claim for every workflow |
| README/docs | concise feature-led README, corrected sourced comparison table, feature-specific visuals, and a seven-second Hunt loop | refresh B2/scaling panels only from accepted canonical records |
| SOLVAX | released 0.8.3 owns the generic algebra consumed by LMX | no further solver migration is required for the B2 smoke |

Current structure after moving the script-only Benchmark-A auditor out of the
package and retiring the undocumented non-projection rectangular autodiff lane
(the richer projection and target-driven paths remain):

| Surface | Current | Active ratchet | CI hard ceiling |
|---|---:|---:|---:|
| package modules | 35 | no new module | 35 |
| package lines | 34,872 | stay below 35,000 through the scaling tranche | 35,100 |
| maintained-core lines | 8,034 | below 8,000 after smoke cleanup | 8,100 |
| test files / lines | 31 / 21,282 | no new file; below 21,000 after fixture consolidation | 32 / 21,300 |
| maintenance scripts | 18 | 17 when the superseded SOLVAX acceptance freezer is retired | 18 |
| tracked checkout | 3,508,671 bytes | do not increase without a user-facing need | 4,194,304 bytes |

These ratchets must come from ownership deletion, shared helpers, or removal of
superseded behavior—not unreadable formatting or arbitrary test merging.

The portable-gate artifact keyed to `ad9b9b3` records 823 passes, 8 expected
external-data skips, 95.0037% combined line/branch coverage, and 152.1 seconds on
the reference Apple M4. It remains below the 300-second engineering target and
600-second hard limit. Coverage remains above the enforced floor but below the
95.5% engineering target. The six-worker record reports 54.6 seconds for the
weighted-modal node and 41.1 seconds for reduced B2. Parallel JUnit durations
are diagnostic rather than isolated timings, but weighted-modal now exceeds the
45-second warning level and is the next CI critical-path target.

## Priority 0: solver-free matched B2 harness — complete

Core schema-2 harness completed at `a97eeaf`; source, setup,
instrumentation, tolerance, and replay-observer checkpoints continued through
`9b41441`:

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

## Priority 1: matched B2 smoke — complete

The refreshed external bundle keyed to LMX source commit `3a22078` passed every
permitted schema-3 gate under a shared 600-second deadline. LMX ran first on one
JAX device in 13.12 seconds; native FreeMHD then ran with two MPI ranks in 7.06
seconds. Both executed two fixed Euler updates and stopped at `step_limit`.

The independent observers report zero LMX restart difference, machine-precision
mass/current closure, nonzero interface-current activity, and closely matching
Courant histories. Cross-code normalized pressure differences are 0.00452 RMS
and 0.01092 maximum, well inside the predeclared 0.16 and 0.32 smoke limits.
Schema, artifacts, contracts, observations, execution, and comparison all pass
with no failed checks. `acceptance_pass` remains false solely because
`harness-smoke` is intentionally ineligible for production acceptance.

The compact tracked summary is
`benchmarks/results/b2-freemhd-harness-smoke-20260715.json`. The 97-file,
1.64 MB raw bundle remains outside Git with a recorded tree hash; future runs
must also record the immutable container image digest and orchestration wall
clock directly. No medium B2 physics run starts from this result.

Exit met: the exact tiny run is contract-valid and numerically consistent under
the frozen smoke gates. Production parity, a three-mesh ladder, experimental
acceptance, and production/steady strong scaling remain open.

## Priority 2: unblock fast iteration and real strong scaling

After the accepted smoke, run these disjoint workstreams in parallel. Timing
measurements themselves run alone.

### CI critical path

The modal pipe test reuses one physical projection and verifies direct
mode-factor algebra without a second integration run. In the latest six-worker
gate it nevertheless reports 54.6 seconds, versus 41.5 seconds for reduced B2
and 25.5 seconds for reduced B1. Isolated measurement attributed roughly half
of that modal duration to worker contention. Reducing only its manufactured
radial grid, while preserving every physical and independent-factor assertion,
lowered the isolated weighted path from 33.7 to 23.5--26.1 seconds and the base
path to 7.5--8.9 seconds. Next profile reduced B2, then the autodiff nodes; refresh the
full-gate wall time only after the next coherent test tranche.

Add the top ten node durations to the portable-gate record and warn on any
portable node above 45 seconds. Preserve the 300-second engineering target,
600-second hard limit, and at least 95% combined line/branch coverage. Prefer
parameterization and shared fixtures inside existing test files; do not create
another test file merely to move lines.

### Canonical sharding and performance

Exact accepted-smoke observables now agree on one, two, and four forced Mac CPU
devices and on one and two deterministic RTX A4000 GPUs. The repair makes flux,
vector, embedding, initialization, CFL, and relaxation layouts explicit and
keeps compact flux components separate until explicit packing. Both device
ladders have exact restart replay; the GPU pressure observable differs by at
most `1.05e-14`. Deterministic GPU correctness uses
`--xla_gpu_exclude_nondeterministic_ops`; default GPU mode is the separate
timing lane. Compact
records are `benchmarks/results/b2-{cpu,gpu}-device-equivalence-20260715.json`.

An explicit one-device request now uses the same named-sharding kernels as the
multi-device path. With that fair baseline, the current-formulation CPU ladder
passes validation, placement, exact restart, and device-equivalence gates at
`16 x 7 x 7`, `64 x 15 x 15`, and `128 x 31 x 31`. The two smaller grids are
communication-bound. At `128 x 31 x 31`, warm medians are 0.768, 0.649, and
0.647 seconds on 1/2/4 devices: 1.18x and 1.19x speedups, with no useful gain
beyond two devices. This is a two-update scaling calibration, not a steady
production-speed claim. The compact record is
`benchmarks/results/b2-cpu-strong-scaling-20260715.json`.

The current `128 x 67 x 67` default-XLA rung passes validation, placement,
restart, and device equivalence on one and two RTX A4000 GPUs. Warm medians are
21.09 and 16.40 seconds with CV below 1.9%, a bounded 1.29x two-update speedup.
The stricter deterministic probe isolated `4.40e-6` relative reduction noise
to corrected face flux; a direct-three versus restart-one-plus-two trajectory
preserved every primary field exactly and reduced that difference to
`6.25e-7`. The calibration therefore requires nonflux state within `1e-12`
and flux within `1e-6` absolute and `1e-5` relative. The exact tiny harness
keeps its stricter all-field gate. The compact record is
`benchmarks/results/b2-gpu-scaling-calibration-20260715.json`; two updates and
the shared host preclude a publishable or production-speed claim.

Next profile one warm 1/2-GPU update and separate projection, electric, momentum,
and halo costs. Only a measured compute/communication bottleneck may justify
one larger bounded rung; steady-production scaling remains the promotion gate.

Separate compilation from repeated timings and report uncertainty, memory,
placement, speedup, and parallel efficiency. Independent-case multiprocessing
is throughput evidence, not strong scaling. Historical 1.66x two-GPU and CPU
surrogate results are diagnostic only and must not appear as canonical claims.

Production fields shard axially. Keep compact cell-shaped positive-face flux
plus one replicated inlet plane; exchange nonperiodic halos explicitly and do
not checkpoint duplicated `nx+1` arrays. Optimize only a profiled bottleneck on
the physics-valid path.

Exit: the full portable suite remains below ten minutes with no critical-path
surprise, and the accepted B2 path retains equivalent observables plus useful,
uncertainty-aware speedup on its target hardware. CPU correctness and bounded
CPU scaling calibration and bounded GPU state/flux calibration are complete;
steady-production scaling remains open.

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

The device-cut audit compared released 0.8.3 with the pending 0.8.4 PCG and
found identical Krylov code; generic two-device standard and single-reduction
PCG both pass. The B2 failure was in LMX-owned layout transitions: eager
embedding, component extraction, slicing, packing, relaxation, and restart
staging across production shards. It was not a generic Krylov or SOLVAX defect,
so no SOLVAX patch or dependency bump is warranted for this fix.

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
