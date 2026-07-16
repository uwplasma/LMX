# LMX authoritative development plan

Status: 2026-07-15. The current two-update B2/FreeMHD smoke and schema-5
stopping contract are keyed to `0ab33b2`; current one-/two-/four-CPU-device
equivalence is keyed to `4c94389`.
The post-map nonlinear momentum residual and
restart schema 4 were keyed to `e6834ee`; schema 5 now versions normalized
stopping. Its fixed-relaxation memory reduction
is keyed to `791e496`, and its exact operator contract is keyed to `2d0fb50`.
Commit `3e731fa` removes the projection reconstruction floor and refreezes the
64x pseudo-time cap on a warm same-state ladder.
The latest complete portable gate exercised source `1a52c6d`. CPU/GPU calibration at
`413185a` remains historical; deterministic GPU equivalence at `3a22078` has
been replaced by the current `8b6f97d` result and the refreshed calibration
record committed at `3311d6d`. The isolated compiler trace keyed to `f379f6b`
is historical and cannot attribute the current accepted path.
The first fresh canonical-mesh coarse trajectory using the current formulation
passes conservation and all linear-solver gates but reaches its 128-update
bound before steady convergence. Its single authorized continuation preserves
those gates but misses its precommitted residual target; it is not promoted.
The smoke closes bounded orchestration and comparison, not production B2
acceptance. Current-source 1/2-GPU correctness and the `128 x 67 x 67`
calibration pass. The trace-authorized validation fusion reaches 1.159x but
misses the 1.2x promotion gate; the
`256 x 67 x 67` rung remains historical, so no production scaling speedup is
claimed. This
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
| B2 ALEX square duct | conservative momentum, mixed axial boundaries, explicit stress, compact corrected flux, post-map physical residual, strict schema-5 replay, refrozen 64x cap, and current CPU/GPU device equivalence have bounded gates | tighter-reference stopping calibration, production parity, and steady-production scaling remain open |
| Matched B2 harness | deterministic inputs, pinned sources, independent observers, and native two-update LMX/FreeMHD execution pass every frozen schema-3 smoke gate | production acceptance and mesh convergence remain open |
| Differentiation | selected objectives pass finite-difference or independent-transpose checks | no blanket end-to-end claim for every workflow |
| README/docs | concise feature-led README, sourced comparison table, feature-specific visuals, seven-second Hunt/Q2D loops, and Li/AlN convergence | refresh B2/scaling panels only from accepted canonical records |
| SOLVAX | released 0.8.3 owns the generic algebra consumed by LMX | no further solver migration is required for the B2 smoke |

Current structure after moving the script-only Benchmark-A auditor out of the
package, retiring the undocumented non-projection rectangular autodiff lane,
deleting the superseded SOLVAX acceptance freezer and its sole test, and
replacing the single-owner Benchmark-B freeze generator with direct hash gates,
folding Benchmark-A evidence freezing into its ladder analyzer, and folding the
single-owner Samper freezer into its campaign runner, and making `Diagnostics`
the single owner of standard NPZ/restart diagnostic fields, and consolidating
four manual-validation workflow stubs into one behavior-preserving fixture (the
immutable evidence, richer projection, and target-driven paths remain), and
deleting stale test-only velocity-statistics, solver-mask, pipe-Laplacian, and
symmetry wrappers:

| Surface | Current | Active ratchet | CI hard ceiling |
|---|---:|---:|---:|
| package modules | 35 | no new module | 35 |
| package lines | 34,968 | stay below 35,000 while preserving the physical residual | 35,100 |
| maintained-core lines | 7,931 | stay below 8,000 | 8,000 |
| test files / lines | 30 / 20,882 | no new file; stay below 21,000 | 31 / 21,100 |
| maintenance scripts | 13 | no new script without retiring an owner | 13 |
| tracked checkout | 3,565,237 bytes | do not increase without a user-facing need | 4,194,304 bytes |

These ratchets must come from ownership deletion, shared helpers, or removal of
superseded behavior—not unreadable formatting or arbitrary test merging.

The portable-gate artifact keyed to `1a52c6d` records 818 passes, 8 expected
external-data skips, 95.0274% combined line/branch coverage, and 157.4 seconds on
the reference Apple M4. The 1.9% change from the previous 154.5-second record is
within run-to-run noise; the gate remains below the 300-second engineering
target and 600-second hard limit. Coverage
remains above the enforced floor but below the 95.5% engineering target. The
six-worker record reports 49.4 seconds for reduced B2 and 53.3 seconds for
weighted modal; these concurrent durations identify contention rather than
isolated regressions, so no scheduling change is promoted from this run.

## Priority 0: solver-free matched B2 harness — complete

The schema-3 harness materializes deterministic LMX and 392-cell FreeMHD inputs,
pins FreeMHD commit `14b54a3e8e1a05b6ee4c98331995abaaae96e7a5`, verifies the
source snapshot, and derives each observed contract independently. Frozen
one-sided mutations cover geometry, materials, fields, boundaries, numerics,
and source drift. The `harness-smoke` role is deliberately unable to promote
itself to `b2-production`.

Exit met: source, artifact, contract, observer-independence, and setup gates are
green. Exact implementation history belongs in the tracked harness record and
validation documentation.

## Priority 1: matched B2 smoke — complete

The record keyed to LMX `45bff84` and the pinned FreeMHD source runs two fixed
Euler updates in 3.32 seconds on one JAX device and 6.76 seconds on two native
FreeMHD MPI ranks. Restart, mass/current closure, interface-current activity,
and Courant gates pass. Cross-code normalized pressure differences are 0.00452
RMS and 0.01092 maximum, within the frozen 0.16 and 0.32 smoke limits.

Authoritative evidence is
`benchmarks/results/b2-freemhd-harness-smoke-20260715.json`; its 1.64 MB raw
bundle remains outside Git. Exit met for orchestration and two-update numerical
consistency only. Production parity, the canonical three-mesh ladder,
experimental acceptance, and steady scaling remain open.

## Priority 2: unblock fast iteration and real strong scaling

After the accepted smoke, run these disjoint workstreams in parallel. Timing
measurements themselves run alone.

### CI critical path

The modal pipe test reuses one physical projection and verifies direct
mode-factor algebra without a second integration run. In the latest six-worker
gate it reports 53.4 seconds, versus 49.5 seconds for reduced B2 and 24.9
seconds for reduced B1. Isolated measurement attributes most of that tail to
worker contention: reducing only the manufactured modal grid lowered its
weighted path to 23.5--26.1 seconds, and the unchanged reduced-B2 restart and
physics node now measures 14.8 seconds alone. Preserve those coverage-rich
tests. A fresh-process A/B of the same six expensive JAX nodes takes 37.69
seconds with six workers and 36.41 with four. The 3.4% change misses the frozen
10% promotion threshold, so retain six-worker work stealing; module grouping
would imbalance the fringing and autodiff owners. Profile another node only if
an isolated measurement crosses the 45-second trigger.

Keep the top ten concurrent node durations in the portable-gate record and
treat any node above 45 seconds as a critical-path review trigger. The suite
driver's warning remains the separate 300-second wall-time warning. Preserve
the 300-second engineering target,
600-second hard limit, and at least 95% combined line/branch coverage. Prefer
parameterization and shared fixtures inside existing test files; do not create
another test file merely to move lines.

### Canonical sharding and performance

Current accepted-smoke observables agree on one, two, and four forced Mac CPU
devices within `5.93e-15`, with exact restart and all pressure solves green. The
one-/two-GPU equivalence predates the terminal fix and is historical. Its
current-source replacement passes repeat, restart, conservation, and
equivalence gates. The repair makes flux,
vector, embedding, initialization, CFL, and relaxation layouts explicit and
keeps compact flux components separate until explicit packing. The CPU ladder
has exact restart replay. Deterministic GPU probes use
`--xla_gpu_exclude_nondeterministic_ops`; default GPU mode remains a separate
timing lane and needs a fresh isolated calibration before promotion. Compact
records are `benchmarks/results/b2-{cpu,gpu}-device-equivalence-20260715.json`.

An explicit one-device request uses the same named-sharding kernels as the
multi-device path. On the pre-terminal-fix source, the `128 x 31 x 31` rung passes
validation, placement, exact restart, and device-equivalence gates. Warm
medians are 0.857, 0.652, and 0.633 seconds on 1/2/4 devices: 1.31x and 1.35x
speedups, with modest gain beyond two devices. This is a two-update scaling calibration, not a steady
production-speed claim. The compact record is
`benchmarks/results/b2-cpu-strong-scaling-20260715.json`.

The pre-terminal-fix `128 x 67 x 67` default-XLA rung passes validation, placement,
restart, and device equivalence on one and two RTX A4000 GPUs. Diagonal momentum
preconditioning reduced its initial warm medians from 21.14 to 3.09 seconds on
one GPU and from 11.98 to 3.19 seconds on two: 6.84x and 3.75x absolute
improvements. The initial fixed-grid ratio is 0.968x, and seven-repeat
one-device confirmations on both physical cards have 6.0--12.5% CV because
unrelated workloads share the host. No current GPU strong-scaling speedup is
therefore promoted.
The stricter deterministic probe isolated `4.40e-6` relative reduction noise
to corrected face flux; a direct-three versus restart-one-plus-two trajectory
preserved every primary field exactly and reduced that difference to
`6.25e-7`. The calibration therefore requires nonflux state within `1e-12`
and flux within `1e-6` absolute and `1e-5` relative. The exact tiny harness
keeps its stricter all-field gate. The compact record is
`benchmarks/results/b2-gpu-scaling-calibration-20260715.json`; two updates and
the shared host preclude a publishable or production-speed claim.

Earlier complete traces showed first that eager sharded current-flux diagnostics
and then momentum line solves dominated. JIT fusion and diagonal momentum
scaling remove both bottlenecks. Dense-reference, implicit-gradient, restart,
placement, and equivalence gates pass, while the implementation deletes ten net
source lines. On the new complete `128 x 67 x 67` trace, mixed pressure
projection occupies 66.0% of named solver wall time, electric potential 28.5%,
and momentum 5.5%. Projection and electric device activity are each about 60%
tridiagonal/PCR work; momentum has no tridiagonal events. Device-activity shares
sum overlapping devices and streams and are not wall-time shares.

The single trace-justified `256 x 67 x 67` calibration is complete. It passes
validation, exact restart, placement, and device equivalence; warm medians are
8.474 and 7.534 seconds with CV below 3.7%, a stable but sub-threshold 1.125x
speedup. Stop larger blind rungs. Bounded `64 x 15 x 15` probes rejected one
transverse line plus axial mean (40% slower), SOLVAX Jacobi plus axial mean
(22% slower), and a relaxed internal projection tolerance (only 1.0% faster
and outside the frozen `1e-12` contract). An exact mixed-boundary DCT-IV coarse
correction passed dense, symmetry, gradient, manufactured-flow, and sharding
gates, but gained only 0.47% and missed the strict restart-state tolerance;
it is also rejected. Do not drop the accepted line blocks, weaken tolerance,
or revive additive coarse corrections. A single released-SOLVAX
batch for both equal-length transverse systems is also exact, symmetry-safe,
gradient-safe, and restart-exact, but is 1.0% slower than its paired control;
do not revive launch-only batching. Pressure-PCG residuals, relative residuals,
iterations, convergence, and status remain retained without another solve or
synchronization phase.

The bounded `8 x 4 x 3` current-source CPU probe is green on one and two forced
devices: both converge in 24 PCG iterations, their maximum relative residuals
agree to `6.2e-19`, and the full signatures differ by at most `5.33e-15`. For
`k=24`, the source-level model is 26 global-reduction stages, 25 preconditioner
applications, 50 transverse line solves, 25 axial-mean line solves, and 26
pressure halo pairs per projection. These are algorithmic counts, not device
kernel counts. The frozen `64 x 15 x 15` GPU phase records 33 and 35 iterations
on both one and two GPUs: 72 reduction stages, 70 preconditioner applications,
140 transverse line invocations, 70 axial-mean invocations, and 72 halo pairs
over two updates. ABBA diagnostic/control ratios are 0.902 and 1.039 with up to
27% shared-host CV, so retention is within noise and no speedup is claimed. The
two-GPU tiny rung is correctly collective-dominated.

The isolated `128 x 67 x 67` compiler trace is complete on two RTX A4000s.
Warm and traced pressure histories agree within `9.2e-18`; both converge in
204/207 iterations, both GPU tracks have identical named kernel counts, and
only 228,296 of four million permitted events are used. Projection takes
0.554/0.567 seconds: fixed-coefficient tridiagonal kernels occupy 74.8--75.4%
of normalized device activity, while all collectives occupy only 8.8--9.2%.
The same trace closes electric attribution without another simulation. Its
95/82-iteration solves take 0.291/0.247 seconds, with 66.8--67.5%
tridiagonal and 5.6--6.6% collective activity. Communication tuning therefore
stops for both phases.

Removing only the replicated axial-mean correction retains validation but
raises pressure work from 33--35 to 180 iterations and is 6.1% slower; reject
it before a full trace. Both transverse SOLVAX line systems are the measured
cost and the required convergence mechanism. Released SOLVAX 0.8.3 exposes no
reusable scalar factor/apply path for its fused GPU tridiagonal solve, so stop
B2 preconditioner microprobes rather than replace the vendor kernel with an
unevidenced sequential implementation. The next physics work is Priority 3,
not another small solver variant.

The current `8 x 7 x 7` refresh passes validation and exact restart on one RTX
A4000. On two RTX A4000s it places every production field on two real shards
and every pressure solve converges, but interface-current balance rises to
`55.28` and restart state/flux errors reach `2.44e-6`/`8.75e-5`. Four identical
same-process solves alternate A/B/A/B; the first difference is a `1.55e-9`
step-one axial-velocity perturbation confined to the second shard, which the
high-Ha second update amplifies. The failure persists on an idle host, with
standard electric PCG, without the electric coarse correction, and after full
output synchronization. The root cause was LMX passing a default-device JAX
scalar into fixed relaxation of sharded state; commit `8b6f97d` keeps the known
factor as a compile-time scalar and skips the identity first update. Four
two-GPU repeats, restart, conservation, placement, and 1/2-device equivalence
then pass exactly or within `1.02e-14`. This was not a SOLVAX defect. The
scaling worker now fails closed on a compact, gauge-invariant, axial-station
signature across every cold and warm repeat; it
passes exactly on the repaired CPU and GPU paths. The frozen failing record
retains the pre-fix `3183.12` signature for regression provenance.

The isolated current-source `128 x 67 x 67` rung passes physical-repeat,
restart-state/flux, placement, convergence, and device-equivalence gates. The
corrected end-to-end public-solve medians at `e3923a2` are 2.797 and 2.690
seconds with CV below 0.4%, only 1.040x speedup. The external repeat signature
is outside timing, while public station history and validation remain inside.
The matched trace passes physical signature and convergence/status gates, with
iteration drift at most one. Its 1/2-GPU spans are 2.925/2.790 seconds, but the
three core phases total 2.647/1.771 seconds, a 1.495x speedup. Two-GPU post-map
work is 0.685 seconds versus 0.040 on one GPU; repeated validation transfers
account for about 0.520 seconds. The 1.2x end-to-end gate requires at most
2.330 seconds, so the actionable recovery budget is 0.360 seconds.

The single authorized validation-transfer optimization reuses the already
materialized station history for host metrics and deletes 12 package lines.
Its matched confirmation passes every fail-closed gate and reduces two-GPU
time by 0.290 seconds, but 2.780/2.400-second medians yield only 1.159x. The
two-GPU promotion limit is 2.317 seconds, missed by 0.083. Retain the smaller
implementation and stop: no larger rung, further validation tweak, or kernel
tuning is authorized without revising this plan. Core phases scale at 1.510x;
the public end-to-end API remains post-map host-transfer limited. Schema-6
Anderson is now the next bounded workstream after SOLVAX publication.

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
CPU/GPU calibration methods and current CPU/GPU correctness are green. Current
fixed-grid GPU scaling misses its promotion threshold after the single
trace-authorized confirmation, so the ladder is stopped.

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

The first fresh current-formulation coarse baseline at `102 x 77 x 77` runs on
two RTX A4000 shards in 131.22 seconds. All 128 pressure solves converge with
complete diagnostics; mass, divergence, charge, and boundary-current gates
pass. The nonlinear residual falls monotonically from `0.9621` to `1.4302e-3`
but does not reach `5e-5`, so the run correctly stops at the 128-update bound
with no steady streak. Its last 16 updates are strictly decreasing with a
log-linear fit of `R^2=0.9999998`; extrapolation suggests a first crossing near
step 519 but is not acceptance evidence. The compact record is
`benchmarks/results/b2-current-coarse-baseline-20260715.json`; the 45.7 MB
restart remains outside Git.

The one authorized exact continuation from restart hash `1d6491e3...` reaches
step 256 in another 130.92 seconds. All 256 pressure and electric solves,
two-shard placement, and conservation remain green; the residual remains
strictly decreasing. It ends at `7.1081e-4`, however, 29.24% above the
precommitted `5.5e-4` gate. Its tail log slope is only `0.5003` of the first
chunk's, moving the non-authoritative crossing estimate from step 519 to about
874. No further continuation is authorized.

The solver-free ownership audit finds a slow velocity pseudo-time mode, not a
physical time-to-steady claim: `u` controls 255 of 256 updates, the effective
`dt=1.85185e-6` is 5,400 times below the requested value, and the stopping
observable is the raw unrelaxed velocity update rather than an equation-error
norm. Fixed relaxation and flux scaling do not change through the tail.

The audit also found a separate chunk-boundary defect: terminal updates skipped
Aitken and flux acceleration. The predeclared `7 x 7 x 7` probe failed in 17.77
seconds, with state/history difference `1.0` and compact-flux relative L2
`1.48e-5`. B2 now accelerates terminal updates too. Direct four versus
serialized terminal-two plus resumed-two passes every state, history, and flux
gate at `1e-12`; the exact current LMX/FreeMHD smoke also remains green with
pressure RMS/Linf `0.004518/0.010917`.

Pre-fix coarse restarts are diagnostic-only and cannot seed the corrected
trajectory. The predeclared `dt` versus `dt/2` probe passes in 13.07 seconds.
Müller and Bühler's electromagnetic scaling defines the B2 map defect as
`max velocity update/(N dt)` because `L=U0=1` and `N=540`; FreeMHD/OpenFOAM and
NekRS residual tolerances are linear-solver gates and cannot supply its steady
threshold. The legacy raw `5e-5` threshold converts to a project-owned map
defect of `0.05`, not a literature tolerance.

A startup-state 1x/2x/4x/16x/64x ladder keeps the normalized map defect within
0.192%, with every linear and conservation gate green. From a warm checkpoint,
eight 64x updates decrease strictly from `0.3530` to `0.1637`, replay bitwise
through a 4+4 restart, and keep CFL below `3.5e-5`. The magnetic pseudo-time cap
is therefore raised from `0.001/N` to `0.064/N`; the existing reduced B2
physics/restart test falls from the prior 45-second gate record to 14.8 seconds.
This startup evidence was later superseded by the warm-state audit below.

Schema 4 records the post-map nonlinear physical momentum residual
`L max|C-D-E-JxB-f+Gp|/(rho U0^2 N)`, or `max|R|/540` for B2. It shares the
projection pressure-face stencil and diffusion boundary assembly, uses the
prescribed electromagnetic scale, and evaluates fresh mapped Lorentz force
before coupling acceleration. The reduced history decreases from `0.9760` to
`0.3103` in four updates and reaches `0.03870` at step 90. JIT/JVP, schema
compatibility, and exact direct-four versus serialized-two-plus-two replay pass.

The operator audit proves this is not the exact split-map fixed-point defect.
Momentum acts on the predictor with old-state weights, Lorentz force, and
relaxed flux; projection interpolates predictor cells to faces and back; the
diagnostic re-evaluates nonlinear operators on the raw mapped state and includes
only the pressure-face correction. At reduced step 4, the pressure-force
contribution is `0.0664` and the omitted face-reconstruction contribution is
`0.0880`; the projection identity closes exactly. The omitted term already
exceeds the observed `~0.04` tail, so a coarse split/discretization floor is
plausible. Retain the historical field name
`iteration_momentum_defect_history` for schema compatibility, but call it the
post-map nonlinear momentum residual in user-facing material. It is a validation
diagnostic, not a stopping condition, and receives no `0.01` threshold. The
failed 120-second outcome study remains runtime/floor evidence only.

The larger-cap same-state probe keeps all solves green, balance below `4e-10`,
and CFL below `1.4e-4`, but 128x/256x change normalized map rate by 6.78%.
Reject 128x and 256x. The predictor momentum
identity suggests a normalized unrelaxed velocity map rate, up to the gated
linear residual, but the pressure-corrected cell map also contains projection
reconstruction. Raising pseudo-time by 64x while retaining raw
`coupling_tolerance=5e-5` accidentally tightened this normalized gate from
`0.05` to `7.8125e-4`. Stopping must not depend on the pseudo-time cap.

The manufactured predictor identity and mixed-boundary reconstruction gate
now pass. A warm same-state `dt/dt2/dt4` decomposition closes the split identity
below `4e-10` with every linear solve green, but its normalized map rates span
8.30%, rejecting the predeclared 0.5% transient-invariance limit. The growing
face-reconstruction contribution as `dt` falls confirms that this decomposition
is an operator audit, not a tolerance calibration. A follow-up lower-cap ladder
at safety `0.032` through `0.001` finds no passing three-level rung. Below
`0.016`, the raw cell-velocity update approaches `3.69e-3` instead of scaling
with `dt`, while `u` remains controlling, every linear solve passes, and balance
stays below `3.6e-10`. Dividing that reconstruction contribution by `N dt`
therefore makes the candidate cell-map rate diverge; lowering the cap cannot
repair the observable.

Commit `3e731fa` preserves predictor cells and reconstructs only the pressure
correction, followed by the existing uniform fixed-flow adjustment. The
conservative corrected face flux is unchanged. A regression proves that zero
pressure correction preserves an arbitrary divergence-free predictor; mixed
flow, JIT/autodiff, conservation, and exact 4-versus-2+2 restart gates pass.
The corrected warm ladder gives map rates `0.265699`, `0.265773`, and `0.265903`
at 64x/32x/16x: 0.0768% spread versus the frozen 0.5% limit, raw updates halve,
`u` remains controlling, and balance stays below `3.4e-10`. Refreeze 64x. The
current two-update native FreeMHD smoke also passes every schema-3 execution and
comparison gate at `0ab33b2`, with pressure RMS/Linf `0.004518/0.010917`.

Schema 5 now versions direct normalized velocity-map stopping with three
sustained passes, leaving pressure/potential updates as diagnostics and retaining
momentum, pressure, electric, and conservation gates. The bounded outcome study
rejects `tau_map=0.05`: it converges at step 30, but versus the more-converged
step-96 state its pressure Linf, velocity Linf, and pressure-gradient relative
L2 differences are `1.676e-3`, `0.1714`, and `0.561%`, exceeding all frozen
limits. The `0.005` branch remains monotone but reaches only `0.01502` at its
96-update ceiling, so it is not calibrated. These are not two converged
endpoints. A positive serialized schema-5 path reaches its third sustained pass
on update five, closing convergence/restart semantics without promoting a
physical threshold.

Fixed-relaxation probes through 8 remain monotone, but improve only 9.33% over
the relaxation-4 control, below the predeclared 15% promotion gate. Retain 2.
Released SOLVAX 0.8.3 remains the supported floor. Before LMX uses coupled
Anderson acceleration, reconcile and release SOLVAX 0.8.4's weight API, then
apply identical weights and damping to fields and conservative compact flux.
Do not store a production-scale full-state history without a measured benefit.

The next bounded tranche is upstream-first. SOLVAX draft PR 21 now carries the
clean 0.8.4 weight API and corrected release metadata; its supported
Python/JAX, lint, and docs matrix is green. It is mergeable with no reviews,
comments, or technical blocker; review, ready-for-review, merge, rebuild from
the merged SHA, tag, and publication remain procedural and require explicit
authorization. Then add one sharding-aware B2 path shared by CPU and GPU with
exact depth two, one prior raw mapped record, one weight calculation shared by
scaled mapped fields and compact plus/inlet flux, and restart schema 6. B2 has
no Anderson lists today; the existing arbitrary-depth lists are B1-only and
must not be repurposed. Keep
schema 6 separate from the Aitken-only diagnostic loader; require all restart
arrays together and prove the distributed residual Gram reduction on every
target device topology. This retains about
35.8 MiB of prior state on the coarse grid instead of about 416.7 MiB for a
depth-16 iterate/residual history. Prove direct versus serialized 1+1 replay and
sharding before testing the shared step-29 and strict step-96 checkpoints.
Promote only if both paths cross `tau_map=0.005`, preserve every linear,
conservation, restart, and sharding gate, and agree within the frozen pressure,
velocity, and pressure-gradient outcome limits. Otherwise retain fixed
relaxation and stop before a coarse run.

The production spec's `5e-5` is now an explicitly provisional, fail-closed
normalized-map bound, not an accepted threshold. Do not authorize a corrected
coarse run until acceleration reaches a tighter reference and the frozen QoI,
linear, conservation, restart, and sharding gates pass.
Tolerance, wall, medium, fine, and production-FreeMHD work remain blocked until
that coarse baseline converges for the correct reason. Evidence is in
`benchmarks/results/b2-pseudotime-map-rate-20260715.json` and
`benchmarks/results/b2-momentum-defect-20260715.json`.

The primary ALEX B2 pressure observable is same-station transverse pressure,
side wall minus top wall; the current operator family and 4.39 cm half-width
match the accessible primary sources. The earlier 15.2 cm axial-pressure and
4.8 cm claims are unsupported and must not enter acceptance. Reed's explicit
finite spacing is instead the side-wall voltage separation `1.73a = 7.59 cm`.
A general pressure-hole correction `delta p=(t_w/d)(phi2-phi1)` exists, but the
available ALEX reports do not supply the hole diameter, potential samples,
sign, raw-versus-corrected marker status, or uncertainty budget. Do not invent
these inputs: keep pressure-hole transfer as a blocked sensitivity until the
raw table/drawing and reduction procedure are recovered. The current `0.004`
band remains repository digitization/marker-scatter allowance, not experimental
uncertainty. The tiny FreeMHD matched probes remain smoke observers, not
experimental-acceptance operators.

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

Keep `solvax>=0.8.3,<1` until the new API is released. PyPI and the latest
GitHub tag are 0.8.3, and CI tests both that minimum and the newest compatible
release. SOLVAX draft PR 21 proposes `release/0.8.4` at `a8603dc` into
`origin/main` at `255d280`; the reusable Anderson-weight implementation is
commit `4808695`, followed only by corrected citation/changelog metadata. Local
gates report 264 passes, 98.10% combined coverage, warning-free docs, lint, and
artifact import checks; the hosted minimum/current Linux and current macOS
matrix is also green. Do not tag or publish until review closes; rebuild
artifacts from the merged SHA.

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
Delete LMX's private B1 `_anderson_extruded_state` wrapper in the same tranche
that adopts the published SOLVAX API.

Released 0.8.3 already owns the valuable generic linear solves and
preconditioners. One manual additive composition remains, but migrating its few
lines could perturb parity trajectories and waits until B2 convergence closes.
Momentum's single diagonal division remains clearer and smaller than wrapping
`solvax.jacobi`; ownership movement without code deletion is not a performance
win. After 0.8.4 is published, use one
`anderson_weights` result for scaled fields and compact-flux histories, and use
`linear_solve(has_aux=True)` to retain momentum diagnostics without a final
extra matvec. Raise the minimum dependency in that same correctness tranche.
The next upstream slimming candidate is a released, CPU/GPU/gradient-gated
complex tridiagonal solve, replacing paired real/imaginary calls in LMX.

The released `block_thomas_factor_fn` B1 prototype is exact against the current
materialized retained-modal factors and through JVP. At `7 x 9 x 16` and
`11 x 17 x 32`, however, it is 27--29% slower cold and 38--39% slower warm;
device peak falls by at most 1.2% while host peak rises about 4%. Reject it
before a production run. Reconsider generated factors or `chunked_jacfwd` only
if a measured production B1 memory blocker outweighs that setup regression.
Do not move finite-volume assembly into SOLVAX, and do not substitute
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
useful visuals, at least 95% combined line/branch coverage, below the ten-minute portable-test
limit, and research-grade for every feature it labels accepted.
