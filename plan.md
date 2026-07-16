# LMX authoritative development plan

Status: 2026-07-16. LMX consumes released SOLVAX 0.8.4;
`aaa41b1` made the
frozen B2 path Anderson depth two with one shared weight vector for mapped
scaled fields and conservative compact flux. Restart schema 6 stores one raw
mapped field, residual, plus flux, and inlet flux all-or-none; direct two-update
and serialized NPZ one-plus-one replay are exact. Its six-update cold outcome
gate is now rejected, so schema 6 is correctness evidence rather than a
promoted B2 acceleration choice. A separately predeclared `max|weight| <= 4`
newest-map fallback is also rejected: it is stable and exact but ends 0.22%
worse than fixed relaxation two, so no SOLVAX or LMX API was added. Schemas
1--5 remain readable. A six-update spectrum audit then closes generic
shared-Euclidean Anderson tuning: 98.254--99.887% of its objective energy is
electric potential, while B2 acceptance uses velocity-map convergence.
A current-source velocity-block minimax audit then closes the bounded
depth-two and depth-three affine families: depth-two updates three and four can
improve by at most 0.0377% and 0.213%, while depth-three update four reaches
only 0.555%, so neither family can meet the frozen 15% gate. Fixed relaxation
two remains the control and no accelerator API is added.
The installable-distribution contract is refreshed at `ca7535e`: frozen benchmark
resources are package-owned, the wheel smoke runs outside the checkout,
wheel/source membership and size are fail-closed, and the numerical core no
longer installs or eagerly imports Matplotlib and Pillow.
The post-map nonlinear momentum residual and
restart schema 4 were keyed to `e6834ee`; schema 5 now versions normalized
stopping. Its fixed-relaxation memory reduction
is keyed to `791e496`, and its exact operator contract is keyed to `2d0fb50`.
Commit `3e731fa` removes the projection reconstruction floor and refreezes the
64x pseudo-time cap on a warm same-state ladder.
The latest complete portable gate exercised source `acdefd9`.
CPU/GPU calibration at
`413185a` remains historical; deterministic GPU equivalence at `3a22078` has
been replaced by the current `8b6f97d` result and the refreshed calibration
record committed at `3311d6d`. The isolated compiler trace keyed to `f379f6b`
is historical and cannot attribute the current accepted path.
The first fresh canonical-mesh coarse trajectory using the current formulation
passes conservation and all linear-solver gates but reaches its 128-update
bound before steady convergence. Its single authorized continuation preserves
those gates but misses its precommitted residual target; it is not promoted.
The smoke closes bounded orchestration and comparison, not production B2
acceptance. Current schema-6 `8 x 7 x 7` one-/two-/four-forced-CPU-device
topology passes exact replay, conservation, Gram/weight equivalence, and
placement. The isolated `256 x 67 x 67` calibration also passes correctness and
timing-stability gates. Its 1.229x/1.360x point speedups promote the two-device
optimization but miss the four-device bound. It is a two-update forced-device
calibration, not a physical-core or production strong-scaling claim. Schema-6
topology, explicit component placement, and exact serialized replay pass on
one and two GPUs; shared-host timing is excluded. The latest multi-minute,
static-preflight Docker CPU-allocation path, keyed to `a92b4e6`, passes
32-update sustained scaling:
every warm trajectory lasts 147--269 s, speedup is 1.317x/1.684x on two/four
versus one JAX device, and all confidence, efficiency, restart, physics,
provenance, memory, and 60-second admission gates pass. A 96-update
one/two-A4000 lane also passes 158--259 s duration, numerical, restart, and
topology gates with 1.626x shared-host speedup, but persistent foreign contexts
keep authoritative GPU scaling open. Peak device memory is 2.50 GB on one GPU
and 1.41/1.31 GB on two. The fixed-work harness accepts
an explicit update count, checkpoints at the deterministic midpoint, replays
the remaining trajectory, and requires one cold plus at least three warm
trajectories, with every warm sample lasting at least 120 s, for a sustained
claim; the two-update default remains the bounded CI/debug gate. One
`--sustained` preset selects `256 x 67 x 67`, 32 CPU or 96 GPU updates, four
repeats, and an 1800-second ceiling. The summary re-derives duration, the
explicit cold/compile timing contract, median, and warm CV (at most 5%); it
never trusts a candidate boolean. It fails closed unless one exact 1/2/4-shard CPU or
1/2-GPU group shares provenance and fixed work and passes sustained duration,
real-solver numerics, placement, peak memory, explicit affinity/idle-host
preflight, and checksummed continuous/postflight monitoring. Incomplete
candidates remain visible without a sustained speedup or plot title. The
earlier multi-minute CPU record passed its static preflight but
is not promotable under the new continuous-monitor gate. Its confirmation was
aborted without a timing record after unrelated local simulations raised host
load above 135 following a clean preflight. A later check still found load 83,
active SFINCS/Python work, and severe swapouts. Require continuous and
postflight host admission before retrying. Retry the 96-update GPU campaign
only after both office A4000s pass their clean-environment gates; idle-looking
instantaneous utilization is insufficient while foreign contexts remain.
This single active plan records accepted baselines, active gates, and stop/go
criteria—not campaign history. Completed campaign details belong in
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
Short scaling runs are smoke or calibration only. A CPU or GPU strong-scaling
claim must predeclare a fixed workload, run one cold plus at least three warm
trajectories per rung, and keep every warm trajectory at or above two minutes;
compilation, restart I/O, and observers remain untimed. Admission must remain
clean through a continuous monitor and postflight; a clean preflight alone is
not sufficient evidence.

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

The `unit` marker denotes isolated scope, not runtime: expensive fringing and
autodiff nodes inherit it at module level. Use direct node IDs or narrow `-k`
expressions for fast development checks; do not advertise `pytest -m unit` as
a quick lane.

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
| B2 ALEX square duct | conservative momentum, mixed axial boundaries, explicit stress, compact corrected flux, post-map physical residual, strict schema-6 Anderson replay, refrozen 64x cap, and current CPU/GPU topology have bounded gates | tighter-reference stopping calibration, production parity, and steady-production scaling remain open |
| Matched B2 harness | deterministic inputs, pinned sources, independent observers, and native two-update LMX/FreeMHD execution pass every frozen schema-3 smoke gate | production acceptance and mesh convergence remain open |
| Differentiation | selected objectives pass finite-difference or independent-transpose checks | no blanket end-to-end claim for every workflow |
| README/docs | concise feature-led README, sourced comparison table, feature-specific visuals, seven-second Hunt/blanket/Q2D loops, and Li/AlN convergence | refresh B2/scaling panels only from accepted canonical records |
| SOLVAX | released 0.8.4 owns generic algebra, point Jacobi, Anderson weights, and accepted `linear_solve(has_aux=True)` diagnostics consumed by LMX | no worthwhile ownership deletion remains; re-audit new compatible releases |
| Distribution | lean installed wheel loads all frozen A/B references and runs a tiny solve without Matplotlib/Pillow; plotting is an explicit extra; source artifact excludes repository tests | bump 1.1.3 before publication; hosted release gate must be green |

Current structure reflects completed ownership moves: frozen resources are
package-owned, evidence freezing lives with its campaign analyzers, diagnostics
have one NPZ/restart owner, stale private wrappers are gone, and repeated CLI,
example, validation, reporting, convergence, and minimal-TOML test setup is
shared without removing cases or assertions.

| Surface | Current | Active ratchet | CI hard ceiling |
|---|---:|---:|---:|
| package modules | 35 | no new module | 35 |
| package lines | 34,692 | stay below 34,700 | 35,100 |
| maintained-core lines | 7,880 | stay below 7,890 | 8,000 |
| test files / lines | 30 / 21,074 | no new file; stay below 21,075 | 31 / 21,100 |
| maintenance scripts | 13 | no new script without retiring an owner | 13 |
| tracked files | 186 | no new file without retiring another owner | 187 |
| tracked checkout | 4,549,272 bytes | stay below 4,550,000 without hiding direct visuals | 4,718,592 bytes |

These ratchets must come from ownership deletion, shared helpers, or removal of
superseded behavior—not unreadable formatting or arbitrary test merging.
The latest ownership audit removed one unreferenced superseded GPU record.
The remaining result records, 13 scripts, and distinct Python/TOML examples
retain current provenance, CI, or runnable-workflow ownership; do not merge
them unless the replacement deletes code without weakening those contracts.

Completed slimming, distribution, SOLVAX, scaling-admission, and portable-gate
history is preserved in Git, checksummed benchmark records, and the
validation/performance documentation; only current ratchets and open gates
remain authoritative here.

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

The record keyed to LMX `0ab33b2` and the pinned FreeMHD source runs two fixed
Euler updates in 4.443 seconds on one JAX device and 7.467 seconds on two native
FreeMHD MPI ranks. Restart, mass/current closure, interface-current activity,
and Courant gates pass. Cross-code normalized pressure differences are 0.00452
RMS and 0.01092 maximum, within the frozen 0.16 and 0.32 smoke limits.

Authoritative evidence is
`benchmarks/results/b2-freemhd-harness-smoke-20260715.json`; its 1.64 MB raw
bundle remains outside Git. Exit met for orchestration and two-update numerical
consistency only. Production parity, the canonical three-mesh ladder,
experimental acceptance, and steady scaling remain open.

## Priority 2: unblock fast iteration and real strong scaling

Two workstreams followed the smoke: maintain the portable critical path and
establish canonical sharding. Their current decisions are below; timing
measurements run alone.

### CI critical path

Prior optimizer, configuration, and plotting duplication has been removed
without dropping physics ownership; rejected grid cuts and worker-count changes
remain in checksummed CI evidence. Keep six-worker work stealing unless a fresh
isolated A/B clears 10%. Record the top ten node durations and review any node
above 45 seconds. Preserve the 300-second engineering target,
600-second hard limit, and at least 95% combined line/branch coverage. Prefer
parameterization and shared fixtures inside existing test files; do not create
another test file merely to move lines.

### Canonical sharding and performance

The schema-6 `8 x 7 x 7` 1/2/4-device gate proves exact topology, restart,
placement, conservation, and Anderson equivalence only. Production fields,
compact flux, and accelerator state shard axially; inlet state is replicated.
The global restricted-grid pressure correction preserves conditioning across
shards. Seconds-scale forced-XLA timings stay calibration evidence, never a
physical-core claim or reason to add an 8-device rung.

Rejected small probes stay in checksummed evidence. Physical-core claims need
verifiable affinity; forced macOS devices prove topology only.

Seconds-scale affinity pilots established the nested 2/4/8-CPU allocation and
correctness contract but remain debug evidence. A 20-update duration pilot was
also too short, so the sustained protocol uses 32 updates. On identical
`256 x 67 x 67` input,
the current `a92b4e6` 1/2/4-device ladder uses 2/4/8 CPU allocations and reports
246.702/187.307/146.524-second warm medians. Speedups are 1.317x/1.684x,
empirical 95% lower bounds are 1.301x/1.661x, and efficiencies are 65.9%/42.1%.
Every warm sample lasts at least 146 s, CVs are below 4.45%, midpoint replay is
exact, and every placement, linear, conservation, Anderson, provenance, static
admission, and cross-topology gate passes. Retain this as multi-minute
fixed-work calibration; current promotion additionally requires a clean
continuous/postflight trace. It is not exact M4 host-core, steady-state, or B2
solution evidence. Peak process RSS is
4.65/5.22/5.43 GB and the solution-bundle estimate is 101.1 MB at every rung;
CPU allocator bytes per device are unavailable. Apply the same protocol to
GPUs only after the shared host passes the 60-second idle/no-foreign-work gate.
The sustained CPU launcher now records one-second host samples through the
worker plus a 15-second postflight and binds the ignored JSONL digest to the
source fingerprint; any probe, affinity, swapout, foreign-work, or gap violation
blocks promotion. Add the equivalent remote GPU supervisor before the next run.

The existing `101 x 77 x 77` coarse checkpoints were screened before launching
that ladder. Three representative files match the current geometry and solver
shape, but all normalize to `legacy_nonexact`, omit compact flux and schema-6
Anderson state, and carry no source fingerprint. They may support a separately
labelled warm-start diagnostic, but they cannot seed exact-restart, current-
source, or steady-production exact-host-core evidence. That restart gate remains
open independently of the matched two-update pilot.

Compact CPU evidence is
`benchmarks/results/b2-schema6-cpu-scaling-20260716.json`; raw worker JSON and
plots remain ignored. The worker fingerprint must include package-owned frozen
specifications before any cross-host comparison is accepted.

The current GPU contract and decision are compact:

| Evidence | Result | Decision |
|---|---|---|
| schema-6 `8 x 7 x 7`, 1/2 RTX A4000 | current placement, exact flux replay, state replay to `2.22e-16`, conservation, linear, repeat, Gram, and Anderson gates pass | topology correctness accepted; discard shared-host timing and make no scaling claim |
| current `256 x 67 x 67`, 96 updates, 1/2 RTX A4000 | 258.913/159.234 s, 1.626x, 81.3% efficiency, every warm sample above 120 s; numerical/topology gates pass | accepted non-idle shared-host sustained calibration; four foreign contexts block authoritative timing |

Every current calibration fails closed on physical repeat signatures, linear
status/history, placement, restart, conservation, and device equivalence.
Replay-driving state must satisfy the frozen elementwise mixed ratio
`|delta| / (2e-9 + 2e-8 |state|) <= 1`; corrected flux uses `1e-6` absolute and
`1e-5` relative tolerances, while the tiny harness remains exact.
Rejected GPU variants and microprobes stay in
`benchmarks/results/b2-gpu-scaling-calibration-20260715.json`; do not revive
one without a new trace, frozen hypothesis, and plan revision.

Separate compilation from repeated timings and report uncertainty, memory,
placement, speedup, and parallel efficiency. Independent-case multiprocessing
is throughput evidence, not strong scaling. Historical 1.66x two-GPU and CPU
surrogate results are diagnostic only and must not appear as canonical claims.

Production fields shard axially. Keep compact cell-shaped positive-face flux
plus one replicated inlet plane; exchange nonperiodic halos explicitly and do
not checkpoint duplicated `nx+1` arrays. Optimize only a profiled bottleneck on
the physics-valid path.

Exit: portable tests stay below ten minutes; CPU/GPU topology and replay remain
exact; a clean continuously monitored sustained CPU/GPU ladder is published.
Exact M4 core mapping and idle-host GPU timing remain open until then.

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
- compact schema-6 restart state containing velocity, pressure, face flux, one
  prior raw mapped scaled field/residual/plus-flux/inlet-flux record,
  convergence streak, CFL/stopping state, and all required histories.

Advance exact coarse, medium, then fine meshes one level at a time. Each level
must pass literature/ALEX pressure, FreeMHD observable, conservation, restart,
wall-thickness, tolerance, steady-state, and mesh-change gates for the correct
reason. A Maxwell-consistent fringe field is a separately labelled sensitivity
study, not a replacement acceptance case.

Refresh README/docs B2 and scaling visuals only from compact accepted records.
Every image states validation status and provenance; no superseded result is
silently relabelled.

The current coarse result is diagnostic, not accepted. The `102 x 77 x 77`
two-GPU run and its one authorized continuation keep every linear,
conservation, placement, and restart gate green through 256 updates, but finish
at `7.1081e-4`, 29.24% above the precommitted `5.5e-4` continuation gate. No
further fixed-relaxation continuation is authorized; the 45.7 MiB restart stays
outside Git. See `benchmarks/results/b2-current-coarse-baseline-20260715.json`.

The accepted stopping/restart contract is:

- terminal updates receive the same acceleration as interior updates, and
  direct versus serialized replay passes at `1e-12`;
- the normalized B2 velocity-map rate is `max|delta u|/(N dt)` with
  `L=U0=1`, `N=540`; it is project-owned, not a FreeMHD/NekRS tolerance;
- predictor-preserving projection refreezes the `0.064/N` pseudo-time cap: the
  warm 64x/32x/16x map rates span 0.0768% against the 0.5% gate;
- schema 6 requires three sustained normalized-map passes and preserves every
  linear, conservation, diagnostic, raw Anderson, and exact-restart field;
- schemas 1--5 remain readable, but an Aitken restart is not an exact Anderson
  continuation and cannot seed a promoted schema-6 trajectory;
- the historical `iteration_momentum_defect_history` field is a post-map
  nonlinear momentum residual, not the split-map defect or a stopping gate.

Frozen outcome gates retain fixed relaxation 2 and reject the tested pseudo-time
changes, relaxation factors, Anderson depth two, bounded newest-map fallback,
shared-norm tuning, and bounded depth-two/depth-three velocity families. The
failures preserve linear, conservation, first-update, and replay contracts but
cannot deliver the required 15% velocity-map gain; do not add or retune those
APIs without a new residual objective. Exact histories, minimax certificates,
and operator audits are in
`benchmarks/results/b2-pseudotime-map-rate-20260715.json` and
`benchmarks/results/b2-momentum-defect-20260715.json`.

SOLVAX 0.8.4 and LMX schema 6 are complete. The CPU topology proves one shared
distributed Gram/weight calculation and exact serialized replay while retaining
about 35.8 MiB of prior state on the coarse grid instead of about 416.7 MiB for
a depth-16 iterate/residual history. B1 retains its separate arbitrary-depth
history.

The validation sequence is fail-closed:

1. the `8 x 7 x 7` schema-6 topology gate is complete on one/two A4000s:
   replay, Gram/weights/observables, sharded field/flux/state, and the replicated
   inlet plane pass; shared-host timing is excluded;
2. the six-update `7 x 7 x 7` Anderson-depth-two comparison is complete and
   rejected: final map rate regresses by 361%, `max|weight|=24.39`, and the 15%
   improvement and stable-weight gates fail;
3. the one authorized bounded-weight fallback is also rejected: stability and
   replay pass, but its final rate is 0.22% worse than the control. Do not add
   that SOLVAX/LMX API or tune either failed configuration;
4. the residual-spectrum rationale gate is complete: zero of five pairs pass,
   and potential dominates the norm. Do not pursue another shared-Euclidean
   Anderson configuration;
5. the velocity-block minimax gate is complete: depth-two has two failing
   updates and depth-three update four cannot exceed 0.555%. Do not implement
   or tune either bounded affine family;
6. a materially different accelerator first requires a predeclared solver-free
   rationale and independent holdout, then must pass this same cold gate before
   regenerating both current-source trajectories to the shared step-29 point;
7. only if step 29 passes, continue to strict step 96. Promotion requires three
   sustained `tau_map=0.005` passes plus exact replay and the frozen pressure,
   velocity, pressure-gradient, linear, conservation, and sharding limits;
8. only then authorize a fresh canonical coarse schema-6 run. Medium/fine,
   production FreeMHD, authoritative idle-host GPU timing, and steady-production
   scaling remain blocked.

This failure stops before step 29 and the coarse run. Fixed relaxation two is
the control; the one bounded fallback is rejected and does not establish that
the provisional `5e-5` target is reachable. After the parallel
algorithm settles, consolidate
schema/scaling diagnostics inside existing owners; add no module, script, or
test file.

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

LMX requires `solvax>=0.8.4,<1`. SOLVAX PR 21 merged at
`0fcc017f45a5c55befa36afbfe21a0a21b1f4837`; tag, GitHub Release, trusted PyPI
publication, minimum/current Linux, current macOS, docs, lint, and clean import
gates are green. The exact release wheel SHA-256 is
`76614c4148f230138c9b992bc0178efccf6b1e8374f5ff62f17a65a6f076a10d` and
the sdist is
`5445b7fba6b3ad19886a80800d468020a09bc4b53159c2760dd949604bc954ac`.
Do not pin LMX to exactly 0.8.4; test the declared minimum and newest compatible
release.

The device-cut audit compared 0.8.3 with 0.8.4 PCG and found identical Krylov
code; generic two-device standard and single-reduction PCG both pass. The B2
failure was in LMX-owned layout transitions: eager
embedding, component extraction, slicing, packing, relaxation, and restart
staging across production shards. It was not a generic Krylov or SOLVAX defect.
The 0.8.4 dependency bump is solely for the reusable Anderson-weight owner.

LMX owns MHD equations, finite-volume stencils and limiters, geometry,
materials, interfaces, open-boundary and gauge semantics, corrected flux,
physical residuals, stopping/restart state, sharding policy, observables, and
acceptance. SOLVAX owns generic linear algebra after primal, residual,
transpose/gradient, JIT, placement, memory, and repeated timing gates pass.
Commit `f070445` already deleted LMX's private B1 Anderson wrapper; the B1 path
now calls released SOLVAX directly and keeps its separately owned arbitrary-depth
history.

Released 0.8.4 owns the valuable generic linear solves, preconditioners, and
Anderson weights. Its scale-aware regularization, condition filtering, damping,
and newest-map fallback were re-audited. Condition filtering cannot guarantee
the frozen coefficient bound, while a proposed generic `max_abs_weight=4`
fallback passed safety but failed LMX's acceleration outcome gate; therefore
neither repository gains an unused API. One manual additive composition remains, but migrating its few
lines could perturb parity trajectories and waits until B2 convergence closes.
Momentum's single diagonal division remains clearer and smaller than wrapping
`solvax.jacobi`; ownership movement without code deletion is not a performance
win. LMX now uses one `anderson_weights` result for scaled fields and
compact-flux histories. The bounded `linear_solve(has_aux=True)` momentum probe
is accepted at `a98590f`: it preserves the dense reference, JVP, gradient, and
B2 restart gates, is timing-neutral within noise, and deletes the final
diagnostic matvec. It does not alter schema 6.
The complex-tridiagonal candidate is now closed as an LMX no-go under its
precommitted performance gate. SOLVAX draft PR 22 at `b059e18` provides the
transparent complex API, current/minimum-JAX differentiation, and genuinely
complex Thomas fallback without a new module. Both Python/JAX endpoints pass
274 tests at 98.90% coverage, strict docs and lint pass, and complex64/128
correctness, JVP, gradient, and actual two-A4000 batch sharding pass. The first
packed-right-hand-side design was rejected before review: it was about 604x
slower on one GPU and 485x slower on two with no memory reduction. The safe
revision is bit-exact with the explicit pair and restores baseline performance,
but its measured gains are only 1.053x on one GPU and 1.070x on two, with
identical compiled memory. That misses the required 1.10x speedup or 10% memory
reduction. Retain LMX's two explicit pairs and the released dependency even
though the transparent call would remove four lines. PR 22 may proceed as a
reviewed SOLVAX capability; it does not self-authorize LMX adoption, merge, tag,
or publication.

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

Visual evidence is part of the acceptance contract. Every curated example and
every important physics, numerics, literature, differentiation, restart, or
parallel result with meaningful spatial, transient, convergence, comparison,
sensitivity, or scaling structure must map to a plot or movie in the README or
its feature page. Configuration/CLI mechanics and scalar-only invariants may
share the result they drive and do not require decorative charts. Every visual
labels maturity, generating record or command, observable, tolerance or
uncertainty, and test owner. Prefer an existing writer and accepted compact
record over a rerun; keep raw data and full-quality media outside Git.

Plots and movies are embedded directly in the README or owning docs page;
poster-only links do not satisfy coverage. Small animations use reproducible
Python/Pillow WebP compression, while source frames and full-quality MP4s stay
in the checksummed release. Shared derivatives may appear on multiple pages
without duplicating bytes.

The visual backlog is evidence-ranked. The stale scaling panel, bounded
operator-convergence gap, frozen Samper/Benchmark-A composite, exact restart,
bounded fringing, mapped-pipe FreeMHD-profile diagnostic, and strict Votyakov
obstacle mismatch are closed. Detailed Q2D external-diagnostic and blanket
current/pressure composites are also closed and embedded directly on their
owning pages. The Samper composite covers all eight high-Ha rows, mesh/order
gates, and current/power residuals without rerunning a solver. Do not relabel
the mapped-pipe transverse profiles as ALEX-B1 pressure evidence. Keep the
README to at most one new concise accepted-validation composite; detailed
evidence belongs on the owning pages.

Comparison-table cells describe the named native workflow, not what could be
implemented through arbitrary custom sources. Keep primary sources beside the
table. The current audit uses the FreeMHD paper/source, FreeMHD2 preprint, NekRS
26 documentation, NekRS MHD report, and NekRS GPU paper. Re-audit changing
capabilities before every release.

Aim for readable 6–8-second loops, stills below 100 KiB, tracked movies below
150 KiB where practical, and all tracked media below 1.25 MiB. The 1.25 MiB
cap admits the two evidence-rich P2 composites while preserving a hard,
test-owned budget. Host source frames, full-quality media, meshes, and raw
outputs stay in checksummed releases. Put provenance and acceptance status
beside every asset.

The README now displays three 7-second dynamics loops directly: an accepted
side-by-side Hunt/Shercliff startup comparison plus research-stage blanket and
Q2D flows. The duct comparison samples 35 paired physical frames at 5 fps from
current 200-step sources (`934c9aea...` Hunt and `3188bd21...` Shercliff); the
other loops use 42 physical frames. No solver-state interpolation or synthetic
motion is involved. Full-quality sources remain outside Git.

The solver-free curved-pipe documentation tranche is complete: a 94,236-byte
release-hosted WebP derives from the accepted bent-pipe overview and Dean
comparison stills, and the README labels it a low-De inductionless baseline
with Dean-vortex physics staged. No solver was rerun and no tracked media was
added. Do not create a B2 movie from the bounded smoke or rejected coarse
trajectories.

The documentation source is now self-contained. The Hunt/Shercliff, blanket,
and Q2D animations replace link-only posters and redundant tracked MP4s; all
20 tracked derivatives, including the detailed Q2D and blanket composites,
total 1,268,448 bytes, below the 1.25 MiB media cap. The closed-channel
derivative now combines the released Ha=20/100 analytical ladder with accepted
FreeMHD observables under its existing filename; no solver or new tracked file
was needed. Read the Docs still serves the
2026-05-01 build at `6be8622`. Build `33554636` failed with `commit: None` and
Git exit 128 because the builder had no credentials to clone the private GitHub
repository; the active repository webhook's latest response is HTTP 406. An
account owner must reconnect the Read the Docs GitHub authorization, then build
`be3ee82`. This is not a Sphinx or repository-configuration defect. Do not add
another movie until an accepted physical trajectory exists.

## Release gate

Distribution ownership, metadata, clean-install resource loading, lazy CLI and
numerical imports, and tiny-solve smoke are green at `a207bf9`. Version `1.1.3` matches the latest
tag and must be bumped consistently in `pyproject.toml`, `CITATION.cff`, and
`docs/conf.py` before publishing a new release; do not bump merely for local
development builds.

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
