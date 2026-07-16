# LMX authoritative development plan

Status: 2026-07-16. The current two-update B2/FreeMHD smoke and schema-5
stopping contract are keyed to `0ab33b2`; current one-/two-/four-CPU-device
equivalence is keyed to `4c94389`.
The installable-distribution contract is keyed to `a207bf9`: frozen benchmark
resources are package-owned, the wheel smoke runs outside the checkout,
wheel/source membership and size are fail-closed, and the numerical core no
longer installs or eagerly imports Matplotlib and Pillow.
The post-map nonlinear momentum residual and
restart schema 4 were keyed to `e6834ee`; schema 5 now versions normalized
stopping. Its fixed-relaxation memory reduction
is keyed to `791e496`, and its exact operator contract is keyed to `2d0fb50`.
Commit `3e731fa` removes the projection reconstruction floor and refreezes the
64x pseudo-time cap on a warm same-state ladder.
The latest complete portable gate exercised source `664da3c`. CPU/GPU calibration at
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
| B2 ALEX square duct | conservative momentum, mixed axial boundaries, explicit stress, compact corrected flux, post-map physical residual, strict schema-5 replay, refrozen 64x cap, and current CPU/GPU device equivalence have bounded gates | tighter-reference stopping calibration, production parity, and steady-production scaling remain open |
| Matched B2 harness | deterministic inputs, pinned sources, independent observers, and native two-update LMX/FreeMHD execution pass every frozen schema-3 smoke gate | production acceptance and mesh convergence remain open |
| Differentiation | selected objectives pass finite-difference or independent-transpose checks | no blanket end-to-end claim for every workflow |
| README/docs | concise feature-led README, sourced comparison table, feature-specific visuals, seven-second Hunt/blanket/Q2D loops, and Li/AlN convergence | refresh B2/scaling panels only from accepted canonical records |
| SOLVAX | released 0.8.3 owns the generic algebra consumed by LMX | no further solver migration is required for the B2 smoke |
| Distribution | lean installed wheel loads all frozen A/B references and runs a tiny solve without Matplotlib/Pillow; plotting is an explicit extra; source artifact excludes repository tests | bump 1.1.3 before publication; hosted release gate must be green |

Current structure reflects completed ownership moves: frozen resources are
package-owned, evidence freezing lives with its campaign analyzers, diagnostics
have one NPZ/restart owner, stale private wrappers are gone, and repeated CLI,
example, validation, reporting, convergence, and minimal-TOML test setup is
shared without removing cases or assertions.

| Surface | Current | Active ratchet | CI hard ceiling |
|---|---:|---:|---:|
| package modules | 35 | no new module | 35 |
| package lines | 34,683 | stay below 35,000 while preserving the physical residual | 35,100 |
| maintained-core lines | 7,854 | stay below 8,000 | 8,000 |
| test files / lines | 30 / 20,618 | no new file; stay below 21,000 | 31 / 21,100 |
| maintenance scripts | 13 | no new script without retiring an owner | 13 |
| tracked files | 173 | delete only superseded or duplicate ownership | 180 |
| tracked checkout | 3,323,782 bytes | do not increase without a user-facing need | 4,194,304 bytes |

These ratchets must come from ownership deletion, shared helpers, or removal of
superseded behavior—not unreadable formatting or arbitrary test merging.
The latest ownership audit removed one unreferenced superseded GPU record.
The remaining result records, 13 scripts, and distinct Python/TOML examples
retain current provenance, CI, or runnable-workflow ownership; do not merge
them unless the replacement deletes code without weakening those contracts.
Three exact artifact/array consolidations remove 60 package lines and four
private symbols: blanket numeric CSV persistence, mirrored multilayer wall
faces, and ordered 2-D/3-D tabulated-field interpolation.
Commit `a735cd7` removes another 66 lines by deleting three undocumented,
unexported persistence/sampling wrappers and the duplicate direct-CLI guard;
their generic owners remain. Commit `664da3c` adds 22 compact public-contract
cases to the existing mesh and FreeMHD modules, with no new test file.

The portable-gate artifact keyed to `664da3c` records 838 passes, 8 expected
external-data skips, 95.5539% combined line/branch coverage, and 148.4 seconds on
the reference Apple M4. It is 0.3% faster than the prior 148.9-second record
despite the added validation cases. The gate stays below half of the 300-second
engineering target and one quarter of the 600-second hard limit. The 95.5%
engineering target is now met; retain the 95% enforced floor until a second
Python endpoint or hosted run independently confirms the margin. The
six-worker record reports 50.8 seconds for reduced B2 and 47.2 seconds for
weighted modal; these concurrent durations still identify contention rather
than isolated regressions, so no scheduling change is promoted from this run.

The clean distribution audit at `a207bf9` produces a 313,854-byte wheel with
48 members and a 296,630-byte source archive with 54 members. Both pass Twine;
the wheel is limited to package source, seven frozen data files, and metadata,
while the source archive adds only build metadata, README, license, and
manifest. A Python 3.12 clean install resolves JAX 0.10.2 and SOLVAX 0.8.3,
loads every packaged Benchmark A/B and Samper reference outside the checkout,
and reaches a `9.95e-9` residual on a tiny Hartmann solve. Matplotlib and Pillow
are absent from the core environment, and importing the CLI, plotting facade,
Q2D, blanket, field, showcase, and solver modules loads neither package. The
`visualization` extra retains all tested plots and movies. The prior local
Python 3.10 endpoint resolves JAX 0.6.2 and the same SOLVAX release. Hosted
endpoint verification remains unavailable because Actions jobs execute zero
steps.

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

Two workstreams followed the smoke: maintain the portable critical path and
establish canonical sharding. Their current decisions are below; timing
measurements run alone.

### CI critical path

The modal pipe test reuses one physical projection and verifies direct
mode-factor algebra without a second integration run. In the latest six-worker
gate it reports 47.2 seconds, versus 50.8 seconds for reduced B2 and 25.2
seconds for reduced B1. Isolated measurement attributes most of that tail to
worker contention: reducing only the manufactured modal grid lowered its
weighted path to 23.5--26.1 seconds, and the unchanged reduced-B2 restart and
physics node now measures 14.8 seconds alone. Preserve those coverage-rich
tests. A fresh-process A/B of the same six expensive JAX nodes takes 37.69
seconds with six workers and 36.41 with four. The 3.4% change misses the frozen
10% promotion threshold, so retain six-worker work stealing; module grouping
would imbalance the fringing and autodiff owners. Profile another node only if
an isolated measurement crosses the 45-second trigger.

A fresh-process example probe required a 25% isolated speedup without changing
workflow assertions. The variable-field cut from `16 x 16 x 9` to
`10 x 10 x 5` reduced 6.17 to 5.81 seconds (5.8%), so it is rejected. The
operator baseline `(12, 24, 48)` took 3.96 seconds; `(8, 16, 32)` and
`(10, 20, 40)` took 3.69 and 3.89 seconds but failed the unchanged observed-order
gate (`gradient_z > 1.8`). Retain both original grids; their longer concurrent
JUnit times are contention, not an isolated size bottleneck.

Centralizing figure persistence permits one real PNG/PDF signature test and
lightweight persistence stubs for the remaining plotting writers. Replacing
two redundant bent-pipe solves with synthetic writer fields then reduces that
node from 6.39 to a 1.24-second median and the isolated plotting module from
9.84 to 5.43 seconds. Bent-pipe physics remains owned by its fringing tests.
That plotting-focused gate reached 147.3 seconds; the current validation-margin
gate is 148.4 seconds, a 0.7% concurrent variation that does not justify a
scheduling change.

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
`--xla_gpu_exclude_nondeterministic_ops`; default GPU mode is calibrated below
and misses its promotion threshold. Compact
records are `benchmarks/results/b2-{cpu,gpu}-device-equivalence-20260715.json`.

An explicit one-device request uses the same named-sharding kernels as the
multi-device path. On the pre-terminal-fix source, the `128 x 31 x 31` rung passes
validation, placement, exact restart, and device-equivalence gates. Warm
medians are 0.857, 0.652, and 0.633 seconds on 1/2/4 devices: 1.31x and 1.35x
speedups, with modest gain beyond two devices. This is a two-update scaling calibration, not a steady
production-speed claim. The compact record is
`benchmarks/results/b2-cpu-strong-scaling-20260715.json`.

The current GPU contract and decision are compact:

| Evidence | Result | Decision |
|---|---|---|
| `8 x 7 x 7`, 1/2 RTX A4000 | repeat, exact restart, conservation, placement, and equivalence pass within `1.02e-14` | production sharding is correct; the fixed-relaxation scalar stays compile-time |
| `128 x 67 x 67`, 1/2 RTX A4000 | 2.780/2.400 s, CV below 1.2%, 1.159x end-to-end and 1.510x core-phase speedup | misses the 1.2x promotion gate; retain the smaller validation fusion and stop |
| historical `256 x 67 x 67` | 8.474/7.534 s, CV below 3.7%, 1.125x | diagnostic only; no larger rung |

The current calibration fails closed on physical repeat signatures, linear
status/history, placement, restart, conservation, and device equivalence.
Nonflux restart state must agree within `1e-12`; corrected flux uses `1e-6`
absolute and `1e-5` relative tolerances, while the tiny harness remains exact.
The compiler trace attributes the remaining cost to transverse PCR work and
post-map host transfers, not collectives; communication tuning therefore stops.

Rejected variants remain rejected: weaker pressure tolerance, removing the
axial-mean correction or line blocks, Jacobi substitutions, mixed-boundary
coarse correction, additive correction, and launch-only SOLVAX batching. Each
either violated the frozen numerical contract or missed its timing gate. Do
not revive preconditioner microprobes or larger rungs without a new trace and a
plan revision. Full data and rejection provenance are in
`benchmarks/results/b2-gpu-scaling-calibration-20260715.json` and the performance
documentation. Schema-6 Anderson is the next bounded workstream after SOLVAX
publication.

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
- schema 5 requires three sustained normalized-map passes and preserves every
  linear, conservation, diagnostic, and exact-restart field;
- the historical `iteration_momentum_defect_history` field is a post-map
  nonlinear momentum residual, not the split-map defect or a stopping gate.

The bounded outcome study rejects `tau_map=0.05`: pressure, velocity, and
pressure-gradient differences exceed every frozen QoI limit. The `0.005` path
reaches only `0.01502` at step 96, so no threshold is calibrated. Relaxation
factors through eight miss the 15% promotion gate; retain fixed relaxation 2.
Pre-fix restarts, 128x/256x pseudo-time caps, and extrapolated convergence steps
are diagnostic only. Exact histories and operator audits are in
`benchmarks/results/b2-pseudotime-map-rate-20260715.json` and
`benchmarks/results/b2-momentum-defect-20260715.json`.

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
matrix plus Codecov project/patch checks are all green. It remains a draft
without requested review or approval. The local SOLVAX checkout contains
unrelated user changes and is not a release-artifact source. Do not tag or
publish until review closes; rebuild
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

The README now links three dynamics loops at exactly 7.00 seconds: accepted
Hunt startup plus research-stage blanket and Q2D flows. The 86,808-byte blanket
derivative retimes its existing 5.75-second H.264 with motion interpolation;
physical frames are unchanged and no solver rerun or tracked media was added.

The next documentation tranche is solver-free: derive one sub-100-KiB,
release-hosted WebP from the already accepted bent-pipe overview and Dean
comparison stills, then add a short “Curved pipes” feature block. Label it a
low-De inductionless baseline with Dean-vortex physics staged; do not imply
production validation and do not rerun the solver. Do not create a B2 movie
from the bounded smoke or rejected coarse trajectories.

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
