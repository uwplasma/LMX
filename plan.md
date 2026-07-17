# LMX authoritative development plan

Status: 2026-07-17. LMX consumes released SOLVAX 0.8.4. Restart schema 6
provides compact, exact one-plus-one replay and correct 1/2/4-CPU plus 1/2-GPU
placement. Its tested Anderson and bounded affine variants miss the frozen 15%
velocity-map gain, so fixed relaxation 2 remains the B2 control. The current
coarse B2 trajectory passes linear, conservation, restart, and placement gates
but misses steady acceptance; no continuation is authorized.

The current monitored `256 x 67 x 67`, 32-update CPU ladder passes every frozen
Docker-allocation gate. Source/evaluator `9e49a9b` gives
244.181/172.682/146.768-second medians, 1.414x/1.664x two/four-device speedups,
4.709/5.201/5.486 GB peak RSS, and 0.653% maximum CV. Large-work, numerics, restart,
placement, and admission/runtime/postflight traces pass. Exact M4 P/E-core
mapping remains open, and step-limit trajectories are not steady-state
evidence. The 1.626x two-A4000 calibration remains non-authoritative because
foreign contexts block GPU admission.

The `--sustained` preset keeps 32 CPU or 96 GPU updates, one cold plus three
warm trajectories, a 120-second warm minimum, an 1800-second ceiling, and
checksummed per-rung admission plus continuous/postflight monitoring. Every
rung needs a fresh source/host/device-bound 60-second admission. Promotion also
requires at least one million global cells and 32 million cell-updates, exact
numerics/placement, memory evidence, warm CV at most 5%, and a clean
runtime/postflight trace. Short or undersized runs remain debug evidence.

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
| SOLVAX | published 0.8.4 owns generic algebra and Anderson weights; main `d0623a2` prepares 0.8.5 with the accepted additive line builder | consume and delete LMX's duplicate only after 0.8.5 is tagged and published |
| Distribution | lean installed wheel loads all frozen A/B references and runs a tiny solve without Matplotlib/Pillow; plotting is an explicit extra; source artifact excludes repository tests | bump 1.1.3 before publication; hosted release gate must be green |

Current structure reflects completed ownership moves: frozen resources are
package-owned, evidence freezing lives with its campaign analyzers, diagnostics
have one NPZ/restart owner, stale private wrappers are gone, and repeated CLI,
example, validation, reporting, convergence, and minimal-TOML test setup is
shared without removing cases or assertions.

| Surface | Current | Active ratchet | CI hard ceiling |
|---|---:|---:|---:|
| package modules | 35 | no new module | 35 |
| package lines | 34,425 | stay below 34,426 | 35,100 |
| maintained-core lines | 7,869 | stay below 7,870 | 8,000 |
| test files / lines | 30 / 20,988 | no new file; next test change must reduce lines | 31 / 21,100 |
| maintenance scripts | 13 | no new script without retiring an owner | 13 |
| tracked files | 186 | no new file without retiring another owner | 187 |
| tracked checkout | 4,550,352 bytes | stay below 4,550,353 without hiding direct visuals | 4,718,592 bytes |

These ratchets must come from ownership deletion, shared helpers, or removal of
superseded behavior—not unreadable formatting or arbitrary test merging.
The latest audits unified route-axis and centerline validation ownership, made
variable-field validation fail closed on nonfinite velocity, and folded two
duplicate solve tests into stronger physics owners. The direct B1 gate remains,
and B2 reuses complete checkpoints.
The remaining result records, 13 scripts, and distinct Python/TOML examples
retain current provenance, CI, or runnable-workflow ownership; do not merge
them unless the replacement deletes code without weakening those contracts.

Completed slimming, distribution, SOLVAX, scaling-admission, and portable-gate
history is preserved in Git, checksummed benchmark records, and the
validation/performance documentation; only current ratchets and open gates
remain authoritative here.

## Completed foundations

The independently observed 392-cell LMX/FreeMHD harness pins FreeMHD
`14b54a3e`, rejects source/contract drift, and passes two-update restart,
closure, Courant, and pressure-consistency gates. It proves orchestration and
bounded numerical consistency only; production parity and the canonical
three-mesh ladder remain open. Evidence is
`benchmarks/results/b2-freemhd-harness-smoke-20260715.json`.

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

The current gate passes 863 tests with 8 expected skips and 95.40% combined
coverage in 121.5 seconds. Shared centerline validation removes 20 source lines;
the layered-flow consolidation removes 17 test lines and one redundant
6.894-second solve while retaining finite-field, current, and physics gates.
The next test change remains a net deletion.

### Canonical sharding and performance

The schema-6 `8 x 7 x 7` 1/2/4-device gate proves exact topology, restart,
placement, conservation, and Anderson equivalence only. Production fields,
compact flux, and accelerator state shard axially; inlet state is replicated.
The global restricted-grid pressure correction preserves conditioning across
shards. Seconds-scale forced-XLA timings stay calibration evidence, never a
physical-core claim or reason to add an 8-device rung.

Rejected small probes stay in checksummed evidence. Physical-core claims need
verifiable affinity; forced macOS devices prove topology only.

The historical 32-update CPU calibration used only static preflight. A later
24-update ladder was correctly rejected for one/four-device instability,
two-device swapout, and one 117.941-second sample. The fresh monitored
evaluation at source/evaluator `9e49a9b` passes:
244.181/172.682/146.768-second medians, 1.414x/1.664x speedups,
4.709/5.201/5.486 GB peak RSS, 0.653% maximum CV, and clean
continuous/postflight traces. This closes Docker CPU-allocation scaling only;
do not relabel it physical-core scaling or steady-state evidence because every
trajectory ends at the fixed 32-update step limit.
The sustained CPU launcher now records one-second host samples through the
worker plus a 15-second postflight and binds the ignored JSONL digest to the
source fingerprint; any probe, affinity, swapout, foreign-work, or gap violation
blocks promotion. Worker records and final summaries independently rederive
the three-sample, two-minute, finite, 5%-CV timing gate. The remote GPU
supervisor additionally binds UUID/PCI
identity, worker contexts, utilization, and safe own-PID timeout cleanup.
Each multi-minute rung now collects and atomically publishes fresh admission
evidence immediately before launch, applies its admitted CPU affinity, and
keeps the local interpreter separate from a preflighted remote interpreter.

Compact CPU evidence is
`benchmarks/results/b2-schema6-cpu-scaling-20260716.json`; raw worker JSON and
plots remain ignored. The worker fingerprint must include package-owned frozen
specifications before any cross-host comparison is accepted.

GPU topology/replay is exact on 1/2 A4000s. The recorded-source `78858f5`
96-update calibration
reports 258.913/159.234 seconds and 1.626x speedup, but foreign contexts block
an authoritative timing claim. The latest audit found five foreign contexts on
each GPU, including a CPU-heavy VMEC-JAX process, so no new timing was launched.
A dedicated JAX/CUDA 0.6.2 and SOLVAX 0.8.4 benchmark environment now passes
the two-device import check. Preserve the calibrated `256 x 67 x 67`, 96-update
workload and rerun its one-cold-plus-three-warm ladder only after clean admission.

Every current calibration fails closed on physical repeat signatures, linear
status/history, placement, restart, conservation, and device equivalence.
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
exact; Docker CPU-allocation scaling is published. Exact M4 core mapping and
idle-host GPU timing remain open.

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

The current FreeMHD harness is B2-only, and the pinned source has no pipe/O-grid
ALEX case. Do not launch coarse B1 or claim parity. First freeze a solver-free
B1 input/observer contract; only then authorize a bounded `7 x 7 x 12`,
two-update Docker harness pilot. Retained-modal results remain numerical
evidence, not exact-formulation evidence.

Magnetic obstacles, Q2D turbulence, blanket models, mapped/pipe geometries,
inverse design, and other research workflows retain bounded portable examples
and tests. Promote each independently only when its own analytical/numerical,
physics, gradient, and performance gates pass; do not let B2 acceptance
self-promote unrelated features.

## SOLVAX ownership

LMX remains on the published range `solvax>=0.8.4,<1`. SOLVAX main `d0623a2`
prepares 0.8.5: 269 tests pass at 98.13% branch coverage, docs/lint pass, and
the built wheel reports version 0.8.5. No `v0.8.5` tag or PyPI release exists,
so LMX must not import the new API or use a Git dependency.

After publication, raise the tested minimum to 0.8.5, adopt
`additive_tridiagonal_line_preconditioner`, and delete LMX's duplicate 3D
builder only after exact primal, gradient, JIT, physics, and timing gates pass.
The prepared A/B is bitwise exact and would delete 35 production plus 28
duplicate-test lines. Then audit the smaller 2D potential-line duplicate under
the same gates.

LMX owns MHD equations, discretization, geometry, boundaries/gauges, corrected
flux, physical residuals, restart, sharding policy, observables, and acceptance.
SOLVAX owns reusable algebra that passes LMX's primal, transpose/gradient, JIT,
placement, memory, and repeated-timing gates. Rejected Anderson variants and
slower block-factor or packing prototypes remain history, not planned APIs.

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

Current README/docs media already covers accepted ducts and analytical/FreeMHD
comparisons plus research-stage blanket, Q2D, curved-pipe, fringing, obstacle,
restart, autodiff, and scaling results. All media is directly embedded and the
20 tracked derivatives remain below the 1.25 MiB cap. Do not create a B2 movie
from bounded or rejected trajectories, relabel mapped-pipe profiles as ALEX-B1
evidence, or add media without a new accepted physical result.

Read the Docs deployment remains externally blocked by stale GitHub
authorization (clone exit 128/webhook 406); reconnect it before release. The
local strict Sphinx build is green, so do not change repository configuration
to mask that account-level failure.

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
