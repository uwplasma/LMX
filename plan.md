# LMX authoritative development plan

Status: 2026-07-17. LMX consumes released SOLVAX 0.8.4. Restart schema 6
provides compact, exact one-plus-one replay and correct 1/2/4-CPU plus 1/2-GPU
placement. Its tested Anderson and bounded affine variants miss the frozen 15%
velocity-map gain, so fixed relaxation 2 remains the B2 control. The current
coarse B2 trajectory passes linear, conservation, restart, and placement gates
but misses steady acceptance; no continuation is authorized.

The accepted callback-free `256 x 67 x 67`, 32-update CPU ladder at source
`3339e95` gives 244.763/173.033/158.354-second medians, 1.415x/1.546x
two/four-device speedups, and 2.318% maximum CV. Observer exclusion, exact
restart, numerics, placement, memory, and continuous/postflight monitoring pass.
This establishes Docker CPU-allocation scaling; exact M4 P/E-core mapping and
steady-state evidence remain open. The multi-minute 1.626x two-A4000
calibration remains non-authoritative because foreign contexts block GPU
admission.

A current-source native four-device Perfetto trace finds collectives in about
48% of device-active projection and momentum time. The 2-update and 32-update
ladders give essentially the same four-device speedup, and doubling axial
length does not improve it, so insufficient duration and axial size are ruled
out. Larger transverse sections improve the compute/communication ratio. The
native trace is not affinity-equivalent to the accepted Docker ladder, however;
heterogeneous M4 P/E cores and JAX CPU scheduling remain confounders. Obtain one
controlled Docker/HLO trace before calling communication the sole sustained-run
bottleneck, and do not repeat a sustained ladder until a bounded, physics-valid
candidate clears the promotion gates below.

A composed momentum/projection JIT is rejected before sustained timing: its
four-device medium screen improves only 3.6%, raises peak RSS 15.4%, and fails
the frozen interface-current gate. Do not spend a multi-minute rung on it.
A one-transfer projection guard is also rejected: its exact 3.23% gain misses
the 5% gate; RSS falls 1.68%.
A broader diagnostic bundle is rejected too: it preserves every numerical,
restart, history, interface-current, and placement gate, but changes the
four-device warm median from 1.971 to 2.011 seconds (2.06% slower) in its clean
screen; an independent rerun regresses further. Do not sustain either variant.

The `--sustained` preset defaults to 32 CPU or 96 GPU updates, one cold plus three
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
- `examples/`: small, self-contained, editable Python workflows;
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
| package lines | 34,248 | stay below 34,249 | 35,100 |
| maintained-core lines | 7,869 | stay below 7,870 | 8,000 |
| test files / lines | 30 / 21,076 | no new file; next test change must reduce lines | 31 / 21,100 |
| maintenance scripts | 13 | no new script without retiring an owner | 13 |
| tracked files | 186 | no new file without retiring another owner | 187 |
| tracked checkout | 4,592,161 bytes | stay below 4,592,162 without hiding direct visuals | 4,718,592 bytes |

These ratchets come from ownership deletion and shared helpers, never unreadable
formatting or arbitrary test merging. Recent moves delete 190 builder lines and
reuse three validated Q2D trajectories; exact nondefault/full-solve, physics,
energy, and artifact gates remain. B1 stays direct and B2 reuses checkpoints.
The remaining result records, 13 scripts, and distinct Python/TOML examples
retain current provenance, CI, or runnable-workflow ownership; do not merge
them unless the replacement deletes code without weakening those contracts.

Completed slimming, distribution, SOLVAX, scaling-admission, and portable-gate
history is preserved in Git, checksummed benchmark records, and the
validation/performance documentation; only current ratchets and open gates
remain authoritative here.

## Priority 1: make the code and examples inspectable

Every Python example follows one visible, uniform workflow:

1. a module docstring states the physics, maturity, runtime, and outputs;
2. imports are followed immediately by a commented block of editable geometry,
   material, solver, runtime, and output parameters;
3. only example-specific local helpers follow; reusable numerical or output
   logic remains in `lmx/`;
4. the file explicitly builds the geometry, field, boundaries, and problem;
5. setup, solve, validation, saving, and plotting run from top to bottom;
6. there is no argument parser, `main()`, or `if __name__ == "__main__"` guard.

Defaults stay deliberately small enough for interactive use. Expensive accepted
records are regenerated by documented research commands, not by making the
introductory example slow. Tests execute examples with `runpy` or a subprocess
and enforce this structure with one shared AST contract.

Migrate in deletion-producing tranches. First convert Hartmann, Hunt, and the
operator check, then delete the example-only `lmx.example_runner` surface and
replace catalog-dependent tests. Next make variable-field, fringing, restart,
and pipe construction explicit and delete one-use convenience builders once no
caller remains. Then reduce the autodiff and FreeMHD workflows. The 1,238-line
strong-scaling program is operational orchestration, not a pedagogical example:
consolidate its reusable admission/monitor/worker logic into the existing
scaling owner and leave only a short editable scaling workflow, or classify the
reproducer as maintenance tooling while keeping a small example. Do not move
1,238 lines unchanged to another folder.

After the Python replacements pass, delete `examples/cases/`, `catalog.toml`,
the root TOML example, and the tracked tabulated-field fixture. This removes
about 1,040 configuration lines, nine case files, a catalog, and a 32 KiB
fixture, and fixes two real broken workflows: the nested tabulated-fringing case
cannot find its NPZ file, and the documented Hartmann restart paths disagree.
Keep one copy-paste TOML schema in documentation only if TOML remains a supported
CLI input.

Slim source by ownership, not by arbitrary splitting. The first audit targets
`fringing.py` (8,275 lines, including a 2,188-line extruded solve), then
`plotting.py`, `external_validation.py`, `solvers.py`, `wall_study.py`,
`blanket_flow.py`, `autodiff.py`, `q2d.py`, `freemhd.py`, and `validation.py`.
For each file, inventory public contracts and call sites; delete one-use example
builders and duplicate adapters; merge repeated numerical phases; and extract a
new module only when it creates a stable reusable owner while retiring at least
as much old surface. Every public function receives a concise contract docstring
and every non-obvious numerical phase records its invariant or reference.
Comments explain decisions and equations, not syntax. Trivial private algebra
does not receive noisy commentary merely to satisfy a count.

Each tranche must reduce package or example lines and tracked files, preserve
all public functionality that remains claimed, pass focused numerical/physics
tests in under two minutes, then pass the complete portable gate before commit
and push. Do not combine a structural move with a numerical algorithm change.

Exit: every example satisfies the shared workflow contract; the broken and
opaque case/catalog layer is gone; no source file remains thousands of lines
without a documented, reviewed ownership rationale; public APIs and numerical
kernels are documented; total files and lines fall without weakening physics,
numerics, differentiation, restart, or parallel evidence.

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

The current gate passes 867 tests with 8 expected skips and 95.41% combined
coverage in 171.1 seconds after the sustained CPU run. It remains below both
the 300-second engineering target and 600-second hard limit; investigate the
thermal/system-state increase before changing worker count.
Merging identical magnetic-obstacle checks preserves 33 assertions, removes two
solves and 28 lines, and cuts focused wall time 5.86 -> 4.52 seconds. The current
suite is 21,076 lines; its next change remains a net deletion.

### Canonical sharding and performance

The schema-6 `8 x 7 x 7` 1/2/4-device gate proves exact topology, restart,
placement, conservation, and Anderson equivalence only. Production fields,
compact flux, and accelerator state shard axially; inlet state is replicated.
The global restricted-grid pressure correction preserves conditioning across
shards. Seconds-scale forced-XLA timings stay calibration evidence, never a
physical-core claim or reason to add an 8-device rung.

Rejected small probes stay in checksummed evidence. Physical-core claims need
verifiable affinity; forced macOS devices prove topology only.

Apply the operating contract's multi-minute ladder after the next
performance-affecting solver or sharding change. Bitwise-equivalent refactors
use focused gates; seconds-scale results remain topology/debug evidence.

Rejected earlier ladders used static preflight, retained a timed checkpoint, or
failed stability, swapout, and duration gates. The authoritative callback-free
ladder at source `3339e95` records 244.763/173.033/158.354-second medians,
1.415x/1.546x speedups, 4.15/4.33/4.48 GiB peak RSS, 2.318% maximum CV, exact
restart, and clean continuous/postflight traces. The sustained CPU launcher
records one-second host samples through the worker plus a 15-second postflight
and binds the ignored JSONL digest to the source fingerprint; any probe,
affinity, swapout, foreign-work, or gap violation blocks promotion. Worker
records and final summaries independently rederive the three-sample,
two-minute, finite, 5%-CV timing gate. The remote GPU
supervisor additionally binds UUID/PCI
identity, worker contexts, utilization, and safe own-PID timeout cleanup.
Each rung collects fresh admission immediately before launch, applies CPU
affinity, and separates local/remote interpreters. Per-rung paths preserve every
raw 1/2/4-CPU and 1/2-GPU observation.

Compact CPU evidence is
`benchmarks/results/b2-schema6-cpu-scaling-20260716.json`; raw worker JSON and
plots remain ignored. The worker fingerprint must include package-owned frozen
specifications before any cross-host comparison is accepted.

GPU topology/replay is exact on 1/2 A4000s. The recorded-source `78858f5`
96-update calibration reports 258.913/159.234 seconds and 1.626x speedup, but
foreign contexts block an authoritative timing claim. The latest audit found
four gunicorn CUDA contexts on each GPU and a `pop-upgrade` daemon near 99% CPU,
so no new timing was launched.
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

The current native four-device CPU profile at `128 x 35 x 35` is diagnostic
only, but profile-versus-timed signatures, linear histories, and placement
agree. Across device-active intervals, mixed projection spends 0.135 of 0.280
seconds in collectives (48.1%) and momentum spends 0.036 of 0.074 seconds
(48.3%). The projection trace attributes about 66.6 ms to permutes, 37.4 ms to
all-gathers, and 31.2 ms to all-reduces; repeated PCG reductions and replicated
transverse or axial coarse corrections dominate the cadence. An earlier
controlled gauge trace estimated only about 5.4% combined collective activity,
so first reproduce the current source inside the immutable Docker environment
with nested 1/2/4-device affinity, dump HLO, and record collective operand bytes,
rendezvous/pending time, solver iteration counts, local line-solve time, shard
bytes, replicated coarse bytes, and actual host P/E-core mapping.

Audit HLO shapes and owners in this order: implicit-momentum neighbor halos,
transverse modal correction, axial-mean preconditioning, then PCG reduction
count. A full-volume all-gather per Krylov iteration or at least 20% controlled
collective-plus-rendezvous occupancy establishes a communication defect.
Preserve nonperiodic halo semantics and restricted-grid pressure conditioning;
the global modal correction cannot be deleted merely because it communicates.

Use three performance gates. A tiny exact gate checks signatures, linear
histories, interface current, restart, placement, and gradients. A bounded
four-device profile candidate must reduce collective time by at least 25%, the
affected phase by 15%, and two-update wall time by 8%, with unchanged or fewer
iterations and no more than 5% RSS growth. A fixed-work 1/2/4-device medium
screen must then improve four-device warm time by at least 10% with at most 5%
CV. Only then run the one-cold-plus-three-warm, multi-minute sustained ladder.
A larger transverse section may confirm asymptotic behavior after the
algorithmic candidate passes; increasing runtime or axial length alone is not a
remedy. Never run rungs concurrently on the same Mac.

Keep the GPU diagnosis separate. Its existing trace spends only about 9% of
device-active time in collectives and about 75% in transverse SOLVAX line work,
so GPU promotion should optimize or fuse compute kernels while preserving exact
placement. Do not impose a CPU communication remedy on the GPU path without a
new GPU trace. The authoritative two-A4000 ladder still waits for clean host
admission.

Exit: portable tests stay below ten minutes; CPU/GPU topology and replay remain
exact; callback-free Docker CPU-allocation scaling remains passing. Next obtain
the authoritative idle-host GPU multi-minute ladder, then multi-minute exact-M4
core mapping; seconds-scale runs remain debug screens. Both claims remain open.

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

The current FreeMHD harness is B2-only, but the pinned source's public additional
cases map "Pipe Flow Fringing B" to `S3_Buhler_Ha616.zip` (Drive ID
`1vrLEVOk2NzH6O_Qv80ze0BPM0kM0rYoF`, 1,930,938,126 bytes, SHA-256
`6aa165e95275f29fd2d014c6ef3a91e4e8d80b16d82b3040d9d6b90ab74e6091`). Its native
fluid/conducting-wall O-grid is geometry-compatible with LMX's pipe mapping and
was prepared for 96 MPI ranks, but the archive does not expose a reuse license
and its Ha=616 PbLi-like setup is not parameter-identical to frozen ALEX B1 at
Ha=6600. Never add the 1.93 GB archive or its 8.77 GB expanded mesh to Git.
The manual runner's `--freemhd-s3-preflight` gate streams the full hash and
verifies seven small members without extraction. An explicit local-only command
now safely extracts a 34-file allowlist, builds the 2,560-fluid plus
512-conducting-wall O-grid, verifies the image's FreeMHD repository at `14b54a3`
and pins the resolved Docker image,
and fails closed on its two-rank, two-update logs. A private pilot passed both
updates in 23.229 seconds end to end under x86 emulation, with exact balanced
1,280-fluid plus 256-wall cells per rank and no fatal or nonfinite marker. This
is only a native-S3 reduced-pipe harness smoke: it always denies full-S3 parity,
ALEX-B1 equivalence, steady acceptance, archived-observer eligibility, and
redistribution. The archive-to-report CLI still needs one end-to-end replay when
the verified 1.93 GB user-owned ZIP is locally available; then freeze only its
compact report and independently match the S3 formulation in LMX without
relabeling it ALEX B1.
Retained-modal results remain numerical evidence, not exact-formulation parity.
The supplied 2.62-million-cell archive ran 8.22 hours on 96 ranks, records no
FreeMHD Git SHA, and its 3.08-second centerline CSV contains a nonfinite row;
the 2.28-to-3.08-second velocity change is still about 10%. Therefore archived
outputs are neither pinned-source nor steady acceptance oracles. A fresh reduced
pinned-source run must own repeat, restart, and conservation evidence, while
nonfinite archived observers fail closed.

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

A `v0.8.5` tag automatically triggers SOLVAX's trusted PyPI publication, so it
requires explicit release authority rather than ordinary commit/push authority.
If publication is authorized after 2026-07-16, first correct the prepared dates
in SOLVAX's `CITATION.cff` and `CHANGELOG.md`, push and await green checks, then
tag the exact green commit and wait for PyPI metadata before changing LMX.

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

Aim for readable loops no longer than 7.000 seconds, stills below 100 KiB, and
tracked movies below 150 KiB where practical. Keep all tracked media below 1.25
MiB. The 1.25 MiB
cap admits the two evidence-rich P2 composites while preserving a hard,
test-owned budget. Host source frames, full-quality media, meshes, and raw
outputs stay in checksummed releases. Put provenance and acceptance status
beside every asset.

The README is 617 words with 14 directly embedded visuals. README/docs media
covers accepted ducts and analytical/FreeMHD comparisons plus research-stage
blanket, Q2D, curved-pipe, fringing, obstacle, restart, autodiff, and scaling
results. The 20 tracked derivatives remain below the 1.25 MiB cap.

Movie duration and simulated duration are separate contracts. Each source
trajectory declares a physical steady-state or statistically steady observable,
tolerance, window, and consecutive-pass count, and stops at the first sustained
pass. The compressed derivative then resamples that complete trajectory to at
most 7.000 seconds without claiming that resampling created convergence. Store
the terminal metric, first-passing step/time, source-frame hash, and compressed
duration in the manifest.

First make the shared animation writer enforce 35 timing slots at 5 fps and
test the sum of encoded WebP frame durations, not just the frame count. The
current blanket source has a valid `3.00e-14 <= 2e-3` relative-update gate, but
its derivative runs 7.014 seconds and continues from its first sustained pass
at step 57, `t=2.85 s`, to `t=15 s`. Trim through the first stored state after
that pass (step 58, `t=2.90 s`) without rerunning, then recompress. Hunt/Shercliff
runs exactly 7.000 seconds but has no recorded terminal residual or early-stop
gate; reuse the solver's velocity, linear, and potential convergence predicate,
stop at its first full pass, and fail publication if the ceiling is reached.
Q2D runs 7.014 seconds, covers only 0.331 turnover times, and ends with energy
and enstrophy still changing substantially. Its current unforced decay cannot
be called statistically steady at nonzero energy: either run it to a
precommitted quiescent endpoint or replace it with an explicitly forced case
that passes rolling energy, enstrophy, and RMS-drift gates after at least three
turnovers. Calibrate coarsely first, then run one production trajectory.

Do not create
a B2 movie from bounded or rejected trajectories, relabel mapped-pipe profiles as ALEX-B1
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
