# LMX authoritative development plan

Status: 2026-07-17. LMX consumes released SOLVAX 0.8.6 through the compatible
`solvax>=0.8.5,<1` range. Restart schema 6
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

A controlled current-source Docker/HLO trace confirms the CPU communication
defect: collectives occupy 70.8% of the captured projection slice and 52.9% of
momentum, with full-volume gathers inside GMRES. The 2-update and 32-update
ladders give essentially the same four-device speedup, and doubling axial
length does not improve it, so insufficient duration and axial size are ruled
out. Larger transverse sections only improve the compute/communication ratio.
Exact M4 P/E-core mapping remains open. Stages A and B of the stencil-owner
refactor are complete. The proposed Stage C compact-boundary topology is
rejected and reverted: it preserved the exact gates but did not improve the
four-CPU screen. The next action is a fresh phase profile of the Stage-B
baseline and an ownership audit of communication inside the SOLVAX Krylov path,
not another halo implementation or sustained ladder.

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
| README/docs | concise feature-led README, sourced comparison table, feature-specific visuals, steady-gated seven-second blanket loop, and Li/AlN convergence | replace legacy Hunt/Shercliff and Q2D loops only after their physical terminal gates pass; refresh B2/scaling only from accepted records |
| SOLVAX | released 0.8.6 owns generic algebra, Anderson weights, and the additive tridiagonal line preconditioner; LMX consumes the compatible `>=0.8.5,<1` API | add profiling ownership in SOLVAX's Krylov layer before another scaling algorithm change |
| Distribution | lean installed wheel loads all frozen A/B references and runs a tiny solve without Matplotlib/Pillow; plotting is an explicit extra; source artifact excludes repository tests | bump 1.1.3 before publication; hosted release gate must be green |

Current structure reflects completed ownership moves: frozen resources are
package-owned, evidence freezing lives with its campaign analyzers, diagnostics
have one NPZ/restart owner, stale private wrappers are gone, and repeated CLI,
example, validation, reporting, convergence, and minimal-TOML test setup is
shared without removing cases or assertions.

| Surface | Current | Active ratchet | CI hard ceiling |
|---|---:|---:|---:|
| package modules | 33 | no new module without retiring an owner | 34 |
| package lines | 32,788 | stay below 32,789 | 34,950 |
| maintained-core lines | 7,614 | stay below 7,615 | 7,800 |
| test files / lines | 29 / 20,243 | no new file; the next tranche must be a net deletion | 30 / 20,900 |
| maintenance scripts | 13 | no new script without retiring an owner | 13 |
| tracked files | 183 | no new file without retiring another owner | 186 |
| tracked checkout | 4,543,473 bytes | stay below 4,543,474 without hiding direct visuals | 4,718,592 bytes |

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

Migrate in deletion-producing tranches. Hartmann, Hunt, operator verification,
variable-field geometry, autodiff, exact extruded restart, and the rectangular
fringing diagnostic now follow the linear contract. The fringing replacement
is a real 113-line, roughly seven-second run with an `8.19e-16` charge residual;
its tranche removes 240 tracked lines and 112 mock-test lines. The example-only
`lmx.example_runner`, variable-field builder, and specialized autodiff plot
writer are deleted. The pipe comparison is now a 160-line explicit, editable
research-stage workflow; its tranche deletes 285 tracked lines and the package
reference-data wrappers rather than hiding geometry and CSV handling behind a
one-use API. The repository contract now covers nine of ten Python examples.
The FreeMHD workflow is a documented top-to-bottom script with editable path,
geometry, material, field, mesh, and solver inputs. It explicitly builds and
solves Shercliff and Hunt, has no parser, `main()` or guard, and scheduled
parity uses environment-backed execution plus the known JSON artifact. Its
two-commit tranche removes the 149-line one-use source wrapper, dedicated mock
fixtures, and import/mutate automation for 269 net tracked lines while
preserving the evidence schema and optional real-physics gate. The 700-line
research-evidence script remains above the preferred 360--430-line target;
reduce it only by finding genuinely reusable validation ownership or deleting
redundant evidence, not by hiding case construction or compressing readable
code. The only remaining Python offender is the 1,238-line scaling driver.

The 1,238-line
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

Slim source by ownership, not by arbitrary splitting. Stage B brought the
package to 34,036 lines before the pipe
tranche. Commit `b111813` removed the confirmed dead station fallback, obsolete
wrapper branch, and solver hook. Commit `8acf862` then moved three test-only
variable-field builders into explicit tests, deleting 142 package lines for 29
test lines; `fabca0a` folded the one-use square-duct helper for another 28
package lines. Finally `05e754a` transferred the duplicate additive line
preconditioner to released SOLVAX, deleting 35 package and 30 test lines while
retaining bitwise primal/JIT/JVP/VJP, coefficient-gradient, and four-shard
placement gates. After phase annotations and the FreeMHD ownership deletion,
the phase-timing diagnostic temporarily brought the tree to 33,485 package
lines. Removing the orphaned `lmx.showcase` workflow owner and its mock-only
test file then deleted 697 package and 233 test lines. The current tree is
32,788 package lines, `fringing.py` is 7,889 lines, and tests are 20,243 lines;
the diagnostic cost is fully recovered. Keep 33 modules and add no file.

The completed hotspot audit sets the next ownership order:

1. reclassify the 1,238-line scaling campaign as operational infrastructure and
   leave a short linear local scaling example; do not move it unchanged;
2. decide the 1,936-line Li/AlN `wall_study` leaf: retain reusable wall physics
   and make its actual workflow visible in one example before retiring wrappers;
3. keep the computational core of `blanket_flow`, but move one-off JSON, CSV,
   plot, frame, and movie assembly into a visible blanket example;
4. keep differentiable objectives and primitives in `autodiff`, while moving
   host gradient-descent loops and history assembly into the design example;
5. decide whether NumPy/SciPy `q2d` is ported to JAX with a real example or
   explicitly demoted from the differentiable core;
6. only then separate the reviewed duct and mapped-pipe owners inside the
   7,889-line fringing solver, preserving current public imports during the
   transition.

`plotting.py` stays one optional visualization owner until orphan writers are
identified. `external_validation.py`, `solvers.py`, `freemhd.py`, and
`validation.py` follow after their consumers are explicit. For each file,
inventory public contracts and call sites; delete one-use example
builders and duplicate adapters; merge repeated numerical phases; and extract a
new module only when it creates a stable reusable owner while retiring at least
as much old surface. Every public function receives a concise contract docstring
and every non-obvious numerical phase records its invariant or reference.
Comments explain decisions and equations, not syntax. Trivial private algebra
does not receive noisy commentary merely to satisfy a count.

Each tranche must reduce package or example lines and tracked files, preserve
all public functionality that remains claimed, and pass focused numerical and
physics tests in under two minutes before commit and push. Run the complete
portable gate once after a coherent structural/source group and before the next
algorithmic stage. Stages A and B finish at 34,036 package lines, below the
34,038-line pre-Stage-A baseline; the pipe, dead-fallback, builder, and SOLVAX
ownership tranches plus the FreeMHD/showcase deletions lower the clean tree to
32,788. Do not combine a structural move with a numerical algorithm change.

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

The current complete gate after the phase diagnostic and orphan-showcase
deletion passes 841 tests with 5 expected skips and 95.33% combined coverage in
131.9 seconds. It remains
below both the 300-second engineering target and 600-second hard limit. Keep
the six-worker setting unless a fresh slow-node profile finds a node above 45
seconds or the gate materially regresses; example/mock deletion and the new axial-injection
gate account for the changed test count.
Merging identical magnetic-obstacle checks preserves 33 assertions, removes two
solves and 28 lines, and cuts focused wall time 5.86 -> 4.52 seconds. The current
suite is 20,662 lines; its next change remains a net deletion. Across the
editable-example, runner-deletion, autodiff-slimming, and steady-media-gate
tranches, package and test lines remain net lower than the prior baseline.

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

The controlled Docker/HLO profile at `128 x 35 x 35` now reproduces the
communication defect with nested eight-vCPU affinity. Profile-versus-timed
signatures and linear histories, exact restart state/flux/history, Anderson,
conservation, and four real shards all pass. The captured projection slice
spends 0.288 of 0.407 seconds in collectives (70.8%); momentum spends 0.045 of
0.084 seconds (52.9%). HLO contains twelve full `128 x 33 x 33 x 3` velocity
all-gathers of 3,345,408 bytes, including six inside GMRES loops, plus eleven
full scalar-field gathers. Pressure PCG uses only 5,120-byte axial-mean gathers;
the electric coarse modal gather is 173,056 bytes. Insufficient duration and
axial size are ruled out, and the communication-defect gate passes.

The accepted Stage-B implementation is now profiled phase by phase before any
new topology change. Zero-numerics annotations cover LMX momentum, projection,
electric, reconstruction, defect, and Anderson work plus SOLVAX matvec,
preconditioner, reductions, and line solves. The non-truncated four-device
trace and ordinary asynchronous controls have exact signatures, linear
histories, restart, and placement. Report union time and per-iteration time,
never summed concurrent device activity.

The zero-numerics annotations are now present in LMX and released on SOLVAX
main at `bb9ee81`. Raw XPlane attribution for the current `128 x 35 x 35`,
four-device, two-update control gives projection 46.23%, electric 36.04%,
momentum 13.15%, defect 4.21%, and reconstruction plus EMF 0.36% of selected
module time. Projection averages 231.60 ms, about 3.5 times momentum. Its two
151/155-iteration solves each perform one all-reduce, one preconditioner
all-gather, and two neighbor exchanges per PCG iteration. Device-zero
communication occupies 19.8/20.6% of projection wall, while the additive
tridiagonal line-solve scope occupies 54.5/54.1%. The bottleneck is therefore
both local line work and fixed synchronization, not an undersized momentum
problem alone. The profiled trace has 28.9% overhead relative to the best
ordinary warm sample, so these shares establish ownership but do not replace
throughput measurements.

Before another algorithm change, use the opt-in synchronized phase timers in
the scaling worker only. They synchronize inputs and outputs around momentum,
projection, electric, reconstruction, and defect while retaining an unbarriered
control. A successor should target the per-iteration
preconditioner all-gather or a distributed/transversely owned line solve, then
reduce PCG iterations or synchronization. Larger transverse workloads may
improve compute/communication ratio but cannot remove the measured fixed costs.

That synchronized screen is complete. The diagnostic is a separate excluded
trajectory; disabled execution retains the original compiled functions and
ordinary asynchronous schedule. At `128 x 35 x 35`, two updates, the 1/2/4
ordinary warm medians are 1.3409/1.1654/1.0023 seconds, or only 1.151x/1.338x
speedup, with 6.5--9.6% CV. Synchronized totals are
1.3413/1.2673/1.1927 seconds. Projection improves only
0.5955 -> 0.5687 -> 0.4828 seconds and electric only
0.5104 -> 0.4720 -> 0.4252, while momentum regresses at four devices from
0.1438 to 0.1756 seconds. Every rung preserves its ordinary-run signature and
linear history; all ten production fields have the requested placement. The
four-device rung misses the frozen interface-current gate at `1.0298e-3`, and
all timing CVs exceed 5%, so this is diagnostic evidence only. It confirms
that a larger problem may improve ratios but will not repair the weak scaling
of the pressure/electric solvers. No sustained ladder is authorized before a
profile-gated solver candidate.

The existing evidence already narrows ownership: momentum is only 0.084 seconds
of the captured active slice versus 0.407 seconds for projection, and the trace
contains roughly 560 PCG-loop collective events versus about 20 occurrences of
each principal GMRES gather. SOLVAX single-reduction PCG is already enabled, so
the next profile must distinguish synchronization from transverse-line,
axial-mean, and replicated coarse-correction compute. Determine whether Krylov
can own compact sharded operator communication without duplicating LMX's physics
algebra. Preserve
restricted-grid pressure conditioning; the global modal correction cannot be
deleted merely because it communicates. Docker verifies guest affinity but
exposes opaque vCPUs, so exact mapping to the M4's four performance and six
efficiency cores remains open.

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

The first explicit-halo momentum candidate is rejected and reverted. It removes
all six full-vector gathers from GMRES, improves the affected phase 20.23%, wall
time 8.42%, and RSS 1.19%, while preserving every exact numerical gate. But
collective time improves only 20.79%, below the predeclared 25% threshold,
because twelve static one-plane permutations replace the gathers and permutation
time more than doubles. The next bounded candidate must compute one reusable
field halo for both diffusion and convection and eliminate the separate west
transport exchange; do not rerun the reverted variant or relax its gate.

The second candidate reuses field and face halos and is also rejected and
reverted. It is 26.94% faster in the bounded wall screen, improves momentum
17.59%, and keeps RSS growth to 2.23%, with every exact gate green. Yet collective
time falls only 17.01% because the remaining directional plane permutations
increase permutation union from 7.83 to 18.23 ms. Do not spend a sustained rung
on this attractive small-case result. A materially different successor must
remove the two remaining setup full-vector gathers and scalar setup gathers, or
change axial operator/partition ownership; further rearrangement of the same
two directional halos is closed.

A proposed two-compact-gather successor is rejected before timing on the source
slimming gate. It compiles and passes a real four-shard `8 x 4 x 3` momentum
solve at `1.18e-11`, but adds 344 net solver lines by duplicating limited
gradients, weights, stress, diffusion, and convection. Do not compress or keep a
second algebra path for speed. The prerequisite shared-owner refactor was later
completed, and the resulting Stage C compact-boundary candidate is separately
rejected below; this topology path is now closed.

The shared-owner refactor completed two useful deletion-producing stages.
Stage A at `725f132` added optional cell-aligned `(west,east)` injection to the
existing gradient, limiter, stress, diffusion, and convection primitives for 13
net source lines while preserving primal, JVP, and VJP oracles. Stage B at
`592b8b5` packs `(u,v,w,q)` gradients and gives solver and post-projection defect
one frozen setup owner and one transport action. It preserves affine-inlet
arithmetic and the distinct defect state and finishes at 34,036 package lines,
below the pre-Stage-A baseline.

The proposed Stage C compact-boundary candidate is rejected and fully reverted.
After fixing its affine-inlet boundary-action bug, exact signatures, histories,
restart, conservation, gradients, and placement pass; one- versus four-device
results differ only at floating-point roundoff. Its production compact gathers
are 278,784 and 348,480 bytes, with the intended removal of full-volume setup
payloads. Nevertheless, on the frozen `128 x 35 x 35`, six-step four-CPU screen,
its 2.192642-second median is 0.03% slower than the Stage-B 2.191924-second
median, far short of the required 10% medium-screen gain. No multi-minute run
was launched. The candidate also raises total electric iterations from 371 to
384, including 57 to 70 on its second step, while pressure iterations stay
effectively unchanged. That extra downstream work and latency-heavy plane
permutations explain why smaller setup payloads did not improve the complete
solve. This topology is cancelled: do not restore it, relax the gate, or
try another arrangement of the same boundary slabs without a new profile and a
materially different communication owner.

Keep the GPU diagnosis separate. Its existing trace spends only about 9% of
device-active time in collectives and about 75% in transverse SOLVAX line work,
so GPU promotion should optimize or fuse compute kernels while preserving exact
placement. Do not impose a CPU communication remedy on the GPU path without a
new GPU trace. The authoritative two-A4000 ladder still waits for clean host
admission.

Exit order: annotations, raw XPlane attribution, and the synchronized Stage-B
1/2/4 screen are complete. Decide whether SOLVAX's Krylov/operator layer can
remove the pressure/electric preconditioner all-gather or own a distributed
line solve without importing LMX physics. The orphan-showcase deletion has
already recovered the 51 diagnostic lines and a further 646 package lines. A
new solver candidate must first pass exact/HLO and bounded four-device gates, then the
fixed-work medium CPU gate, and only then the multi-minute local CPU ladder with
exact M4 core mapping. Optimize the separately profiled GPU compute path and rerun its
multi-minute ladder only after clean office-host admission. Portable tests stay
below ten minutes and all topology, replay, numerical, and restart gates remain
exact; seconds-scale runs remain debug screens.

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

The SOLVAX Anderson integration and LMX schema 6 are complete. The CPU topology proves one shared
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

LMX now uses the published range `solvax>=0.8.5,<1`; the clean environment
resolves released 0.8.6. Commit `05e754a` adopts
`additive_tridiagonal_line_preconditioner` and deletes LMX's duplicate 3D
builder and direct unit contract. Duct, pipe, periodic, primal, JIT, JVP, VJP,
coefficient-gradient, and forced four-shard comparisons are bitwise exact. The
Python 3.10 compatibility lane retains 0.8.5 as the tested minimum, so this is a
compatible range rather than a latest-release lock. Next audit the smaller 2D
potential-line duplicate under the same gates.

Before another scaling algorithm change, add profiling scopes to SOLVAX GMRES
and PCG ownership: matvec, preconditioner, Arnoldi/inner-product reductions, and
line solves. Make that change from a clean worktree based on current
`origin/main`; the existing local SOLVAX checkout has unrelated user changes
and must not be overwritten.

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

The shared animation writer enforces at most 7.000 seconds and tests encoded
WebP duration, not just frame count. The blanket trajectory is physically
accepted: its 18-update gate first passes at step 57 (`t=2.85 s`) with
`1.860420834008e-3 <= 2e-3`; the visible trajectory stops at step 58
(`t=2.90 s`) with `1.676904288502e-3`, plays for exactly 7.000 seconds, and is
62,522 bytes. The `3.000354787140735e-14` metric belongs only to the full
15-second source. Commit `2df2336` links the displayed derivative, accepted
window, source hash, and historical scalar records in the manifest and docs;
do not rerun it.

The current Hunt/Shercliff loop is unaccepted even though it plays for exactly
7.000 seconds: it has no recorded terminal residual. The shared velocity,
linear, and potential predicate now stops snapshot generation at its first full
pass and fails publication at the ceiling; robust volume-weighted CG passes the
bounded backend gate while legacy CG does not. Keep the old 200-update asset
explicitly labelled transient. Calibrate the physical horizon, then regenerate
serially only after clean host admission and record the terminal triplet and
source hashes.

The current weakly forced Q2D loop is also unaccepted: it plays for 7.014
seconds, covers only 0.331 turnover times, and ends with energy and enstrophy
still changing substantially. Replace it only with an explicitly forced
trajectory that lasts at most 7.000 display seconds and passes precommitted
rolling energy, enstrophy, and RMS-drift gates after at least three turnover
times. Calibrate coarsely first, then run one production trajectory.

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
