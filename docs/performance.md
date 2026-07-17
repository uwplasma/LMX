# Performance and scaling

LMX runs through JAX on CPUs and GPUs. Performance claims are accepted only for
the real solver path with identical numerical results; visibility of multiple
devices alone is not evidence of parallel execution.

![Monitored multi-minute CPU scaling and shared-host GPU calibration](_static/strong_scaling.webp)

The upper row is the accepted callback-free fixed-work ladder on 2/4/8 Docker
guest CPUs for 1/2/4 JAX devices. The lower row is a multi-minute one/two-GPU
shared-host calibration; foreign contexts still fail its idle gate. Short
timings remain debug/calibration evidence; the multi-minute GPU ladder remains
calibration until idle-host admission passes.

## Current evidence

| Path | Hardware and grid | Result | Interpretation |
|---|---|---|---|
| portable test gate | Apple M4, six workers | 859 pass, 8 skip, 95.40% combined line/branch coverage, 146.7 s | 49% of the five-minute target and 24% of the ten-minute budget |
| B2 CPU smoke | Apple M4, `8 x 7 x 7`, 1/2/4 forced CPU devices | current-source pressure observable agrees within `5.93e-15`; closure and exact restart pass | production sharding correctness; too small for scaling claims |
| B2 monitored CPU scaling | `256 x 67 x 67`, 32 updates, 2/4/8 guest CPUs, 1/2/4 JAX devices, three warm trajectories per topology | 244.763/173.033/158.354 s; 1.415x/1.546x; CV at most 2.318% | accepted CPU-allocation scaling at source `3339e95`; observer exclusion, exact restart, numerics, placement, memory, and continuous/postflight monitoring pass |
| B2 multi-minute GPU calibration | `256 x 67 x 67`, 1/2 RTX A4000, 96 updates, three warm trajectories per topology | 258.913/159.234 s; 1.626x; peak device memory 2.50 GB vs 1.41/1.31 GB; CV below 0.29% | recorded source `78858f5`; numerical/duration gates pass, but persistent foreign contexts block an authoritative timing claim |
| B2 pseudo-time gate | Apple M4, corrected warm `7 x 7 x 7`, canonical `Ha=2900`, `N=540` equations | 64x/32x/16x map-rate spread 0.0768%; raw updates halve; exact restart | 64x refrozen; versioned stopping threshold open |
| B2 physical-residual probe | Apple M4, `7 x 7 x 7` | residual 0.03870 at step 90; omitted reconstruction force is 0.0880 at step 4 | diagnostic has a plausible coarse split floor; never a stopping gate |
| CPU-allocation restart audit | existing `101 x 77 x 77` coarse checkpoints | geometry/shape load, but sampled files normalize to `legacy_nonexact` with no source fingerprint or schema-6 Anderson/compact-flux state | unsuitable for exact or promoted current-source scaling evidence |
| B2 projection audit/fix | Apple M4, warm same-state `7 x 7 x 7` | pre-fix raw cell-update floor `3.69e-3`; corrected predictor-preserving projection removes axial floor | pre-fix trajectories invalid; focused physics/autodiff/restart gates pass |
| B2 stopping study | Apple M4, schema 5, restart segments to step 96 | `0.05` converges at 30 but fails all QoI limits; `0.005` reaches only `0.01502` | fail closed; accelerate before tighter-reference calibration |
| B2 acceleration gates | Apple M4, cold `7 x 7 x 7`, six updates | raw Anderson ends at `0.5281`; bounded fallback ends at `0.114718` vs `0.114466` control; 98.254--99.887% of shared residual energy is potential | both rejected; shared-norm objective is misaligned with velocity convergence |
| B2 fixed relaxation | canonical `min=max=2` | exact trajectory/restart; prior residual omitted | saves 18.46 MiB at `102 x 77 x 77` and two global dot products/update |
| B2 GPU smoke | 1/2 RTX A4000 GPUs, `8 x 7 x 7` | current schema-6 placement, replay, conservation, linear, and repeat gates pass; two-GPU state error `2.22e-16`, flux exact | topology correctness only; shared-host timing excluded |
| historical SOLVAX PCG equivalence | Apple M4 CPU and RTX A4000 GPU | 0.8.2 forward, gradient, transpose, memory, and Hartmann gates pass; one-shot GPU warm ratio is 1.184 | archival; PCG is unchanged and no refresh is planned |
| legacy B2 medium independence | 2 x RTX A4000, `152 x 113 x 113` | all four numerical variants pass; final confirmations take 34.94--57.64 s | superseded no-inertia/stationwise-flow formulation |
| legacy B2 fine campaign | 2 x RTX A4000, `202 x 149 x 149` | baseline 457.37 s; iteration and wall confirmations 65.23/62.10 s | superseded formulation; tight-tolerance independence remains open |
| B1 modal setup | RTX A4000, `11 x 17 x 32` | 57.85 s first, 10.63 s restart | accepted setup optimization |
| B1 large solve | RTX A4000, `21 x 24 x 64` | 270.42 s for two updates | pressure projection is 91.2% of runtime |
| B1 physical-pilot gate | RTX A4000, `21 x 24 x 64` | 669 iterations for solve plus restart vs 768 fixed ceiling | all four physical projections pass; shared-host wall time is not a speedup claim |

![B2 field, pressure, and rejected raw and bounded acceleration gates](_static/readme-alex-b2-field-pressure.webp)

The six-update gate is intentionally small: it rejects the accelerator before
the step-29 trajectory while preserving exact restart and physics diagnostics.
Zero of five adjacent residual pairs passes the predeclared predicted-gain,
conditioning, and coefficient-bound rationale gate. This closes generic
shared-Euclidean Anderson tuning; it does not authorize another long run.

The CPU/GPU correctness calibrations and larger results are recorded in
`benchmarks/results/b2-cpu-device-equivalence-20260715.json`,
`benchmarks/results/b2-gpu-device-equivalence-20260715.json`,
`benchmarks/results/b2-schema6-cpu-scaling-20260716.json`, and
`benchmarks/results/b2-gpu-scaling-calibration-20260715.json`.
The seconds-scale diagnostic trace measures a 1.510x phase ratio across
momentum, projection, and electric work; it is not strong-scaling evidence.
Reusing the existing host station payload for validation cuts
the two-GPU end-to-end median by 0.290 s and removes 12 source lines, but the
remaining post-map transfer tail leaves the result below 1.2x. The fixed-grid
ladder therefore stops without a larger run or production-speed claim.
The later 96-update sustained GPU lane uses the current mixed replay contract
and supersedes that conclusion only as a shared-host calibration: every warm
trajectory lasts 158--259 s, one/two-GPU topology agrees, and the median ratio
is 1.626x. Four persistent external service contexts were present before and
after the run, so this result is visible but not promoted to authoritative GPU
strong scaling.
The large deterministic probe isolated restart variation to corrected face
flux (`4.40e-6` relative). A three-update trajectory preserved every primary
field exactly and reduced the flux difference to `6.25e-7`, within the frozen
`1e-5` calibration gate. The tiny canonical smoke retains its strict all-field
replay gate. GPU steady-production scaling remains open. The older,
larger GPU B2 result passes its historical two-device equivalence gate, but
the measured path omits canonical inertia and uses stationwise flow forcing.
It therefore does not establish scaling for the matched formulation. The B1
timings do not promote its experimental physics result.

The first complete trace exposed eager current-flux diagnostics, then momentum
line solves. Fusing the diagnostics and replacing momentum's line
preconditioner with diagonal scaling reduced the full-rung warm medians from
21.14 to 3.09 seconds on one GPU and from 11.98 to 3.19 seconds on two. The
diagonal solve passes dense-reference, implicit-gradient, restart, placement,
and device-equivalence gates and removes all momentum tridiagonal events.

The new complete `128 x 67 x 67` trace assigns 66.0% of named solver wall time
to mixed pressure projection, 28.5% to electric potential, and 5.5% to
momentum. Projection and electric device activity are each about 60%
tridiagonal/PCR work; device-activity shares include overlapping devices and
are not wall-time shares. Seven-repeat one-GPU confirmations have 6.0–12.5% CV
because unrelated processes share both cards, so the initial 0.968x fixed-grid
ratio is not promoted as strong scaling. Doubling the axial extent gives a
stable 1.125x speedup, below the predeclared 1.2x promotion threshold; larger
blind rungs therefore stop. The compact profile record is
`benchmarks/results/b2-gpu-profile-20260715.json`.

The same record now exposes mixed-pressure PCG work directly. The frozen small
phase takes 33 and 35 iterations on both one and two GPUs, for 72 source-level
reduction stages over two updates. Paired diagnostic/control timing changes
sign across repeats while shared-host CV reaches 27%; retaining five SOLVAX
scalars is therefore within noise, not a performance claim.

The isolated representative trace is now complete. At `128 x 67 x 67`, mixed
projection converges in 204 and 207 iterations and takes 0.554 and 0.567
seconds. Fixed-coefficient tridiagonal kernels occupy 75% of normalized device
activity; all-reduce, all-gather, and halo kernels together occupy only
8.8--9.2%. The same raw trace assigns electric solves of 95 and 82 iterations:
they take 0.291 and 0.247 seconds, with 67% tridiagonal and 5.6--6.6%
collective activity. Named kernel counts agree on both GPUs, and the trace has
228,296 events versus its four-million-event cap. Activity sums concurrent GPU
streams and is not wall time.

Small paired probes also reject one transverse projection line, SOLVAX Jacobi
plus axial mean, a relaxed projection tolerance, and a mixed-boundary DCT-IV
coarse correction. The first two are slower; the tolerance change gains only
1.0% outside the frozen numerical contract. The coarse correction passes its
dense, symmetry, gradient, flow, and sharding gates, but gains only 0.47% and
misses the strict restart-state tolerance. Combining both transverse line
systems into one released-SOLVAX batch is algebraically and restart exact, but
is 1.0% slower than its same-window control. The accepted source is unchanged;
neither rejected candidate advanced to a full trace or larger grid.
Removing only the axial-mean block is also rejected: pressure iterations rise
from 33--35 to 180 and the tiny two-GPU solve becomes 6.1% slower. Projection
and electric communication experiments therefore stop on released SOLVAX;
their retained transverse line systems are the measured cost and the required
convergence mechanism.

Fusing the three eager `u x B` arrays into the existing electromagnetic JIT
kernels is also rejected. The exact-signature and restart screen regresses the
warm median from 1.840 to 2.000 seconds and lowers observed host RSS by only
1.2 MB, far below the frozen 16 MiB memory threshold. It does not advance to a
sustained ladder.

A separate released-SOLVAX `block_thomas_factor_fn` prototype generated B1
retained-modal factors without materializing the quadratic axial action tuple.
It is algebraically and JVP exact, but at `7 x 9 x 16` and `11 x 17 x 32` it is
27--29% slower cold and 38--39% slower warm. Device peak falls by at most 1.2%
while host peak rises about 4%, so it is rejected before a production B1 run.

The historical compact SOLVAX timing record remains the 0.8.1 measurement. A
matched JAX 0.8.0 replay measured warm-time ratios of 1.155 for an immediate
0.8.1 control and 1.184 for 0.8.2, so both miss the one-shot 1.10 threshold
despite passing every forward, gradient, transpose, residual, memory, device,
and Hartmann gate. SOLVAX PCG is unchanged between those releases; this is not
evidence of a 0.8.2 regression, and no refresh is planned. SOLVAX 0.8.4 now
owns the Anderson-weight API used by B2 restart schema 6.
The raw [0.8.1 control](https://github.com/uwplasma/LMX/releases/download/lmx-research-assets-v1/solvax-pcg-0.8.1-control-gpu.json)
(`ab54c5aa4a4787e1024d72d29ac5cd1c465c951bcaed82f179539cf75544fc7b`)
and [0.8.2 probe](https://github.com/uwplasma/LMX/releases/download/lmx-research-assets-v1/solvax-pcg-0.8.2-equivalence-gpu.json)
(`092fb878cd0182b0856b41afea9b90c99931a29b4b1976b4b5fd6c29272c8d36`)
remain outside Git.

On the Mac, `512` axial stations silently forced replication at six devices;
the placement gate now rejects that invalid point. The divisible `516`-station
diagnostic verifies 2/4/6 real shards and shows that four CPU devices are best
for this small operator workload. It is not a production-solver speedup claim.

The earlier schema-6 CPU precursor uses the real two-update B2 path on a fixed
`256 x 67 x 67` grid. A global correction on the restricted coarse grid removes
artificial Neumann interfaces at shard boundaries and restores electric PCG
counts from `159/129` and `193/157` to `109/87`. Warm medians are 15.159,
12.337, and 11.146 seconds on one/two/four forced JAX CPU devices. The 95%
speedup lower bounds are 1.209 and 1.319: two devices pass, but four devices
miss the frozen 1.40 speedup and 35% efficiency gates. It established
correctness but not scaling; the later callback-free 32-update ladder
supersedes its timing classification.

A paired `128 x 35 x 35` probe rejected removing the separate axial-mean
correction when the global modal basis is active: four-device time regressed
7.4% with unchanged iterations and bitwise-identical outputs. The next profile
targets the Neumann electric gauge reduction; no larger scaling rung is
authorized from the rejected variant.

The resulting four-device trace records one all-reduce, one coarse all-gather,
and two halo permutations per electric PCG iteration. Each contributes under
2% of the 744 ms profiled solve, and no distinct second gauge collective exists.
This misses the 3% trigger for an anchored-gauge prototype; the remaining local
gap is compute/scheduling dominated rather than a single communication defect.

Composing momentum and projection gains only 3.6%, raises RSS 15.4%, and fails
interface-current balance. Combining seven exact projection guards into one
host transfer gains 3.23% and lowers RSS 1.68%, but misses the 5% gate. Both
`128 x 35 x 35` candidates were reverted without a multi-minute rerun.
Bundling all successful-path B2 scalar diagnostics into one host transfer also
preserves exact physics, restart, history, and placement gates, but regresses
the clean four-device warm median from 1.971 to 2.011 seconds (2.06%); an
independent rerun is slower still. It is reverted without consuming a sustained
ladder. Small screens remain promotion filters, never scaling evidence.

The usual macOS single-thread XLA flags also fail as a core-control method:
default and flagged one-device profiles both use 11 HLO worker threads and take
1.000 and 0.995 seconds. A subsequent HLO-owned batching probe halved the two
isolated transverse SOLVAX Thomas traversals, but improved the full four-device
solve only 0.85% (`0.7437` to `0.7374` seconds). That fast path is rejected
before the large rung. These results remain forced-device sharding evidence,
not physical-core strong scaling.

The accepted 32-update evaluation at source `3339e95` gives
244.763/173.033/158.354 s medians and 1.415x/1.546x speedups; peak RSS is
4.15/4.33/4.48 GiB and the maximum warm CV is 2.318%. Timed trajectories are
callback-free, and the untimed exact-restart audit plus continuous/postflight
monitoring pass. This establishes Docker CPU-allocation scaling, not exact Apple
M4 P/E-core scaling.

A controlled current-source Docker/HLO profile isolates why four-device scaling
flattens. All numerical, restart, conservation, and placement checks pass, but
momentum contains twelve full `128 x 33 x 33 x 3` all-gathers (3.35 MB each),
six inside GMRES loops. Collectives occupy 70.8% of the captured projection
slice and 52.9% of momentum. Runtime and axial-size screens do not improve
efficiency, while a larger transverse section only hides some overhead. The
next bounded candidate therefore makes momentum halo and convection assembly
sharding-aware; another multi-minute ladder is allowed only after the candidate
reduces collective time by 25%, the affected phase by 15%, and two-update wall
time by 8% with unchanged physics and less than 5% RSS growth.

The first such candidate is rejected despite removing every full-vector gather
from GMRES. It improves wall time 8.42% and momentum 20.23%, but twelve one-plane
permutations replace the gathers and collective time falls only 20.79%, missing
the frozen 25% gate. The implementation is reverted. A successor must reuse one
halo across diffusion and convection and avoid the separate west-transport
exchange before another bounded profile is justified.

That fused successor is rejected too. Its bounded wall time improves 26.94%,
momentum 17.59%, and RSS grows only 2.23%, with exact numerics, but collective
time falls only 17.01%. The two remaining directional plane exchanges increase
permutation union from 7.83 to 18.23 ms. This small-case speedup is not promoted
to a sustained run. The next candidate must remove remaining setup gathers or
change axial operator/partition ownership, rather than rearranging the same
halo pair again.

A two-compact-gather draft stops before timing. Although it compiles and passes
a real four-shard functional solve, it adds 344 net solver lines by duplicating
five stencil owners. LMX rejects that maintenance cost. The topology can return
only after the existing gradient, limiter, stress, diffusion, and convection
primitives share precomputed axial neighbors in a concise refactor.

Superseded pilots and rejected ladders remain auditable in
`benchmarks/results/b2-schema6-cpu-scaling-20260716.json`.

The medium B2 tight solve converged across two restart-safe segments and ended
at residual `2.500e-5`, divergence `1.829e-6`, and charge residual `1.149e-4`.
The source-identical baseline, doubled-iteration, and confirmation-wall runs
then passed in 57.64, 34.94, and 57.10 seconds. Their tolerance and iteration
deltas are below `5.79e-4` of the frozen uncertainty; this closes medium-grid
numerical independence, not the experimental or three-mesh acceptance gate.
The fine-grid baseline remains outside acceptance. In that historical run, a region-preserving
transverse Galerkin correction is now added to the accepted line and axial-mean
preconditioners. Each shard diagonalizes its local Neumann axial block with a
DCT; one generalized transverse eigendecomposition serves every axial mode, so
the correction introduced no cross-shard FFT. The current schema-6 CPU path
instead uses the global restricted-grid correction described above. On the historical identical `202 x 149 x
149` checkpoint, electric PCG fell from 1,200/1,200 to 232/231 iterations and
matched warm time fell from 183.37 to 98.12 seconds (1.87x). It also beats the
previously accepted 109.18-second control. Residual histories agree within
`1.67e-16`, field norms are identical, and two-shard conservation gates pass.
The raw records are release assets; the compact accepted summary is
`benchmarks/results/b2-fine-fast-diagonalization-20260714.json`.
The first promoted continuation checkpointed 12 updates in 177.37 seconds and
reached `6.3606e-5`. Its post-transient slope is only `1.80e-8` per update, so
the 128-update ceiling projects to `6.152e-5`, not the `5e-5` fine gate. The
valid 115 MiB checkpoint is a release asset. This bounded stop moves the next
optimization target from the inner electric solve to the outer fixed-point
map; brute-force continuation is not accepted evidence. Matched four-update
probes rejected Anderson because the residual grew to `7.2240e-5`. They showed
that the previous Aitken minimum of 0.05 caused the plateau: minimum 1.0 exactly
matched stable Picard steps, while minimum 2.0 doubled the sustained reduction.
An eight-update restart-safe confirmation decreased monotonically to
`5.9578e-5` in 133.93 seconds with all balance gates passing, so the frozen B2
specification now uses a minimum relaxation of 2.0. The resumed production
baseline then reached `4.8760e-5` in 28 updates and 457.37 seconds. Its
doubled-iteration and confirmation-wall variants each converged in three
updates; their wall observables agree to `1.76e-13` relative.

The tighter `2.5e-5` variant reached a durable 48-update checkpoint and latest
recorded residual `3.4770e-5`. A dynamic Aitken probe was monotone but improved
only about `1.53e-8` per update; settled relaxation 3 and 4 oscillated. Scalar
tuning is therefore bounded and rejected. Componentwise Aitken was also
rejected after the residual jumped to `5.18e-3`. The 115 MiB restart and raw
records are release assets.

The fine curve has weighted RMS `1.389`, weighted maximum `4.218`, and
integrated pressure error `0.251`, missing the frozen ALEX limits `1.0`, `2.0`,
and `0.10`. A directional comparison to the accepted medium curve changes by
`0.0319`, above the `0.02` mesh gate; the records have different source
fingerprints, so this is diagnosis rather than formal acceptance. The released
SOLVAX Anderson-weight API and bounded schema-6 depth-two field/flux path pass
CPU and 1/2-GPU topology and exact-replay gates, but the current cold outcome
gate rejects it for B2 acceleration. The bounded fallback passes safety but
ends 0.22% worse than fixed relaxation two, so no unused API is added.
The residual-spectrum audit attributes the failure to objective mismatch, not
an untried conditioning threshold: electric potential contributes at least
98.254% of the shared norm while acceptance measures velocity convergence.
Production-mesh FreeMHD,
observable/model normalization, fine numerical independence, and experimental
acceptance remain blocked until the current coarse formulation converges for
the correct reason.
Enabling an axial line block across the sharded dimension was rejected: the
same two updates took 209.75 rather than 109.18 seconds, with unchanged 1,200-
iteration electric solves and equivalent residuals.
Removing the shard-local axial coarse correction was also rejected: it reduced
only 1--2 iterations and raised wall time to 155.88 seconds.
A multiplicative line-then-coarse correction saved only 3--4 iterations and
took 135.59 seconds, so it was rejected as well. A rediscretized balanced
hierarchy also regressed the bounded CPU probe; exact Galerkin fast
diagonalization supplied the accuracy and cost reduction needed for promotion.

The B1 pressure path first screens one 24-iteration GMRES cycle against the
actual mean-free divergence and normalized fixed-flow tolerance. Passing states
stop there; failing states continue through the original tight GMRES solve.
The large gate used `21/216` iterations initially and `216/216` after restart,
with maximum divergence below `6.1e-5` and charge residual below `5.1e-9`.

Run the checkpointed B1 acceptance path with:

```bash
python scripts/run_benchmark_b_independence.py \
  --cases B1-fringing-pipe --mesh-level coarse --variants baseline --resume
```

Run one frozen B2 solve across both visible GPUs with:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false \
python scripts/run_benchmark_b_independence.py \
  --cases B2-fringing-square --mesh-level medium --variants baseline \
  --spatial-devices 2 --prolong-restart \
  --variant-restart baseline=/release-assets/b2-coarse-102.npz
```

The sharded builder rounds axial minima `101/151/201` to `102/152/202`, which
still satisfies the frozen minimum-resolution contract. Cross-section grids,
physics, and tolerances are unchanged. The runner refuses to combine spatial
sharding with the separate one-process-per-GPU campaign mode and fails if the
result does not actually have the requested number of addressable JAX shards.
`--prolong-restart` is explicit: it trilinearly interpolates only B2 initial
fields in physical coordinates, records source/target shapes and method, and
then subjects the solved state to the normal projection and physics gates. It
does not present the interpolated state as a mesh result. The real coarse-to-
medium initialization (`102 x 77 x 77` to `152 x 113 x 113`) takes 1.70 seconds
and produces finite fields.
The exact pushed `8f003d0` production gate reported two shards on `cuda:0` and
`cuda:1`, converged three updates in 38.81 seconds including compilation, and
kept divergence below `5.8e-7` and charge residual below `2.9e-5`.

Frozen B1 cases use the compatible steady and retained-modal implementations
directly and record both choices in the result. The superseded environment
switches were removed after small, medium, large, and restart parity passed.
Completed coarse, medium, and fine directories can be combined with
`--acceptance-mesh LEVEL=DIR`; exact-case FreeMHD evidence is supplied with
`--freemhd-record CASE=PATH`. Assembly rejects missing levels, mixed source
fingerprints, malformed curves, and unchecked external evidence without
starting another expensive solve.

Authoritative records:

- `benchmarks/results/b2-schema6-cpu-scaling-20260716.json`
- `benchmarks/results/b2-gpu-scaling-calibration-20260715.json`
- `benchmarks/results/b2-gpu-profile-20260715.json`
- `benchmarks/results/b2-gpu-device-equivalence-20260715.json`
- `benchmarks/results/b1-retained-modal-blocks-20260713.json`
- `benchmarks/results/portable-gate-20260716.json`

## Run a bounded benchmark

The user-facing example works on any available JAX backend:

```bash
python examples/strong_scaling_demo.py --help
```

For controlled workers and machine-readable records:

```bash
python scripts/run_strong_scaling_worker.py --help
```

The matched-B2 worker keeps two updates as its fast debug/CI default. Such runs
can verify numerics but are never scaling evidence. A sustained record must use
at least one million global cells and 32 million cell-updates, predeclare a
two-minute minimum, run one cold plus at least three warm trajectories per
rung, and require every warm trajectory to meet it; direct and midpoint-restart
paths must still finish identically. The sustained preset uses
256 × 67 × 67, 32 CPU updates, 96 GPU updates, four repeats, a 120-second warm
minimum, and an 1800-second worker ceiling:

```bash
python examples/strong_scaling_demo.py --benchmark-kind matched_b2_smoke \
  --sustained --remote-host office \
  --source-commit "$(git rev-parse HEAD)" \
  --remote-python /path/to/lmx-gpu-venv/bin/python
```

Sustained mode rereads a checksummed 60-second admission JSON immediately before
each worker. Top-level `backend`, `host`, and `source_commit` must match the
requested run. The selected rung must match `num_devices`, be verified, and set
`admission_ended_unix_seconds`; at launch it must satisfy
`-5 <= now - admission_end <= 120` seconds. CPU `host` is `os.uname().nodename`;
GPU `host` is the exact `--remote-host` value. CPU rungs also bind the exact
2/4/8-entry affinity allocation and require at most 5% full-window utilization
on every selected CPU. GPU rungs bind unique visible UUID/PCI pairs, zero
compute contexts, no host process above 25% CPU, and at most 5% utilization.
Missing, stale, or mismatched evidence stops before the multi-minute solve.

Sustained mode collects and atomically writes a new 60-second admission before
every rung; the evidence options only override its output paths. Non-sustained
smoke runs may instead read a supplied static record. `--python` selects the
local CPU interpreter, while `--remote-python` selects and preflights the remote
GPU interpreter.

The CPU lane still runs inside a Linux Docker allocation exposing eight CPUs;
the example applies nested `taskset` masks that provide
2/4/8 guest CPUs for 1/2/4 JAX shards and establish CPU-allocation scaling, not
exact M4 P/E-core placement. Its built-in collector writes distinct per-rung
evidence after each 60-second clean observation. New timing evidence also needs
a checksummed continuous monitor through every rung and a clean postflight; the
CPU monitor covers the Docker guest allocation, not the macOS host, while the
GPU monitor covers the remote host. Sustained runs create an
ignored `<backend>_N.monitor.jsonl` automatically; keep promoted raw traces as
release artifacts. Each worker record must carry compact schema-2
`resource_monitoring` evidence: matching backend/device count, a raw SHA-256,
source fingerprint, bracketed worker timestamps, sample period and maximum gap
at most two seconds, full cold-plus-warm coverage, at least 15 postflight
seconds, and zero violations. Missing or malformed monitoring leaves timings
visible as candidates but blocks promotion.
The protocol rejected a cleanly admitted 24-update ladder after runtime
swapout, unstable samples, and one sub-120-second sample. The subsequent
unchanged 32-update ladder passes every promotion gate. This manual lane stays
outside portable tests.

The solver-faithful example requires a validated restart matching each timed
grid; it fails before launching workers when one is missing:

```bash
python examples/strong_scaling_demo.py --benchmark-kind extruded_solve \
  --cpu-restart artifacts/restarts/b2_cpu.npz
```

Use `--gpu-restart` as well when `--remote-host` is set. Cold-start convergence
belongs outside the timed strong-scaling region.

Select a backend before importing JAX:

```bash
JAX_PLATFORMS=cpu python examples/strong_scaling_demo.py
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0,1 python examples/strong_scaling_demo.py
```

Production B2 runs accept `num_devices=N`. LMX partitions the axial dimension
with named JAX sharding, validates the global shard count, and compares velocity,
potential, current, conservation, and steady-state signatures against the
one-device baseline. Other extruded solvers remain single-device until they
pass the same gates.

## Measurement contract

A publishable strong-scaling record contains:

- fixed global grid, physics, tolerance, precision, and update count;
- one cold plus at least three warm trajectories per rung, with a predeclared
  two-minute minimum met by every warm trajectory;
- backend, device model, device count, JAX version, and source fingerprint;
- cold time, warm time, throughput, speedup, and parallel efficiency;
- compilation separated from repeated execution;
- shard-placement evidence, peak host RSS, and per-device accelerator memory
  when the backend exposes it;
- numerical equivalence, conservation, and convergence results, including B2
  pressure-PCG iteration counts and all-solves-converged status;
- one-device baseline measured in the same environment.

The summary grants a sustained claim only to a complete CPU 1/2/4-shard or
GPU 1/2-device group whose provenance, memory, placement, numerics, and
affinity or idle-host evidence all pass. Individual long runs remain visible as
candidates but cannot set the sustained plot title or reported sustained
speedup.

Speedup and efficiency are

```text
S(N) = T(1) / T(N)
E(N) = S(N) / N
```

Small grids often slow down when sharded because dispatch and collectives
dominate. Use representative production grids and report negative results.

## CPU parallelism

JAX/XLA already uses the host CPU pool inside compiled kernels. Running one
solver process per core usually oversubscribes the machine. LMX uses process
parallelism for independent tests or validation cases, and array sharding for
a single sufficiently large solve. The full test driver chooses a bounded
worker count and enforces its wall-clock budget.

## GPU memory and restarts

Long runs should disable aggressive preallocation when several processes share
a host:

```bash
XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda python -m lmx CASE.toml
```

Use the restart controls in the case file for expensive extruded runs. Restart
files carry source and input metadata so incompatible state is rejected rather
than silently reused.

## Next performance work

1. Repeat the passing 96-update one/two-GPU lane after the host passes the idle gate.
2. Keep fixed relaxation two; do not tune the rejected shared-norm Anderson
   mechanisms or add a SOLVAX coefficient option.
3. Reopen acceleration only after a solver-free block-balanced,
   physics-aligned inner-product rationale and predeclared globalization rule;
   the unchanged cold gate must pass before steps 29 and 96.
4. Close canonical coarse/medium/fine and experimental-observable acceptance.

See [Testing](testing.md) for the portable gate and [Benchmark matrix](benchmark_matrix.md)
for physics promotion criteria.
