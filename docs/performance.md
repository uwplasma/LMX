# Performance and scaling

LMX runs through JAX on CPUs and GPUs. Performance claims are accepted only for
the real solver path with identical numerical results; visibility of multiple
devices alone is not evidence of parallel execution.

![B2 two-update CPU and GPU scaling evidence](_static/strong_scaling.webp)

## Current evidence

| Path | Hardware and grid | Result | Interpretation |
|---|---|---|---|
| portable test gate | Apple M4, six workers | 837 pass, 8 skip, 95.58% combined line/branch coverage, 169.6 s | 14.3% slower under shared JAX contention; below five-minute target |
| B2 CPU smoke | Apple M4, `8 x 7 x 7`, 1/2/4 forced CPU devices | current-source pressure observable agrees within `5.93e-15`; closure and exact restart pass | production sharding correctness; too small for scaling claims |
| B2 pseudo-time gate | Apple M4, corrected warm `7 x 7 x 7`, canonical `Ha=2900`, `N=540` equations | 64x/32x/16x map-rate spread 0.0768%; raw updates halve; exact restart | 64x refrozen; versioned stopping threshold open |
| B2 physical-residual probe | Apple M4, `7 x 7 x 7` | residual 0.03870 at step 90; omitted reconstruction force is 0.0880 at step 4 | diagnostic has a plausible coarse split floor; never a stopping gate |
| B2 projection audit/fix | Apple M4, warm same-state `7 x 7 x 7` | pre-fix raw cell-update floor `3.69e-3`; corrected predictor-preserving projection removes axial floor | pre-fix trajectories invalid; focused physics/autodiff/restart gates pass |
| B2 stopping study | Apple M4, schema 5, restart segments to step 96 | `0.05` converges at 30 but fails all QoI limits; `0.005` reaches only `0.01502` | fail closed; accelerate before tighter-reference calibration |
| B2 relaxation probe | Apple M4, shared step-29 checkpoint, six updates | factors 5/6/8 gain at most 9.33% over factor 4 | below 15% gate; retain factor 2 |
| B2 fixed relaxation | canonical `min=max=2` | exact trajectory/restart; prior residual omitted | saves 18.46 MiB at `102 x 77 x 77` and two global dot products/update |
| B2 GPU smoke | 1/2 RTX A4000 GPUs, `8 x 7 x 7`, deterministic XLA | current repeats and restart are exact; pressure observable agrees within `1.02e-14`; closure and placement pass | production sharding correctness; too small for scaling claims |
| B2 scaling calibration | Apple M4, `128 x 31 x 31`, 1/2/4 forced CPU devices | 0.857/0.652/0.633 s; 1.31x/1.35x speedup; exact restart and device equivalence pass | historical pre-terminal-fix calibration; rerun before promotion |
| B2 GPU calibration | 1/2 RTX A4000 GPUs, `128 x 67 x 67`, default XLA | current medians 2.780/2.400 s; repeat, trace, restart, placement, and equivalence pass | 1.159x misses the 1.2x promotion gate; no scaling claim |
| B2 doubled-axial calibration | 1/2 RTX A4000 GPUs, `256 x 67 x 67`, default XLA | 8.47/7.53 s; 1.125x speedup; CV below 3.7% | historical pre-terminal-fix calibration below the 1.2x threshold |
| historical SOLVAX PCG equivalence | Apple M4 CPU and RTX A4000 GPU | 0.8.2 forward, gradient, transpose, memory, and Hartmann gates pass; one-shot GPU warm ratio is 1.184 | archival; PCG is unchanged and no refresh is planned |
| sharded 3D operator | Apple M4, `516 x 32 x 32` | 1.16x on 2 cores, 1.28x on 4, 0.93x on 6 | actual shard placement verified; surrogate only |
| B2 axial sharding | 2 x RTX A4000, `102 x 77 x 77` | 36.96 s on one GPU, 22.23 s on two | diagnostic 1.66x result for superseded formulation |
| legacy B2 medium independence | 2 x RTX A4000, `152 x 113 x 113` | all four numerical variants pass; final confirmations take 34.94--57.64 s | superseded no-inertia/stationwise-flow formulation |
| legacy B2 fine campaign | 2 x RTX A4000, `202 x 149 x 149` | baseline 457.37 s; iteration and wall confirmations 65.23/62.10 s | superseded formulation; tight-tolerance independence remains open |
| B1 modal setup | RTX A4000, `11 x 17 x 32` | 57.85 s first, 10.63 s restart | accepted setup optimization |
| B1 large solve | RTX A4000, `21 x 24 x 64` | 270.42 s for two updates | pressure projection is 91.2% of runtime |
| B1 physical-pilot gate | RTX A4000, `21 x 24 x 64` | 669 iterations for solve plus restart vs 768 fixed ceiling | all four physical projections pass; shared-host wall time is not a speedup claim |

The current CPU/GPU correctness and `128 x 67 x 67` calibration, plus historical larger results, are recorded in
`benchmarks/results/b2-{cpu,gpu}-device-equivalence-20260715.json` and
`benchmarks/results/b2-{cpu-strong-scaling,gpu-scaling-calibration}-20260715.json`.
The current trace measures 1.510x scaling across momentum, projection, and
electric phases. Reusing the existing host station payload for validation cuts
the two-GPU end-to-end median by 0.290 s and removes 12 source lines, but the
remaining post-map transfer tail leaves the result below 1.2x. The fixed-grid
ladder therefore stops without a larger run or production-speed claim.
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
evidence of a 0.8.2 regression, and no refresh is planned. The next upstream
work is the published Anderson-weight API required by B2 restart schema 6.
The raw [0.8.1 control](https://github.com/uwplasma/LMX/releases/download/lmx-research-assets-v1/solvax-pcg-0.8.1-control-gpu.json)
(`ab54c5aa4a4787e1024d72d29ac5cd1c465c951bcaed82f179539cf75544fc7b`)
and [0.8.2 probe](https://github.com/uwplasma/LMX/releases/download/lmx-research-assets-v1/solvax-pcg-0.8.2-equivalence-gpu.json)
(`092fb878cd0182b0856b41afea9b90c99931a29b4b1976b4b5fd6c29272c8d36`)
remain outside Git.

On the Mac, `512` axial stations silently forced replication at six devices;
the placement gate now rejects that invalid point. The divisible `516`-station
diagnostic verifies 2/4/6 real shards and shows that four CPU devices are best
for this small operator workload. It is not a production-solver speedup claim.

The medium B2 tight solve converged across two restart-safe segments and ended
at residual `2.500e-5`, divergence `1.829e-6`, and charge residual `1.149e-4`.
The source-identical baseline, doubled-iteration, and confirmation-wall runs
then passed in 57.64, 34.94, and 57.10 seconds. Their tolerance and iteration
deltas are below `5.79e-4` of the frozen uncertainty; this closes medium-grid
numerical independence, not the experimental or three-mesh acceptance gate.
The fine-grid baseline remains outside acceptance. A region-preserving
transverse Galerkin correction is now added to the accepted line and axial-mean
preconditioners. Each shard diagonalizes its local Neumann axial block with a
DCT; one generalized transverse eigendecomposition serves every axial mode, so
the correction introduces no cross-shard FFT. On the identical `202 x 149 x
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
fingerprints, so this is diagnosis rather than formal acceptance. Exact-case
FreeMHD and observable/model normalization now take priority over another outer
acceleration search. Fine numerical independence and experimental acceptance
remain open.
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

- `benchmarks/results/b2-cpu-strong-scaling-20260715.json`
- `benchmarks/results/b2-gpu-scaling-calibration-20260715.json`
- `benchmarks/results/b2-gpu-profile-20260715.json`
- `benchmarks/results/b2-{cpu,gpu}-device-equivalence-20260715.json`
- `benchmarks/results/b1-retained-modal-blocks-20260713.json`
- `benchmarks/results/portable-gate-20260715.json`

## Run a bounded benchmark

The user-facing example works on any available JAX backend:

```bash
python examples/strong_scaling_demo.py --help
```

For controlled workers and machine-readable records:

```bash
python scripts/run_strong_scaling_worker.py --help
```

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
- backend, device model, device count, JAX version, and source fingerprint;
- cold time, warm time, throughput, speedup, and parallel efficiency;
- compilation separated from repeated execution;
- shard-placement evidence and per-device memory estimate;
- numerical equivalence, conservation, and convergence results, including B2
  pressure-PCG iteration counts and all-solves-converged status;
- one-device baseline measured in the same environment.

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

1. Publish SOLVAX 0.8.4 from a reviewed merged SHA after explicit authorization.
2. Gate one sharding-aware depth-two B2 Anderson path with restart schema 6.
3. Close the canonical B2 mesh and experimental-observable acceptance ladder.
4. Revisit production scaling and four-GPU points only after that acceptance.

See [Testing](testing.md) for the portable gate and [Benchmark matrix](benchmark_matrix.md)
for physics promotion criteria.
