# Performance and scaling

LMX runs through JAX on CPUs and GPUs. Performance claims are accepted only for
the real solver path with identical numerical results; visibility of multiple
devices alone is not evidence of parallel execution.

## Current evidence

| Path | Hardware and grid | Result | Interpretation |
|---|---|---|---|
| portable test gate | Apple M4, six workers | 767 pass, 8 skip, 95.28% branch coverage, 190.6 s | below the ten-minute budget |
| SOLVAX PCG equivalence | Apple M4 CPU and RTX A4000 GPU | 0.8.1 forward, gradient, transpose, memory, timing, and Hartmann gates pass | compact current record; raw evidence external |
| sharded 3D operator | Apple M4, `516 x 32 x 32` | 1.16x on 2 cores, 1.28x on 4, 0.93x on 6 | actual shard placement verified; surrogate only |
| B2 axial sharding | 2 x RTX A4000, `102 x 77 x 77` | 36.96 s on one GPU, 22.23 s on two | 1.66x speedup, 83.1% efficiency |
| B2 medium independence | 2 x RTX A4000, `152 x 113 x 113` | all four numerical variants pass; final confirmations take 34.94--57.64 s | tolerance, iteration, wall, steady, and conservation gates pass |
| B1 modal setup | RTX A4000, `11 x 17 x 32` | 57.85 s first, 10.63 s restart | accepted setup optimization |
| B1 large solve | RTX A4000, `21 x 24 x 64` | 270.42 s for two updates | pressure projection is 91.2% of runtime |
| B1 physical-pilot gate | RTX A4000, `21 x 24 x 64` | 669 iterations for solve plus restart vs 768 fixed ceiling | all four physical projections pass; shared-host wall time is not a speedup claim |

The B2 result passes the two-device target and exact observable-equivalence
gate. A general or four-device scaling claim remains open. The B1 timings do
not promote its experimental physics result.

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
The fine-grid baseline remains outside acceptance. An exact eight-update probe
from the 112-update checkpoint reached residual `6.451e-5` in 359.01 seconds,
but every electric solve still used both 600-iteration PCG stages and the
initial Aitken jump returned to the slow asymptotic rate. The resulting 248 MiB
checkpoint is a checksummed release asset; a stronger fine-grid preconditioner,
not more brute-force updates, precedes another acceptance run.

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

- `benchmarks/results/gpu-strong-scaling-20260713.json`
- `benchmarks/results/b1-retained-modal-blocks-20260713.json`
- `benchmarks/results/portable-gate-20260714.json`

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
- numerical equivalence, conservation, and convergence results;
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

1. Re-measure the accepted B1 physical-pilot path on an otherwise idle GPU.
2. Close B1/B2 mesh and experimental-observable acceptance.
3. Add a four-GPU B2 point on suitable hardware.
4. Keep compilation-cache and memory measurements in every accelerator report.

See [Testing](testing.md) for the portable gate and [Benchmark matrix](benchmark_matrix.md)
for physics promotion criteria.
