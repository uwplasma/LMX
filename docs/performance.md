# Performance and Scaling

LMX keeps the fully developed duct solver on a JAX-native operator path and
benchmarks the production ALEX B2 solve directly. The goal is practical
research throughput:

- keep the routine validation lane under five minutes
- preserve a differentiable fixed-iteration lane
- provide explicit CPU and multi-GPU scaling evidence for larger studies

## Performance closeout gates

LMX should separate user-facing runtime feedback from benchmark claims.

Required user-facing behavior:

- progress output with current step, simulated time, residuals, flow rate,
  current diagnostics, elapsed time, and estimated remaining time
- explicit grid sizes, write stride, and output paths in summaries
- memory-relevant problem dimensions in benchmark JSON files
- compile time and warm runtime reported separately for JAX paths

Required benchmark behavior:

- CPU scaling tied to the executable `extruded_inductionless` projection
  arithmetic or a documented higher-intensity surrogate
- GPU scaling run on problem sizes large enough that sharding overhead does not
  dominate
- profiler traces before promoting a new strong-scaling figure
- memory-allocation and host-to-device placement checks for large arrays
- persistent JAX compilation cache on long artifact workflows where supported

Archived surrogate figures remain reproducibility evidence, not a current
performance claim. A paper-facing panel must use the production solve and a
profile that identifies memory bandwidth, communication, compilation, and
kernel-launch costs.

## What is benchmarked

The scaling workflow has two explicit benchmark kinds:

- `extruded_solve`
  - runs the actual rectangular `solve_extruded_inductionless(...)` path
  - applies named axial sharding to every production ALEX B2 3-D field
  - fails if a requested multi-device solve does not return that shard count
  - records grid size, estimated array memory, warm cell-updates per second,
    and optional JAX trace directory
  - is the only solver-faithful timing gate for release candidates
- `extruded3d`
  - runs a sharded fixed-iteration MHD operator surrogate
  - keeps explicit multi-device sharding over the fixed global grid
  - remains useful for isolating CPU/GPU stencil and communication behavior

This distinction matters for publication claims. `extruded_solve` is the
research-code runtime evidence because it uses the same projection loop as the
examples. `extruded3d` is the scaling-algorithm evidence because it exposes the
current multi-device sharding behavior on a denser fixed operator.

The older public `stencil2d` microbenchmark has been removed. It duplicated
sharding/reporting machinery without exercising the 3-D inductionless solver;
the retained surrogate isolates operator costs, while `extruded_solve` is the
only release-facing performance gate.

### Production-sharding checkpoint (12 July 2026)

The first end-to-end ALEX B2 checkpoint uses a deliberately small
`24 x 24 x 24` total grid and two outer steps. One RTX A4000 took `29.10 s`; two
A4000s took `40.72 s`, so this overhead-dominated probe is a measured scaling
miss, not a speedup claim. It is nevertheless a correctness gate: relative
one/two-GPU differences in the velocity, potential, and current L2 signatures
were `2.4e-9`, `7.3e-7`, and `1.4e-6`, respectively, and every returned 3-D
field retained two axial shards. The production wrappers now reuse a
process-stable device mesh and cached JIT kernels, so repeated two-GPU solves
remain physics-identical. Warm time is `8.79 s` on one GPU and `21.43 s` on two
for the small case; a `48 x 36 x 36` case measures `10.48 s` versus `38.28 s`.
Cross-section-only pressure line blocks did not improve that result and were
rejected. The coarse-scale `102 x 77 x 77` footprint runs two outer steps in
`34.08 s` warm. The earlier `15.7 GiB` observation mostly measured JAX's
default memory preallocation; production campaigns now disable preallocation
unless the user overrides it. Global PCG reductions and halo traffic remain
the multi-device bottlenecks. Independent variants therefore occupy the two
GPUs concurrently, one process per GPU, in restart-aware waves. Those workers
share a source-fingerprinted persistent compilation cache under the system
temporary directory, while an explicit `JAX_COMPILATION_CACHE_DIR` override is
preserved.

SOLVAX 0.7.0 adds an opt-in algebraically equivalent single-reduction PCG
recurrence, and sharded B2 duct solves now use it for momentum, projection, and
electric solves. On two A4000s its compiled per-step scalar products form one
tuple all-reduce. It improves one/two-GPU numerical agreement: relative L2
signature differences are below `1.1e-8` on the `48 x 36 x 36` probe. It does
not yet improve wall time: the same probe measures `10.23 s` on one GPU and
`38.33 s` on two. This is a correctness and synchronization-count improvement,
not a strong-scaling claim.

On the Mac CPU backend, two forced virtual devices are also slower than JAX's
normal single CPU device (`5.03 s` versus `3.19 s` warm on the small probe).
Normal CPU execution already uses threaded kernels, so virtual-device sharding
is not the recommended local acceleration path.

## Run the benchmark

Local CPU only:

```bash
python examples/strong_scaling_demo.py --output artifacts/examples/strong_scaling_cpu
```

Solver-faithful rectangular `extruded_inductionless` timing with a JAX trace:

```bash
python examples/strong_scaling_demo.py \
  --benchmark-kind extruded_solve \
  --cpu-counts 1,2,4 \
  --repeats 2 \
  --profile \
  --output artifacts/examples/extruded_solve_scaling
```

Local CPU plus a remote GPU host reachable over SSH:

```bash
python examples/strong_scaling_demo.py \
  --remote-host office \
  --output artifacts/examples/strong_scaling_full
```

The example writes:

- raw JSON timing records
- optional JAX trace directories under `profiles/`
- `strong_scaling_summary.json`
- `strong_scaling_diagnostics.json`
- `strong_scaling_table.csv`
- `strong_scaling.png`
- `strong_scaling.pdf`

`strong_scaling_table.csv` is the compact artifact intended for CI/release
review. It reports fixed-problem speedup, parallel efficiency, warm
Mcell-updates/s, estimated array memory, profiler-trace availability, and
whether each row came from the actual `solve_extruded_inductionless` path. The
JSON diagnostics carry the same rows plus summary counters, so release gates can
reject surrogate-only timing evidence when a solver-faithful scaling claim is
being made.

## Backend selection for CLI runs

Standard `lmx input.toml` runs inherit the active JAX backend from the shell
environment. Typical launch patterns are:

CPU:

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 lmx examples/hartmann_case.toml
XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu OMP_NUM_THREADS=1 lmx examples/hartmann_case.toml
```

Single GPU:

```bash
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 lmx cases/ducts/hunt_case.toml
```

Those commands select the execution device for the normal CLI solver run. Use
`examples/strong_scaling_demo.py --benchmark-kind extruded_solve` for a
production-path device-count sweep. For remote GPU runs on the `office` host,
the same pattern works over SSH:

```bash
ssh office 'cd /home/rjorge/tmp/lmx_scaling_repo && PYTHONPATH=/home/rjorge/tmp/lmx_scaling_repo CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda python3 -m lmx cases/ducts/hunt_case.toml'
```

## Current conclusion

The production `extruded_solve` checkpoints above supersede the archived
surrogate curve as the current scaling result. They prove device placement and
physics equivalence but miss the strong-scaling target on two A4000s. Forced
virtual-device sharding also slows the Mac CPU path, whose normal JAX kernels
already use host threads. Until a production solve meets the frozen efficiency
gate, the supported throughput strategy is one solve per GPU and normal
single-device JAX execution on the Mac. Treat shard placement alone as an
implementation check, never as a physics or performance result.

The `auto` path resolves to released SOLVAX 0.5.1 PCG. Reproduce its
native/SOLVAX forward, implicit-gradient, independent-transpose, compile,
warm-time, and compiler-memory comparison on CPU with:

```bash
JAX_ENABLE_X64=true uv run --locked --extra dev \
  python scripts/benchmark_solvax_pcg_backend.py --expected-backend cpu
```

The tracked CPU and RTX A4000 x64 records are
`benchmarks/results/solvax-pcg-equivalence-cpu.json` and
`benchmarks/results/solvax-pcg-equivalence-gpu.json`. Both pass. On the recorded
GPU, SOLVAX has field relative difference `1.54e-12`, implicit-gradient error
`1.13e-16`, transpose residual `2.54e-13`, warm-time ratio `0.230`, and
temporary-memory ratio `1.000` relative to native CG. The four-level Ha=20
FreeMHD ladder and all eight high-Ha Table I rows pass at the same promoted
solver-core fingerprint in `benchmarks/results/solvax-pcg-acceptance.json`.

## Recent compatibility and platform validation

The current tree has also been validated on:

- local Python `3.13` with JAX `0.9.2`
- remote Python `3.10.12` on the `office` host with JAX `0.6.2`

That remote run confirms both the broad JAX-version compatibility work and the
Python `3.10` TOML fallback path on a real two-GPU machine.

Recent small validation run on `office`:

- CPU `256 x 256`, `32` iterations:
  - `1` device warm runtime: `1.46e-3 s`
  - `2` devices warm runtime: `2.00e-3 s`
- GPU `512 x 512`, `32` iterations:
  - `1` GPU warm runtime: `6.11e-4 s`
  - `2` GPUs warm runtime: `3.91e-3 s`

Those numbers are not the main scaling artifact; they are a smoke validation
that the remote-GPU orchestration still works on a
live two-GPU host.

Recent matched one-device comparison used to confirm the expected CPU/GPU
direction of travel for the current tree:

- local CPU `512 x 512`, `32` iterations:
  - `warm_seconds ≈ 4.31e-3`
- office GPU `512 x 512`, `32` iterations:
  - `warm_seconds ≈ 6.65e-4`

At that problem size the remote single-GPU warm runtime is faster than the
local one-device CPU warm runtime. Small two-GPU runs are still dominated by
overhead, so multi-GPU strong scaling should be judged on the larger committed
main artifact, not on these smoke runs.

## Design notes

- `lmx/scaling.py`
  - sharded stencil benchmark kernel
- `scripts/run_strong_scaling_worker.py`
  - single-environment worker used by the example/orchestrator
- `examples/strong_scaling_demo.py`
  - explicit user-facing orchestration for CPU and remote GPU runs
- `lmx/plotting.py`
  - scaling plot generation

## References

- [JAX distributed arrays and automatic parallelization](https://docs.jax.dev/en/latest/notebooks/Distributed_arrays_and_automatic_parallelization.html)
- [JAX sharded computation guide](https://docs.jax.dev/en/latest/sharded-computation.html)
