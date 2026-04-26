# Performance and Scaling

LMX keeps the fully developed duct solver on a JAX-native operator path and
includes a dedicated strong-scaling benchmark for a dense structured-grid
inductionless MHD operator. The goal is practical research throughput:

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

The current scaling figure is useful evidence, but it should not be the final
CPU-scaling claim for a paper. The next accepted CPU panel needs to be backed
by a profile explaining whether the limiting factor is memory bandwidth,
cross-device communication, compile overhead, or kernel launch overhead.

## What is benchmarked

The strong-scaling workflow now has two explicit benchmark kinds:

- `extruded_solve`
  - runs the actual rectangular `solve_extruded_inductionless(...)` path
  - records grid size, estimated array memory, warm cell-updates per second,
    and optional JAX trace directory
  - is the solver-faithful timing gate for release candidates
- `extruded3d`
  - runs the sharded fixed-iteration MHD operator surrogate used for the
    current strong-scaling figure
  - keeps explicit multi-device sharding over the fixed global grid
  - remains useful for CPU/GPU sharding studies while the production solver
    is still being moved toward explicit domain decomposition

This distinction matters for publication claims. `extruded_solve` is the
research-code runtime evidence because it uses the same projection loop as the
examples. `extruded3d` is the scaling-algorithm evidence because it exposes the
current multi-device sharding behavior on a denser fixed operator.

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
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 lmx examples/hunt_case.toml
```

Those commands select the execution device for the normal CLI solver run. The
committed strong-scaling figures use `examples/strong_scaling_demo.py`,
because that benchmark intentionally exercises the sharded stencil kernel across
multiple CPU or GPU devices. For remote GPU runs on the `office` host, the same
pattern works over SSH:

```bash
ssh office 'cd /home/rjorge/tmp/lmx_scaling_repo && PYTHONPATH=/home/rjorge/tmp/lmx_scaling_repo CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda python3 -m lmx examples/hunt_case.toml'
```

## Current artifact

The current scaling artifact is stored under
`docs/_static/generated/strong_scaling.png` and is generated from:

- a local CPU sweep on a fixed `8192 x 64 x 64` extruded operator with `256`
  iterations
- a remote GPU sweep on a fixed `6144 x 96 x 96` extruded operator with `4096`
  iterations

![LMX strong scaling](_static/generated/strong_scaling.png)

The figure shows warm runtime only. First-run compile / JIT
overhead is still stored in the JSON summary, but it is no longer plotted in
the main scaling figure because it dominated the left panel without helping
the actual strong-scaling interpretation.

Observed warm-runtime points from that artifact:

- CPU:
  - `1` device: `80.5495 s`
  - `2` devices: `74.6580 s`
  - `4` devices: `65.5038 s`
- GPU:
  - `1` GPU: `78.5812 s`
  - `2` GPUs: `46.8238 s`

The CPU sweep is reported as measured rather than idealized. On this
workstation, the denser operator improves through `4` logical CPU devices,
which is still consistent with a memory-bandwidth and communication limit on
the host path beyond that range. The remote GPU path shows about `1.68x`
speedup from `1` to `2` GPUs on the larger fixed problem.

This is also the point where the current JAX implementation strategy matters.
The official JAX guidance distinguishes automatic sharding from explicit
per-device kernels with [`jax.shard_map`](https://docs.jax.dev/en/latest/notebooks/shard_map.html),
and the profiling docs recommend validating those choices with Perfetto or
XProf traces rather than inferring bottlenecks from wall time alone. For LMX,
that means the next CPU-scaling step is not another presentation-only rerun of
the same host benchmark: it is a profiler-guided check on the current
`extruded3d` path and, if needed, an explicit `shard_map` / halo-exchange
version of the most communication-heavy stencil/projection kernels.

Recent local profiling confirms that the current CPU benchmark is still the
wrong place to claim a final CPU strong-scaling result. A JAX trace collected
for the `2`-device CPU benchmark (`/tmp/lmx_cpu_scaling_profile`) shows the
current path is dominated by a memory-bound sharded stencil on forced logical
host devices. The next CPU scaling benchmark should therefore move closer to
the executable `extruded_inductionless` projection loop or a higher-intensity
3D operator path rather than relying on the present host-device sharding curve
alone.

That solver-faithful entry point is now available as
`--benchmark-kind extruded_solve`. Until the production projection loop has
explicit multi-device domain decomposition, treat `extruded_solve` device-count
sweeps as backend/runtime diagnostics rather than final strong-scaling claims.

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
