# Performance and Scaling

LMX keeps the fully developed duct solver on a JAX-native operator path and
includes a dedicated strong-scaling benchmark for a dense structured-grid
inductionless MHD operator. The goal is practical research throughput:

- keep the routine validation lane under five minutes
- preserve a differentiable fixed-iteration lane
- provide explicit CPU and multi-GPU scaling evidence for larger studies

## What is benchmarked

The strong-scaling workflow benchmarks a fixed-iteration coupled operator on
the same global cross-section while increasing the number of devices:

- local CPU runs use logical CPU devices via `XLA_FLAGS`
- multi-GPU runs use JAX sharding over the first mesh dimension

The benchmark applies repeated viscous, electric-potential, and Lorentz-like
updates on a structured cross-section. It is still a kernel benchmark rather
than a fully domain-decomposed 3D fringing solver, but it is closer to the
arithmetic mix of the current solver family than the earlier Jacobi-only path.

## Run the benchmark

Local CPU only:

```bash
python examples/strong_scaling_demo.py --output artifacts/examples/strong_scaling_cpu
```

Local CPU plus a remote GPU host reachable over SSH:

```bash
python examples/strong_scaling_demo.py \
  --remote-host office \
  --output artifacts/examples/strong_scaling_full
```

The example writes:

- raw JSON timing records
- `strong_scaling_summary.json`
- `strong_scaling.png`
- `strong_scaling.pdf`

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

- a local CPU sweep on a fixed `4096 x 4096` cross-section with `1024`
  operator iterations
- a remote GPU sweep on a fixed `10240 x 10240` cross-section with `4096`
  operator iterations

![LMX strong scaling](_static/generated/strong_scaling.png)

The figure shows warm runtime only. First-run compile / JIT
overhead is still stored in the JSON summary, but it is no longer plotted in
the main scaling figure because it dominated the left panel without helping
the actual strong-scaling interpretation.

Observed warm-runtime points from that artifact:

- CPU:
  - `1` device: `56.9564 s`
  - `2` devices: `45.8947 s`
  - `4` devices: `62.8218 s`
  - `8` devices: `62.5103 s`
- GPU:
  - `1` GPU: `58.1897 s`
  - `2` GPUs: `31.2451 s`

The CPU sweep is reported as measured rather than idealized. On this
workstation, the denser operator reaches its best warm runtime at `2` logical
CPU devices and then flattens, which is consistent with a memory-bandwidth
limit on the host path. The remote GPU path shows about `1.86x` speedup from
`1` to `2` GPUs on the larger fixed problem.

Recent local profiling confirms that the current CPU benchmark is still the
wrong place to claim a final CPU strong-scaling result. A JAX trace collected
for the `2`-device CPU benchmark (`/tmp/lmx_cpu_scaling_profile`) shows the
current path is dominated by a memory-bound sharded stencil on forced logical
host devices. The next CPU scaling benchmark should therefore move closer to
the actual `extruded_inductionless` operator path or a higher-intensity 3D
kernel rather than relying on the present host-device sharding curve alone.

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
