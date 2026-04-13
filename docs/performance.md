# Performance and Scaling

LMX keeps the fully developed duct solver on a JAX-native operator path and now
ships a dedicated strong-scaling benchmark for the dominant stencil/linear-solve
kernel. The goal is practical research throughput:

- keep the routine validation lane under five minutes
- preserve a differentiable fixed-iteration lane
- provide explicit CPU and multi-GPU scaling evidence for publication figures

## What is benchmarked

The shipped strong-scaling workflow benchmarks a fixed-iteration Poisson/Jacobi
stencil solve on the same global cross-section while increasing the number of
devices:

- local CPU runs use logical CPU devices via `XLA_FLAGS`
- multi-GPU runs use JAX sharding over the first mesh dimension

This is intended as a kernel-scaling benchmark for the current `1.0` solver
family. It is not yet a full domain-decomposed 3D fringing-field solver.

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
- publication-style `strong_scaling.png`
- publication-style `strong_scaling.pdf`

## Backend selection for CLI runs

Standard `lmx input.toml` runs inherit the active JAX backend from the shell
environment. Typical launch patterns are:

CPU:

```bash
JAX_PLATFORMS=cpu OMP_NUM_THREADS=8 lmx examples/hartmann_case.toml
```

Single GPU:

```bash
JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=0 lmx examples/hunt_case.toml
```

Those commands select the execution device for the normal CLI solver run. The
publication strong-scaling figures use `examples/strong_scaling_demo.py`,
because that benchmark intentionally exercises the sharded stencil kernel across
multiple CPU or GPU devices.

## Publication artifact

The current `1.0` publication artifact is committed under
`docs/_static/generated/strong_scaling.png` and is generated from:

- a local CPU sweep on a fixed `1024 x 1024` cross-section with `96` Jacobi iterations
- a remote GPU sweep on a fixed `2048 x 2048` cross-section with the same iteration count

![LMX strong scaling](_static/generated/strong_scaling.png)

Observed warm-runtime points from that artifact:

- CPU:
  - `1` device: `0.0898 s`
  - `2` devices: `0.1548 s`
  - `4` devices: `0.0563 s`
  - `8` devices: `0.0549 s`
- GPU:
  - `1` GPU: `0.0524 s`
  - `2` GPUs: `0.0392 s`

The CPU sweep is intentionally reported as measured rather than idealized:
logical CPU sharding on a single workstation is sensitive to host-thread
contention, so the figure is useful as a reproducible performance baseline, not
as a claim of perfect monotone scaling. The remote GPU path uses the highest
available single GPU index for the one-device baseline so the measurement is
not distorted by workstation display load.

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

Those numbers are not the publication artifact; they are the current
post-`1.0` smoke validation that the remote-GPU orchestration still works on a
live two-GPU host.

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
