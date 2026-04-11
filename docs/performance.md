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
  --remote-host <your_gpu_host> \
  --output artifacts/examples/strong_scaling_full
```

The example writes:

- raw JSON timing records
- `strong_scaling_summary.json`
- publication-style `strong_scaling.png`
- publication-style `strong_scaling.pdf`

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
