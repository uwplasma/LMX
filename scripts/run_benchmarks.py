#!/usr/bin/env python3
"""Time LMX on whatever device it is given, and write the result as JSON.

Three numbers are reported for every case and they are not interchangeable.
``compile_seconds`` is the first call, which includes tracing and XLA
compilation. ``warm_seconds`` is the best of several later calls, which is what
a long run pays. ``seconds_per_step`` divides the warm time by the number of
steps in the trajectory, which is the only number that compares across sizes.

The core trajectory is a compiled call; Q2D includes its reporting wrapper.
Trajectory-length ratios describe timing only: a fixed synchronization cost
per step also gives constant time per step. Use traces to locate synchronization.

``accepted`` is a correctness flag, not a timing one: a case that produced a
non-finite field or drifted off its divergence-free constraint is reported and
its timings are not to be quoted.

Every report also records what a timing needs before it can be quoted:

- JAX's ``jax_default_matmul_precision``;
- the ``NVIDIA_TF32_OVERRIDE`` and ``XLA_FLAGS`` environment values;
- the GPU driver and CUDA versions when a GPU is present;
- the host load average.

Left unset, the matmul precision lets Ampere GPUs run float32 contractions in
TensorFloat-32. On an RTX A4000 those were 3e-4 from float64, against 2.6-6.1e-7
at true float32. LMX pins ``'highest'`` unless the precision is already set, and
this script refuses to write a float32 GPU report at any level that allows
TensorFloat-32 (ADR 0005, D15).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import socket
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Levels at which JAX computes float32 dense contractions in float32 itself.
# Unset, ``'default'``, ``'high'`` and ``'tensorfloat32'`` allow TensorFloat-32;
# the bfloat16 and float8 presets are coarser still.
TRUE_FLOAT32_MATMUL = frozenset({"highest", "float32", "F32_F32_F32", "F64_F64_F64"})


def _commit() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    # Untracked files -- the results directory this run is about to write into --
    # do not make the measured code different from the commit it names.
    dirty = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def _gpu_versions() -> dict:
    """Return the NVIDIA driver version and the CUDA version it reports, or nulls."""
    versions = {"gpu_driver": None, "cuda_version": None}
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, check=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return versions
    for key, label in (("gpu_driver", "Driver Version"), ("cuda_version", "CUDA Version")):
        match = re.search(rf"{label}:\s*([0-9.]+)", result.stdout)
        if match:
            versions[key] = match.group(1)
    return versions


def _load_average() -> list[float] | None:
    """Return the 1, 5 and 15 minute load averages; a shared host times differently under load."""
    try:
        return [float(value) for value in os.getloadavg()]
    except (AttributeError, OSError):
        return None


def _environment(jax) -> dict:
    devices = jax.devices()
    on_gpu = devices[0].platform == "gpu"
    return {
        "host": socket.gethostname(),
        "platform": devices[0].platform,
        "device_kind": devices[0].device_kind,
        "device_count": len(devices),
        "cpu_count": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
        "python": platform.python_version(),
        "jax": jax.__version__,
        "x64": bool(jax.config.jax_enable_x64),
        "jax_default_matmul_precision": jax.config.jax_default_matmul_precision,
        "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        "xla_flags": os.environ.get("XLA_FLAGS"),
        **(_gpu_versions() if on_gpu else {"gpu_driver": None, "cuda_version": None}),
        "load_average": _load_average(),
        "lmx_commit": _commit(),
    }


def _unquotable_float32(environment: dict) -> str | None:
    """Say why a report must not be written, or return None when it may.

    Only float32 on a GPU is at stake: the matmul precision changes nothing on a
    CPU or for float64 arrays.
    """
    if environment.get("platform") != "gpu" or environment.get("x64") is True:
        return None
    if "jax_default_matmul_precision" not in environment:
        return "a float32 GPU report must record jax_default_matmul_precision"
    precision = environment["jax_default_matmul_precision"]
    if precision not in TRUE_FLOAT32_MATMUL:
        return (
            f"jax_default_matmul_precision={precision!r} allows TensorFloat-32, so these float32 GPU "
            "timings are not float32; unset JAX_DEFAULT_MATMUL_PRECISION or set it to 'highest'"
        )
    return None


def _timed(jax, call, repeats: int) -> tuple[float, float, object]:
    """Return the compile time, the best warm time, and the result."""
    started = time.perf_counter()
    result = jax.block_until_ready(call())
    compile_seconds = time.perf_counter() - started
    warm = float("inf")
    for _ in range(max(repeats, 1)):
        started = time.perf_counter()
        result = jax.block_until_ready(call())
        warm = min(warm, time.perf_counter() - started)
    return compile_seconds, warm, result


def _core3d_case(jax, cells: int, steps: int, repeats: int, precision: str = "state") -> dict:
    """One compiled trajectory of the staggered core on a cubic duct.

    ``precision="mixed"`` runs the fast-diagonal solves in float32 with float64
    correction (:mod:`lmx.poisson`); the environment block records the matmul precision it needs.
    """
    import jax.numpy as jnp
    import numpy as np

    from lmx.bc import NEUMANN, PERIODIC, BoundaryCondition
    from lmx.core3d import ChannelProblem, zero_velocity
    from lmx.grid import Grid, uniform_faces
    from lmx.ops import divergence
    from lmx.timeloop import advance

    periodic, wall = BoundaryCondition(PERIODIC), BoundaryCondition(NEUMANN)
    grid = Grid(
        uniform_faces(cells, 0.0, 1.0),
        uniform_faces(cells, -1.0, 1.0),
        uniform_faces(cells, -1.0, 1.0),
    )
    problem = ChannelProblem(
        grid=grid,
        conditions=(periodic, wall, wall),
        conductivity=1.0,
        magnetic_field=(0.0, 20.0, 0.0),
        forcing=(1.0, 0.0, 0.0),
        dt=2.0e-3,
        precision=precision,
    )
    factorization = problem.factorization()
    viscous = problem.viscous_factorizations()
    start = zero_velocity(problem)

    def run(velocity):
        trajectory = advance(problem, steps, velocity, factorization=factorization, viscous=viscous)
        return trajectory.velocity, trajectory.divergence_residual

    compiled = jax.jit(run)
    compile_seconds, warm, (velocity, residual) = _timed(jax, lambda: compiled(start), repeats)
    finite = bool(np.all([np.all(np.isfinite(np.asarray(field.data))) for field in velocity]))
    divergence_residual = float(jnp.max(jnp.abs(divergence(velocity).data)))
    # The projection removes the divergence to whatever the arithmetic can hold, and
    # what it can hold depends on both the precision and the number of cells the
    # round-off accumulates over. A fixed absolute bound would read a correct float32
    # run as a failure at 128 cubed, which is a statement about the bound.
    epsilon = float(np.finfo(np.asarray(velocity[0].data).dtype).eps)
    tolerance = 1.0e3 * epsilon * cells
    return {
        "case": "core3d_advance",
        "cells": cells**3,
        "shape": [cells, cells, cells],
        "steps": steps,
        "precision": precision,
        "compile_seconds": compile_seconds,
        "warm_seconds": warm,
        "seconds_per_step": warm / steps,
        "divergence_residual": divergence_residual,
        "divergence_tolerance": tolerance,
        "accepted": finite and divergence_residual < tolerance,
    }


def _q2d_case(jax, cells: int, steps: int, repeats: int) -> dict:
    """A quasi-two-dimensional decay, the spectral half of the workload."""
    import jax.numpy as jnp
    import numpy as np

    import lmx

    wavenumber = np.fft.fftfreq(cells, d=1.0 / cells)
    total = np.sqrt(wavenumber[:, None] ** 2 + wavenumber[None, :] ** 2)
    amplitude = np.where(total > 0, (total / 12.0) ** 4 * np.exp(-2.0 * (total / 12.0) ** 2), 0.0)
    phase = np.exp(2j * np.pi * np.asarray(jax.random.uniform(jax.random.PRNGKey(0), (cells, cells))))
    vorticity = np.fft.ifftn(amplitude * phase).real
    vorticity *= 4.0 / np.sqrt(np.mean(vorticity**2))
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32
    problem = lmx.Q2DProblem(
        jnp.asarray(vorticity, dtype=dtype),
        length=(2.0 * np.pi, 2.0 * np.pi),
        viscosity=2.0e-4,
        hartmann_friction=2.0e-2,
        dt=2.0e-3,
        steps=steps,
        history_stride=steps,
    )

    def run():
        result = lmx.solve(problem)
        return result.vorticity, result.status

    compile_seconds, warm, (result, status) = _timed(jax, run, repeats)
    finite = bool(np.all(np.isfinite(np.asarray(result))))
    return {
        "case": "q2d_evolve",
        "cells": cells**2,
        "shape": [cells, cells],
        "steps": steps,
        "compile_seconds": compile_seconds,
        "warm_seconds": warm,
        "seconds_per_step": warm / steps,
        "solver_status": status,
        "accepted": finite and status == "completed",
    }


def _host_sync_case(jax, cells: int, steps: int, repeats: int) -> dict:
    """Report length sensitivity, which cannot prove absence of host synchronization."""
    short = _core3d_case(jax, cells, steps, repeats)
    long = _core3d_case(jax, cells, 4 * steps, repeats)
    ratio = long["seconds_per_step"] / short["seconds_per_step"]
    return {
        "case": "core3d_host_sync",
        "cells": cells**3,
        "shape": [cells, cells, cells],
        "steps": [steps, 4 * steps],
        "seconds_per_step": [short["seconds_per_step"], long["seconds_per_step"]],
        "per_step_ratio": ratio,
        "host_sync_verified": False,
        "accepted": bool(short["accepted"] and long["accepted"]),
    }


def _shard_case(jax, cells: int, steps: int, repeats: int) -> dict:
    """Run the quasi-2D evolution on every device at once and see what it costs.

    The array is placed with a :class:`jax.sharding.NamedSharding` and nothing
    else is changed, so this measures what XLA's partitioner does unaided --
    the baseline any hand-written decomposition has to beat. Two numbers come
    out: whether the sharded answer is the same one (it must be, to round-off)
    and the strong-scaling efficiency, single-device time over device count
    times sharded time.

    The staggered three-dimensional core is not here. Its three components have
    lengths ``n+1`` on their own axis and ``n`` on the others, so no single axis
    divides evenly across all of them, and every solve contracts along all three
    axes. That needs a decomposition designed for it, not a placement.
    """
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as Spec

    import lmx

    devices = jax.devices()
    if len(devices) < 2:
        return {"case": "q2d_shard", "shape": [cells, cells], "accepted": False, "error": "one device"}
    wavenumber = np.fft.fftfreq(cells, d=1.0 / cells)
    total = np.sqrt(wavenumber[:, None] ** 2 + wavenumber[None, :] ** 2)
    amplitude = np.where(total > 0, (total / 12.0) ** 4 * np.exp(-2.0 * (total / 12.0) ** 2), 0.0)
    phase = np.exp(2j * np.pi * np.asarray(jax.random.uniform(jax.random.PRNGKey(0), (cells, cells))))
    vorticity = np.fft.ifftn(amplitude * phase).real
    vorticity *= 4.0 / np.sqrt(np.mean(vorticity**2))
    dtype = jnp.float64 if jax.config.jax_enable_x64 else jnp.float32

    def evolve(field):
        # `lmx.solve` compiles internally and reports a status, so it cannot be
        # wrapped in another `jit`; the placement of its input is what the
        # partitioner sees.
        problem = lmx.Q2DProblem(
            field,
            length=(2.0 * np.pi, 2.0 * np.pi),
            viscosity=2.0e-4,
            hartmann_friction=2.0e-2,
            dt=2.0e-3,
            steps=steps,
            history_stride=steps,
        )
        result = lmx.solve(problem)
        return result.vorticity, result.status

    single = jax.device_put(jnp.asarray(vorticity, dtype=dtype), devices[0])
    _, single_seconds, (reference, single_status) = _timed(jax, lambda: evolve(single), repeats)
    mesh = Mesh(np.array(devices), ("d",))
    placed = jax.device_put(jnp.asarray(vorticity, dtype=dtype), NamedSharding(mesh, Spec("d", None)))
    _, sharded_seconds, (result, sharded_status) = _timed(jax, lambda: evolve(placed), repeats)
    # Compare on the host: the two results live on different device sets, and
    # subtracting them on device is itself the error this case exists to avoid.
    single_values = np.asarray(jax.device_get(reference))
    sharded_values = np.asarray(jax.device_get(result))
    difference = float(np.max(np.abs(sharded_values - single_values)))
    scale = float(np.max(np.abs(single_values)))
    tolerance = 1.0e3 * float(np.finfo(single_values.dtype).eps) * scale
    return {
        "case": "q2d_shard",
        "cells": cells**2,
        "shape": [cells, cells],
        "steps": steps,
        "devices": len(devices),
        "scaling_scope": "shared_host_logical_devices"
        if devices[0].platform == "cpu"
        else "accelerator_devices",
        "single_seconds": single_seconds,
        "sharded_seconds": sharded_seconds,
        "efficiency": single_seconds / (len(devices) * sharded_seconds),
        "agreement": difference,
        "agreement_tolerance": tolerance,
        "solver_status": [single_status, sharded_status],
        "accepted": difference < tolerance and single_status == sharded_status == "completed",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", default="core3d,q2d,hostsync", help="Comma-separated case names.")
    parser.add_argument("--core3d-sizes", default="32,64,128", help="Cubic cell counts per side.")
    parser.add_argument("--q2d-sizes", default="256,512,1024", help="Square cell counts per side.")
    parser.add_argument("--steps", type=int, default=20, help="Steps per timed trajectory.")
    parser.add_argument("--repeats", type=int, default=3, help="Timed repetitions after the first call.")
    parser.add_argument("--x64", default="1", choices=("0", "1"), help="Run in float64 (1) or float32 (0).")
    parser.add_argument(
        "--precision",
        default="state",
        choices=("state", "mixed"),
        help="core3d solves in the state dtype, or float32 with float64 correction (mixed).",
    )
    parser.add_argument("--output", default="", help="Where to write the JSON report.")
    arguments = parser.parse_args(argv)

    requested = [name.strip() for name in arguments.cases.split(",") if name.strip()]
    if not requested or set(requested) - {"core3d", "q2d", "hostsync", "shard"}:
        parser.error("--cases must select core3d, q2d, hostsync or shard")
    if arguments.steps < 1 or arguments.repeats < 1:
        parser.error("--steps and --repeats must be positive")
    try:
        core_sizes = [int(value) for value in arguments.core3d_sizes.split(",")]
        q2d_sizes = [int(value) for value in arguments.q2d_sizes.split(",")]
        if min(core_sizes + q2d_sizes) < 2:
            raise ValueError
    except ValueError:
        parser.error("sizes must be comma-separated integers of at least two cells")

    import jax

    from lmx import _pin_matmul_precision

    jax.config.update("jax_enable_x64", arguments.x64 == "1")
    # Pin before reading the environment, so the report records the precision the cases run at.
    _pin_matmul_precision()
    environment = _environment(jax)
    refusal = _unquotable_float32(environment) if arguments.output else None
    if refusal:
        parser.exit(2, f"{parser.prog}: error: {refusal}; no report written\n")
    report = {"environment": environment, "cases": []}
    plan = []
    if "core3d" in requested:
        plan += [(_core3d_case, size) for size in core_sizes]
    if "q2d" in requested:
        plan += [(_q2d_case, size) for size in q2d_sizes]
    if "hostsync" in requested:
        plan += [(_host_sync_case, core_sizes[0])]
    if "shard" in requested:
        plan += [(_shard_case, size) for size in q2d_sizes]
    for builder, size in plan:
        try:
            options = {"precision": arguments.precision} if builder is _core3d_case else {}
            entry = builder(jax, size, arguments.steps, arguments.repeats, **options)
        except Exception as error:  # a size that does not fit is a result, not a crash
            entry = {"case": builder.__name__, "shape": [size], "accepted": False, "error": str(error)[:200]}
        report["cases"].append(entry)
        print(json.dumps(entry), flush=True)
    if arguments.output:
        path = Path(arguments.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")
    return 0 if all(entry["accepted"] for entry in report["cases"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
