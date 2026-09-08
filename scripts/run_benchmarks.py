#!/usr/bin/env python3
"""Time LMX on whatever device it is given, and write the result as JSON.

Three numbers are reported for every case and they are not interchangeable.
``compile_seconds`` is the first call, which includes tracing and XLA
compilation. ``warm_seconds`` is the best of several later calls, which is what
a long run pays. ``seconds_per_step`` divides the warm time by the number of
steps in the trajectory, which is the only number that compares across sizes.

Every case is a single compiled call. Anything that synchronises with the host
inside the loop would show up here as a warm time that scales with the step
count rather than with the work, which is the failure this harness exists to
catch on an accelerator.

``accepted`` is a correctness flag, not a timing one: a case that produced a
non-finite field or drifted off its divergence-free constraint is reported and
its timings are not to be quoted.
"""

from __future__ import annotations

import argparse
import json
import platform
import socket
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
    dirty = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def _environment(jax) -> dict:
    devices = jax.devices()
    return {
        "host": socket.gethostname(),
        "platform": devices[0].platform,
        "device_kind": devices[0].device_kind,
        "device_count": len(devices),
        "cpu_count": len(__import__("os").sched_getaffinity(0))
        if hasattr(__import__("os"), "sched_getaffinity")
        else None,
        "python": platform.python_version(),
        "jax": jax.__version__,
        "x64": bool(jax.config.jax_enable_x64),
        "lmx_commit": _commit(),
    }


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


def _core3d_case(jax, cells: int, steps: int, repeats: int) -> dict:
    """One compiled trajectory of the staggered core on a cubic duct."""
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
    compile_seconds, warm, result = _timed(jax, lambda: lmx.solve(problem).vorticity, repeats)
    finite = bool(np.all(np.isfinite(np.asarray(result))))
    return {
        "case": "q2d_evolve",
        "cells": cells**2,
        "shape": [cells, cells],
        "steps": steps,
        "compile_seconds": compile_seconds,
        "warm_seconds": warm,
        "seconds_per_step": warm / steps,
        "accepted": finite,
    }


def _host_sync_case(jax, cells: int, steps: int, repeats: int) -> dict:
    """Time the same trajectory at two lengths; a host sync per step would show here.

    A compiled `scan` costs the same per step however many steps it runs. Any
    synchronisation with the host inside the loop adds a fixed latency per step
    that does not compile away, so the ratio of the two per-step times is a
    direct measurement of whether the loop is really running ahead.
    """
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
        "ratio_tolerance": 1.15,
        "accepted": bool(short["accepted"] and long["accepted"] and ratio < 1.15),
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
        problem = lmx.Q2DProblem(
            field,
            length=(2.0 * np.pi, 2.0 * np.pi),
            viscosity=2.0e-4,
            hartmann_friction=2.0e-2,
            dt=2.0e-3,
            steps=steps,
            history_stride=steps,
        )
        return lmx.solve(problem).vorticity

    compiled = jax.jit(evolve)
    single = jax.device_put(jnp.asarray(vorticity, dtype=dtype), devices[0])
    _, single_seconds, reference = _timed(jax, lambda: compiled(single), repeats)
    mesh = Mesh(np.array(devices), ("d",))
    placed = jax.device_put(jnp.asarray(vorticity, dtype=dtype), NamedSharding(mesh, Spec("d", None)))
    _, sharded_seconds, result = _timed(jax, lambda: compiled(placed), repeats)
    difference = float(jnp.max(jnp.abs(jnp.asarray(result) - jnp.asarray(reference))))
    scale = float(jnp.max(jnp.abs(jnp.asarray(reference))))
    tolerance = 1.0e3 * float(np.finfo(np.asarray(reference).dtype).eps) * scale
    return {
        "case": "q2d_shard",
        "cells": cells**2,
        "shape": [cells, cells],
        "steps": steps,
        "devices": len(devices),
        "single_seconds": single_seconds,
        "sharded_seconds": sharded_seconds,
        "efficiency": single_seconds / (len(devices) * sharded_seconds),
        "agreement": difference,
        "agreement_tolerance": tolerance,
        "accepted": difference < tolerance,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cases", default="core3d,q2d,hostsync", help="Comma-separated case names.")
    parser.add_argument("--core3d-sizes", default="32,64,128", help="Cubic cell counts per side.")
    parser.add_argument("--q2d-sizes", default="256,512,1024", help="Square cell counts per side.")
    parser.add_argument("--steps", type=int, default=20, help="Steps per timed trajectory.")
    parser.add_argument("--repeats", type=int, default=3, help="Timed repetitions after the first call.")
    parser.add_argument("--x64", default="1", choices=("0", "1"), help="Run in float64 (1) or float32 (0).")
    parser.add_argument("--output", default="", help="Where to write the JSON report.")
    arguments = parser.parse_args(argv)

    import jax

    jax.config.update("jax_enable_x64", arguments.x64 == "1")
    report = {"environment": _environment(jax), "cases": []}
    requested = [name.strip() for name in arguments.cases.split(",") if name.strip()]
    plan = []
    if "core3d" in requested:
        plan += [(_core3d_case, int(value)) for value in arguments.core3d_sizes.split(",") if value]
    if "q2d" in requested:
        plan += [(_q2d_case, int(value)) for value in arguments.q2d_sizes.split(",") if value]
    if "hostsync" in requested:
        plan += [(_host_sync_case, int(arguments.core3d_sizes.split(",")[0]))]
    if "shard" in requested:
        plan += [(_shard_case, int(value)) for value in arguments.q2d_sizes.split(",") if value]
    for builder, size in plan:
        try:
            entry = builder(jax, size, arguments.steps, arguments.repeats)
        except Exception as error:  # a size that does not fit is a result, not a crash
            entry = {"case": builder.__name__, "shape": [size], "accepted": False, "error": str(error)[:200]}
        report["cases"].append(entry)
        print(json.dumps(entry), flush=True)
    if arguments.output:
        path = Path(arguments.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
