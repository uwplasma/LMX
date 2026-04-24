from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
import json
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .fringing import build_square_duct_extruded_problem, solve_extruded_inductionless
from .mesh import generate_rect_duct_mesh


@dataclass(frozen=True)
class StrongScalingRecord:
    backend: str
    device_kind: str
    num_devices: int
    ny: int
    nz: int
    iterations: int
    repeats: int
    cold_seconds: float
    warm_seconds: float
    mean_seconds: float
    python_version: str
    jax_version: str
    nx: int | None = None
    benchmark_kind: str = "stencil2d"
    operator_path: str = "synthetic_stencil2d"
    total_cells: int | None = None
    cell_updates: int | None = None
    warm_cell_updates_per_second: float | None = None
    memory_bytes_estimate: int | None = None
    profile_path: str | None = None


StrongScalingRecordLike = StrongScalingRecord | Mapping[str, object]


_SCALING_TABLE_COLUMNS = (
    "benchmark_kind",
    "operator_path",
    "backend",
    "device_kind",
    "num_devices",
    "nx",
    "ny",
    "nz",
    "iterations",
    "warm_seconds",
    "speedup",
    "parallel_efficiency",
    "warm_mcell_updates_per_second",
    "memory_mib",
    "profile_path",
    "solver_faithful",
)


def _record_mapping(record: StrongScalingRecordLike) -> dict[str, Any]:
    if isinstance(record, StrongScalingRecord):
        return asdict(record)
    return dict(record)


def _float_or_none(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _scaling_group_key(record: Mapping[str, object]) -> tuple[object, ...]:
    return (
        record.get("benchmark_kind", "stencil2d"),
        record.get("operator_path", "synthetic_stencil2d"),
        record.get("backend", ""),
        record.get("device_kind", ""),
        record.get("nx"),
        record.get("ny"),
        record.get("nz"),
        record.get("iterations"),
    )


def summarize_strong_scaling_records(records: Sequence[StrongScalingRecordLike]) -> dict[str, object]:
    """Return derived strong-scaling diagnostics for JSON summaries and CI gates.

    The raw benchmark records intentionally stay close to the timing worker
    output. This helper adds fixed-problem speedup, parallel efficiency,
    throughput, memory, and solver-path flags without changing the benchmark
    itself. Records are grouped by backend, device kind, operator path, grid,
    and iteration count so CPU and GPU studies are not mixed.
    """

    normalized = [_record_mapping(record) for record in records]
    grouped: dict[tuple[object, ...], list[dict[str, Any]]] = {}
    for record in normalized:
        grouped.setdefault(_scaling_group_key(record), []).append(record)

    rows: list[dict[str, object]] = []
    for _, group_records in sorted(grouped.items(), key=lambda item: tuple(str(value) for value in item[0])):
        sorted_records = sorted(group_records, key=lambda row: (_int_or_none(row.get("num_devices")) or 0, str(row.get("backend", ""))))
        baseline = sorted_records[0]
        baseline_devices = max(_int_or_none(baseline.get("num_devices")) or 1, 1)
        baseline_warm = max(_float_or_none(baseline.get("warm_seconds")) or 0.0, 1.0e-20)
        for record in sorted_records:
            num_devices = max(_int_or_none(record.get("num_devices")) or 1, 1)
            warm_seconds = max(_float_or_none(record.get("warm_seconds")) or 0.0, 1.0e-20)
            speedup = baseline_warm / warm_seconds
            device_ratio = num_devices / baseline_devices
            cell_rate = _float_or_none(record.get("warm_cell_updates_per_second"))
            memory_bytes = _float_or_none(record.get("memory_bytes_estimate"))
            operator_path = str(record.get("operator_path", ""))
            rows.append(
                {
                    "benchmark_kind": str(record.get("benchmark_kind", "")),
                    "operator_path": operator_path,
                    "backend": str(record.get("backend", "")),
                    "device_kind": str(record.get("device_kind", "")),
                    "num_devices": num_devices,
                    "nx": _int_or_none(record.get("nx")),
                    "ny": _int_or_none(record.get("ny")),
                    "nz": _int_or_none(record.get("nz")),
                    "iterations": _int_or_none(record.get("iterations")),
                    "warm_seconds": warm_seconds,
                    "speedup": speedup,
                    "parallel_efficiency": speedup / max(device_ratio, 1.0e-20),
                    "warm_mcell_updates_per_second": None if cell_rate is None else cell_rate / 1.0e6,
                    "memory_mib": None if memory_bytes is None else memory_bytes / (1024.0**2),
                    "profile_path": str(record.get("profile_path") or ""),
                    "solver_faithful": operator_path == "solve_extruded_inductionless",
                }
            )

    solver_faithful_count = sum(1 for row in rows if row["solver_faithful"])
    profiled_count = sum(1 for row in rows if row["profile_path"])
    best_speedup = max((float(row["speedup"]) for row in rows), default=0.0)
    best_parallel_efficiency = max((float(row["parallel_efficiency"]) for row in rows), default=0.0)
    return {
        "record_count": len(rows),
        "solver_faithful_record_count": solver_faithful_count,
        "profiled_record_count": profiled_count,
        "best_speedup": best_speedup,
        "best_parallel_efficiency": best_parallel_efficiency,
        "validation_status": "solver_faithful_records_present" if solver_faithful_count else "surrogate_only",
        "rows": rows,
    }


def write_strong_scaling_summary_table(records: Sequence[StrongScalingRecordLike], path: str | Path) -> Path:
    """Write a compact CSV table with derived strong-scaling diagnostics."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = summarize_strong_scaling_records(records)["rows"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_SCALING_TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)  # type: ignore[arg-type]
    return path


def _build_operator_problem(ny: int, nz: int) -> tuple[np.ndarray, ...]:
    mesh = generate_rect_duct_mesh(width=2.0, height=2.0, ny=ny, nz=nz)
    y, z = np.meshgrid(np.asarray(mesh.y_centers), np.asarray(mesh.z_centers), indexing="ij")
    y_scale = max(float(np.max(np.abs(np.asarray(mesh.y_centers)))), 1.0e-12)
    z_scale = max(float(np.max(np.abs(np.asarray(mesh.z_centers)))), 1.0e-12)
    field = np.sin(np.pi * y / y_scale) * np.cos(np.pi * z / z_scale)
    potential = np.cos(0.5 * np.pi * y / y_scale) * np.sin(0.5 * np.pi * z / z_scale)
    forcing = (
        0.5 * np.sin(2.0 * np.pi * y / y_scale)
        - 0.35 * np.cos(3.0 * np.pi * z / z_scale)
        + 0.15 * np.sin(np.pi * y * z / max(y_scale * z_scale, 1.0e-12))
    )
    return field.astype(np.float32), potential.astype(np.float32), forcing.astype(np.float32)


def _build_extruded_operator_problem(nx: int, ny: int, nz: int) -> tuple[np.ndarray, ...]:
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, ny, dtype=np.float32)
    z = np.linspace(-1.0, 1.0, nz, dtype=np.float32)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    sigma = (1.0 + 0.15 * np.cos(2.0 * np.pi * xx) * np.cos(np.pi * yy)).astype(np.float32)
    forcing = (
        0.35 * np.sin(2.0 * np.pi * xx)
        - 0.22 * np.cos(np.pi * yy)
        + 0.18 * np.sin(np.pi * zz)
        + 0.05 * np.sin(2.0 * np.pi * xx * yy)
    ).astype(np.float32)
    u = (0.2 * np.sin(np.pi * yy) * np.cos(np.pi * zz)).astype(np.float32)
    v = (0.12 * np.cos(np.pi * xx) * np.sin(np.pi * zz)).astype(np.float32)
    w = (0.08 * np.sin(np.pi * xx) * np.cos(np.pi * yy)).astype(np.float32)
    phi = (0.1 * np.cos(0.5 * np.pi * xx) * np.sin(np.pi * yy) * np.sin(np.pi * zz)).astype(np.float32)
    return u, v, w, phi, forcing, sigma


def _array_nbytes(array: object) -> int:
    shape = getattr(array, "shape", ())
    dtype = getattr(array, "dtype", np.dtype(float))
    try:
        return int(np.prod(shape, dtype=np.int64) * np.dtype(dtype).itemsize)
    except TypeError:
        return 0


def _bundle_memory_bytes(bundle: object) -> int:
    fields = (
        "u",
        "v",
        "w",
        "p",
        "phi",
        "jx",
        "jy",
        "jz",
        "lorentz_x",
        "lorentz_y",
        "lorentz_z",
    )
    return sum(_array_nbytes(getattr(bundle, name, None)) for name in fields)


def _row_or_replicated_sharding(mesh: Mesh, shape: tuple[int, ...], num_devices: int) -> NamedSharding:
    if shape and shape[0] >= num_devices and shape[0] % num_devices == 0:
        return NamedSharding(mesh, P("d", None))
    return NamedSharding(mesh, P())


def _factor_device_mesh(num_devices: int) -> tuple[int, int]:
    for rows in range(int(np.sqrt(num_devices)), 0, -1):
        if num_devices % rows == 0:
            return rows, num_devices // rows
    return 1, num_devices


def _two_axis_mesh_and_sharding(
    devices: list[object],
    *,
    num_devices: int,
    shape: tuple[int, ...],
) -> tuple[Mesh, NamedSharding]:
    if len(shape) >= 3 and shape[0] >= 2 * max(shape[1:]):
        selected = np.asarray(devices[:num_devices], dtype=object)
        mesh = Mesh(selected, ("d",))
        if shape[0] % num_devices == 0:
            partition = ("d", *([None] * max(0, len(shape) - 1)))
            return mesh, NamedSharding(mesh, P(*partition))
        return mesh, NamedSharding(mesh, P())
    rows, cols = _factor_device_mesh(num_devices)
    if rows == 1 or cols == 1:
        selected = np.asarray(devices[:num_devices], dtype=object)
        mesh = Mesh(selected, ("d",))
        if shape and shape[0] % num_devices == 0:
            partition = ("d", *([None] * max(0, len(shape) - 1)))
            return mesh, NamedSharding(mesh, P(*partition))
        return mesh, NamedSharding(mesh, P())
    selected = np.asarray(devices[:num_devices], dtype=object).reshape(rows, cols)
    mesh = Mesh(selected, ("x", "y"))
    if len(shape) >= 2 and shape[0] % rows == 0 and shape[1] % cols == 0:
        partition = ("x", "y", *([None] * max(0, len(shape) - 2)))
        return mesh, NamedSharding(mesh, P(*partition))
    if shape and shape[0] % num_devices == 0:
        partition = (("x", "y"), *([None] * max(0, len(shape) - 1)))
        return mesh, NamedSharding(mesh, P(*partition))
    raise ValueError(f"Shape {shape} is not compatible with a {rows}x{cols} device mesh.")


def benchmark_sharded_stencil(
    *,
    ny: int = 1024,
    nz: int = 1024,
    iterations: int = 120,
    repeats: int = 3,
    num_devices: int | None = None,
) -> StrongScalingRecord:
    devices = jax.devices()
    if not devices:
        raise RuntimeError("No JAX devices are available for scaling benchmark.")
    if num_devices is None:
        num_devices = len(devices)
    if num_devices < 1 or num_devices > len(devices):
        raise ValueError(f"Requested {num_devices} devices, but only {len(devices)} are visible.")
    if ny % num_devices != 0:
        raise ValueError(f"ny={ny} must be divisible by num_devices={num_devices} for y-sharded scaling.")

    mesh, field_sharding = _two_axis_mesh_and_sharding(devices, num_devices=num_devices, shape=(ny, nz))
    field, potential, forcing = _build_operator_problem(ny, nz)
    _, potential_sharding = _two_axis_mesh_and_sharding(devices, num_devices=num_devices, shape=potential.shape)
    _, forcing_sharding = _two_axis_mesh_and_sharding(devices, num_devices=num_devices, shape=forcing.shape)
    field = jax.device_put(field, field_sharding)
    potential = jax.device_put(potential, potential_sharding)
    forcing = jax.device_put(forcing, forcing_sharding)

    kernel = jax.jit(
        lambda u0, phi0, src: _benchmark_operator_iterations(u0, phi0, src, iterations=iterations),
        in_shardings=(
            field_sharding,
            potential_sharding,
            forcing_sharding,
        ),
        out_shardings=field_sharding,
    )

    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = kernel(field, potential, forcing)
        jax.block_until_ready(result)
        timings.append(time.perf_counter() - start)

    return StrongScalingRecord(
        backend=jax.default_backend(),
        device_kind=devices[0].device_kind,
        num_devices=num_devices,
        ny=ny,
        nz=nz,
        iterations=iterations,
        repeats=repeats,
        cold_seconds=timings[0],
        warm_seconds=min(timings[1:] or timings),
        mean_seconds=sum(timings) / len(timings),
        python_version=platform.python_version(),
        jax_version=jax.__version__,
        operator_path="synthetic_stencil2d",
        total_cells=ny * nz,
        cell_updates=ny * nz * iterations,
        warm_cell_updates_per_second=(ny * nz * iterations) / max(min(timings[1:] or timings), 1.0e-20),
        memory_bytes_estimate=sum(_array_nbytes(array) for array in (field, potential, forcing)),
    )


def benchmark_sharded_extruded_operator(
    *,
    nx: int = 384,
    ny: int = 96,
    nz: int = 96,
    iterations: int = 96,
    repeats: int = 3,
    num_devices: int | None = None,
) -> StrongScalingRecord:
    devices = jax.devices()
    if not devices:
        raise RuntimeError("No JAX devices are available for scaling benchmark.")
    if num_devices is None:
        num_devices = len(devices)
    if num_devices < 1 or num_devices > len(devices):
        raise ValueError(f"Requested {num_devices} devices, but only {len(devices)} are visible.")
    mesh, sharding = _two_axis_mesh_and_sharding(devices, num_devices=num_devices, shape=(nx, ny, nz))
    u, v, w, phi, forcing, sigma = _build_extruded_operator_problem(nx, ny, nz)
    u = jax.device_put(u, sharding)
    v = jax.device_put(v, sharding)
    w = jax.device_put(w, sharding)
    phi = jax.device_put(phi, sharding)
    forcing = jax.device_put(forcing, sharding)
    sigma = jax.device_put(sigma, sharding)

    kernel = jax.jit(
        lambda u0, v0, w0, phi0, src0, sigma0: _benchmark_extruded_operator_iterations(
            u0, v0, w0, phi0, src0, sigma0, iterations=iterations
        ),
        in_shardings=(sharding, sharding, sharding, sharding, sharding, sharding),
        out_shardings=sharding,
    )

    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = kernel(u, v, w, phi, forcing, sigma)
        jax.block_until_ready(result)
        timings.append(time.perf_counter() - start)

    return StrongScalingRecord(
        backend=jax.default_backend(),
        device_kind=devices[0].device_kind,
        num_devices=num_devices,
        nx=nx,
        ny=ny,
        nz=nz,
        iterations=iterations,
        repeats=repeats,
        cold_seconds=timings[0],
        warm_seconds=min(timings[1:] or timings),
        mean_seconds=sum(timings) / len(timings),
        python_version=platform.python_version(),
        jax_version=jax.__version__,
        benchmark_kind="extruded3d",
        operator_path="sharded_extruded_operator_surrogate",
        total_cells=nx * ny * nz,
        cell_updates=nx * ny * nz * iterations,
        warm_cell_updates_per_second=(nx * ny * nz * iterations) / max(min(timings[1:] or timings), 1.0e-20),
        memory_bytes_estimate=sum(_array_nbytes(array) for array in (u, v, w, phi, forcing, sigma)),
    )


def benchmark_extruded_inductionless_solve(
    *,
    nx: int = 48,
    ny: int = 24,
    nz: int = 24,
    ha_peak: float = 20.0,
    max_steps: int = 12,
    potential_iterations: int = 24,
    coupling_iterations: int = 4,
    repeats: int = 2,
    num_devices: int | None = None,
    profile_dir: str | Path | None = None,
) -> StrongScalingRecord:
    """Benchmark the executable rectangular ``extruded_inductionless`` solve path.

    This is intentionally solver-faithful rather than a synthetic sharded
    stencil. It records the number of visible devices for the launched worker,
    but the current solver path does not yet perform explicit multi-device
    domain decomposition.
    """

    devices = jax.devices()
    if not devices:
        raise RuntimeError("No JAX devices are available for scaling benchmark.")
    if num_devices is None:
        num_devices = len(devices)
    if num_devices < 1 or num_devices > len(devices):
        raise ValueError(f"Requested {num_devices} devices, but only {len(devices)} are visible.")

    problem = build_square_duct_extruded_problem(
        ha_peak=ha_peak,
        nx_stations=nx,
        ny=ny,
        nz=nz,
    )
    case = replace(
        problem.case,
        time_stepper=replace(
            problem.case.time_stepper,
            max_steps=max_steps,
            potential_iterations=potential_iterations,
        ),
        solver=replace(
            problem.case.solver,
            coupling_iterations=coupling_iterations,
        ),
    )
    problem = replace(problem, case=case)
    outer_steps = max(2, min(max_steps, max(6, coupling_iterations * 2)))

    timings: list[float] = []
    last_bundle = None
    profile_path: Path | None = None
    for repeat_index in range(repeats):
        trace_started = False
        if profile_dir is not None and repeat_index == 0:
            profile_path = Path(profile_dir)
            profile_path.mkdir(parents=True, exist_ok=True)
            try:
                jax.profiler.start_trace(str(profile_path))
                trace_started = True
            except Exception:
                trace_started = False
        start = time.perf_counter()
        try:
            solution = solve_extruded_inductionless(problem)
            jax.block_until_ready((solution.bundle.u, solution.bundle.phi, solution.bundle.jx))
        finally:
            if trace_started:
                try:
                    jax.profiler.stop_trace()
                except Exception:
                    pass
        timings.append(time.perf_counter() - start)
        last_bundle = solution.bundle

    total_cells = nx * ny * nz
    cell_updates = total_cells * outer_steps
    warm_seconds = min(timings[1:] or timings)
    return StrongScalingRecord(
        backend=jax.default_backend(),
        device_kind=devices[0].device_kind,
        num_devices=num_devices,
        nx=nx,
        ny=ny,
        nz=nz,
        iterations=outer_steps,
        repeats=repeats,
        cold_seconds=timings[0],
        warm_seconds=warm_seconds,
        mean_seconds=sum(timings) / len(timings),
        python_version=platform.python_version(),
        jax_version=jax.__version__,
        benchmark_kind="extruded_solve",
        operator_path="solve_extruded_inductionless",
        total_cells=total_cells,
        cell_updates=cell_updates,
        warm_cell_updates_per_second=cell_updates / max(warm_seconds, 1.0e-20),
        memory_bytes_estimate=_bundle_memory_bytes(last_bundle) if last_bundle is not None else None,
        profile_path=str(profile_path) if profile_path is not None else None,
    )


def _benchmark_operator_iterations(
    u0: jnp.ndarray,
    phi0: jnp.ndarray,
    forcing: jnp.ndarray,
    *,
    iterations: int,
) -> jnp.ndarray:
    def body(_, state):
        u, phi = state
        lap_u = (
            jnp.roll(u, 1, axis=0)
            + jnp.roll(u, -1, axis=0)
            + jnp.roll(u, 1, axis=1)
            + jnp.roll(u, -1, axis=1)
            - 4.0 * u
        )
        lap_phi = (
            jnp.roll(phi, 1, axis=0)
            + jnp.roll(phi, -1, axis=0)
            + jnp.roll(phi, 1, axis=1)
            + jnp.roll(phi, -1, axis=1)
            - 4.0 * phi
        )
        grad_phi_y = 0.5 * (jnp.roll(phi, -1, axis=0) - jnp.roll(phi, 1, axis=0))
        grad_phi_z = 0.5 * (jnp.roll(phi, -1, axis=1) - jnp.roll(phi, 1, axis=1))
        lorentz = 0.18 * grad_phi_z - 0.07 * grad_phi_y
        u_new = (
            0.92 * u
            + 0.06 * lap_u
            + 0.035 * forcing
            + lorentz
            + 0.01 * jnp.sin(3.0 * u)
            + 0.005 * jnp.cos(2.0 * phi)
        )
        phi_source = 0.15 * u_new - 0.04 * forcing + 0.03 * jnp.sin(phi)
        phi_new = 0.9 * phi + 0.08 * lap_phi + phi_source
        return u_new, phi_new

    final_u, _ = jax.lax.fori_loop(0, iterations, body, (u0, phi0))
    return final_u


def _laplacian_3d_benchmark(field: jnp.ndarray) -> jnp.ndarray:
    return (
        jnp.roll(field, 1, axis=0)
        + jnp.roll(field, -1, axis=0)
        + jnp.roll(field, 1, axis=1)
        + jnp.roll(field, -1, axis=1)
        + jnp.roll(field, 1, axis=2)
        + jnp.roll(field, -1, axis=2)
        - 6.0 * field
    )


def _gradient_3d_benchmark(field: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    d_dx = 0.5 * (jnp.roll(field, -1, axis=0) - jnp.roll(field, 1, axis=0))
    d_dy = 0.5 * (jnp.roll(field, -1, axis=1) - jnp.roll(field, 1, axis=1))
    d_dz = 0.5 * (jnp.roll(field, -1, axis=2) - jnp.roll(field, 1, axis=2))
    return d_dx, d_dy, d_dz


def _benchmark_extruded_operator_iterations(
    u0: jnp.ndarray,
    v0: jnp.ndarray,
    w0: jnp.ndarray,
    phi0: jnp.ndarray,
    forcing: jnp.ndarray,
    sigma: jnp.ndarray,
    *,
    iterations: int,
) -> jnp.ndarray:
    bx = 0.0
    by = 1.0
    bz = 0.0

    def body(_, state):
        u, v, w, phi = state
        lap_u = _laplacian_3d_benchmark(u)
        lap_v = _laplacian_3d_benchmark(v)
        lap_w = _laplacian_3d_benchmark(w)
        lap_phi = _laplacian_3d_benchmark(phi)
        dphi_dx, dphi_dy, dphi_dz = _gradient_3d_benchmark(phi)
        uxb_x = v * bz - w * by
        uxb_y = w * bx - u * bz
        uxb_z = u * by - v * bx
        jx = sigma * (-dphi_dx + uxb_x)
        jy = sigma * (-dphi_dy + uxb_y)
        jz = sigma * (-dphi_dz + uxb_z)
        lorentz_x = jy * bz - jz * by
        lorentz_y = jz * bx - jx * bz
        lorentz_z = jx * by - jy * bx
        div_j = (
            0.5 * (jnp.roll(jx, -1, axis=0) - jnp.roll(jx, 1, axis=0))
            + 0.5 * (jnp.roll(jy, -1, axis=1) - jnp.roll(jy, 1, axis=1))
            + 0.5 * (jnp.roll(jz, -1, axis=2) - jnp.roll(jz, 1, axis=2))
        )
        u_new = 0.935 * u + 0.048 * lap_u + 0.022 * forcing + 0.032 * lorentz_x - 0.008 * div_j + 0.004 * jnp.sin(2.5 * u)
        v_new = 0.94 * v + 0.045 * lap_v + 0.018 * lorentz_y - 0.006 * div_j + 0.003 * jnp.cos(2.0 * phi)
        w_new = 0.94 * w + 0.045 * lap_w + 0.018 * lorentz_z - 0.006 * div_j + 0.003 * jnp.sin(1.5 * phi)
        phi_rhs = 0.11 * u_new - 0.035 * forcing - 0.025 * div_j + 0.015 * (jx + jy + jz)
        phi_new = 0.91 * phi + 0.075 * lap_phi + phi_rhs
        return u_new, v_new, w_new, phi_new

    final_u, _, _, _ = jax.lax.fori_loop(0, iterations, body, (u0, v0, w0, phi0))
    return final_u


def write_scaling_report(records: list[StrongScalingRecord], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(record) for record in records]
    path.write_text(json.dumps(payload, indent=2))
    return path
