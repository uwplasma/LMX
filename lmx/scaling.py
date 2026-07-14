from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .benchmarks import build_benchmark_b_field_profile, build_benchmark_b_problem
from .fringing import solve_extruded_inductionless
from .io import load_extruded_restart_bundle, validate_extruded_restart_bundle


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
    benchmark_kind: str = "extruded3d"
    operator_path: str = "sharded_extruded_operator_surrogate"
    total_cells: int | None = None
    cell_updates: int | None = None
    warm_cell_updates_per_second: float | None = None
    memory_bytes_estimate: int | None = None
    profile_path: str | None = None
    spatially_sharded: bool = False
    global_shard_count: int = 1
    velocity_l2: float | None = None
    potential_l2: float | None = None
    current_l2: float | None = None
    max_charge_balance_residual: float | None = None
    max_boundary_current_residual: float | None = None
    max_electric_local_residual: float | None = None
    electric_solves_converged: bool | None = None
    validation_passed: bool | None = None
    initialization: str = "cold_start"
    restart_sha256: str | None = None
    final_update_residual: float | None = None
    steady_state_passed: bool | None = None
    signature_relative_tolerance: float = 2.0e-6


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
    "spatially_sharded",
    "global_shard_count",
    "initialization",
    "validation_passed",
    "steady_state_passed",
    "signature_relative_tolerance",
    "physics_equivalent",
    "solver_faithful",
)

_BUNDLE_FIELD_NAMES = (
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
        record.get("benchmark_kind", "extruded3d"),
        record.get("operator_path", "sharded_extruded_operator_surrogate"),
        record.get("backend", ""),
        record.get("device_kind", ""),
        record.get("nx"),
        record.get("ny"),
        record.get("nz"),
        record.get("iterations"),
        record.get("initialization", "cold_start"),
    )


def summarize_strong_scaling_records(
    records: Sequence[StrongScalingRecordLike],
) -> dict[str, object]:
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
    for _, group_records in sorted(
        grouped.items(), key=lambda item: tuple(str(value) for value in item[0])
    ):
        sorted_records = sorted(
            group_records,
            key=lambda row: (
                _int_or_none(row.get("num_devices")) or 0,
                str(row.get("backend", "")),
            ),
        )
        baseline = sorted_records[0]
        baseline_devices = max(_int_or_none(baseline.get("num_devices")) or 1, 1)
        baseline_warm = max(
            _float_or_none(baseline.get("warm_seconds")) or 0.0, 1.0e-20
        )
        baseline_signature = tuple(
            _float_or_none(baseline.get(name))
            for name in ("velocity_l2", "potential_l2", "current_l2")
        )
        for record in sorted_records:
            num_devices = max(_int_or_none(record.get("num_devices")) or 1, 1)
            warm_seconds = max(
                _float_or_none(record.get("warm_seconds")) or 0.0, 1.0e-20
            )
            speedup = baseline_warm / warm_seconds
            device_ratio = num_devices / baseline_devices
            cell_rate = _float_or_none(record.get("warm_cell_updates_per_second"))
            memory_bytes = _float_or_none(record.get("memory_bytes_estimate"))
            operator_path = str(record.get("operator_path", ""))
            signature = tuple(
                _float_or_none(record.get(name))
                for name in ("velocity_l2", "potential_l2", "current_l2")
            )
            signature_rtol = (
                _float_or_none(record.get("signature_relative_tolerance")) or 2.0e-6
            )
            physics_equivalent = all(
                reference is not None
                and value is not None
                and np.isclose(value, reference, rtol=signature_rtol, atol=1.0e-10)
                for value, reference in zip(signature, baseline_signature, strict=True)
            ) and bool(record.get("validation_passed", False))
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
                    "warm_mcell_updates_per_second": None
                    if cell_rate is None
                    else cell_rate / 1.0e6,
                    "memory_mib": None
                    if memory_bytes is None
                    else memory_bytes / (1024.0**2),
                    "profile_path": str(record.get("profile_path") or ""),
                    "spatially_sharded": bool(record.get("spatially_sharded", False)),
                    "global_shard_count": _int_or_none(record.get("global_shard_count"))
                    or 1,
                    "initialization": str(record.get("initialization", "cold_start")),
                    "validation_passed": bool(record.get("validation_passed", False)),
                    "steady_state_passed": bool(
                        record.get("steady_state_passed", False)
                    ),
                    "signature_relative_tolerance": signature_rtol,
                    "physics_equivalent": physics_equivalent,
                    "solver_faithful": operator_path == "solve_extruded_inductionless",
                }
            )

    solver_faithful_count = sum(1 for row in rows if row["solver_faithful"])
    profiled_count = sum(1 for row in rows if row["profile_path"])
    physics_equivalent_count = sum(1 for row in rows if row["physics_equivalent"])
    best_speedup = max((float(row["speedup"]) for row in rows), default=0.0)
    best_parallel_efficiency = max(
        (float(row["parallel_efficiency"]) for row in rows), default=0.0
    )
    return {
        "record_count": len(rows),
        "solver_faithful_record_count": solver_faithful_count,
        "profiled_record_count": profiled_count,
        "physics_equivalent_record_count": physics_equivalent_count,
        "best_speedup": best_speedup,
        "best_parallel_efficiency": best_parallel_efficiency,
        "validation_status": "solver_faithful_records_present"
        if solver_faithful_count
        else "surrogate_only",
        "rows": rows,
    }


def write_strong_scaling_summary_table(
    records: Sequence[StrongScalingRecordLike], path: str | Path
) -> Path:
    """Write a compact CSV table with derived strong-scaling diagnostics."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = summarize_strong_scaling_records(records)["rows"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_SCALING_TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)  # type: ignore[arg-type]
    return path


def _build_extruded_operator_problem(
    nx: int, ny: int, nz: int
) -> tuple[np.ndarray, ...]:
    x = np.linspace(0.0, 1.0, nx, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, ny, dtype=np.float32)
    z = np.linspace(-1.0, 1.0, nz, dtype=np.float32)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    sigma = (1.0 + 0.15 * np.cos(2.0 * np.pi * xx) * np.cos(np.pi * yy)).astype(
        np.float32
    )
    forcing = (
        0.35 * np.sin(2.0 * np.pi * xx)
        - 0.22 * np.cos(np.pi * yy)
        + 0.18 * np.sin(np.pi * zz)
        + 0.05 * np.sin(2.0 * np.pi * xx * yy)
    ).astype(np.float32)
    u = (0.2 * np.sin(np.pi * yy) * np.cos(np.pi * zz)).astype(np.float32)
    v = (0.12 * np.cos(np.pi * xx) * np.sin(np.pi * zz)).astype(np.float32)
    w = (0.08 * np.sin(np.pi * xx) * np.cos(np.pi * yy)).astype(np.float32)
    phi = (
        0.1 * np.cos(0.5 * np.pi * xx) * np.sin(np.pi * yy) * np.sin(np.pi * zz)
    ).astype(np.float32)
    return u, v, w, phi, forcing, sigma


def _array_nbytes(array: object) -> int:
    shape = getattr(array, "shape", ())
    dtype = getattr(array, "dtype", np.dtype(float))
    try:
        return int(np.prod(shape, dtype=np.int64) * np.dtype(dtype).itemsize)
    except TypeError:
        return 0


def _shard_placement(array: object, expected_shards: int) -> tuple[bool, int]:
    """Validate a benchmark result and report its actual global placement."""

    global_shards = len(getattr(array, "global_shards", ()))
    sharding = getattr(array, "sharding", None)
    replicated = bool(getattr(sharding, "is_fully_replicated", global_shards == 1))
    if expected_shards > 1 and (global_shards != expected_shards or replicated):
        raise RuntimeError(
            "Operator result is not spatially partitioned across the requested "
            f"{expected_shards} devices (global_shards={global_shards}, "
            f"replicated={replicated})."
        )
    return expected_shards > 1 and not replicated, global_shards


def _bundle_memory_bytes(bundle: object) -> int:
    return sum(
        _array_nbytes(getattr(bundle, name, None)) for name in _BUNDLE_FIELD_NAMES
    )


def _row_or_replicated_sharding(
    mesh: Mesh, shape: tuple[int, ...], num_devices: int
) -> NamedSharding:
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
    raise ValueError(
        f"Shape {shape} is not compatible with a {rows}x{cols} device mesh."
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
        raise ValueError(
            f"Requested {num_devices} devices, but only {len(devices)} are visible."
        )
    mesh, sharding = _two_axis_mesh_and_sharding(
        devices, num_devices=num_devices, shape=(nx, ny, nz)
    )
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
    spatially_sharded, global_shard_count = _shard_placement(result, num_devices)

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
        warm_cell_updates_per_second=(nx * ny * nz * iterations)
        / max(min(timings[1:] or timings), 1.0e-20),
        memory_bytes_estimate=sum(
            _array_nbytes(array) for array in (u, v, w, phi, forcing, sigma)
        ),
        spatially_sharded=spatially_sharded,
        global_shard_count=global_shard_count,
    )


def benchmark_extruded_inductionless_solve(
    *,
    nx: int = 48,
    ny: int = 24,
    nz: int = 24,
    max_steps: int = 12,
    potential_iterations: int = 24,
    coupling_iterations: int = 4,
    repeats: int = 2,
    num_devices: int | None = None,
    profile_dir: str | Path | None = None,
    strict_validation: bool = True,
    restart_path: str | Path | None = None,
) -> StrongScalingRecord:
    """Benchmark the production ALEX B2 ``extruded_inductionless`` solve path.

    This solver-faithful path applies named axial sharding to the production
    fields and verifies that the returned solution remains distributed.
    """

    devices = jax.devices()
    if not devices:
        raise RuntimeError("No JAX devices are available for scaling benchmark.")
    if num_devices is None:
        num_devices = len(devices)
    if num_devices < 1 or num_devices > len(devices):
        raise ValueError(
            f"Requested {num_devices} devices, but only {len(devices)} are visible."
        )

    problem = build_benchmark_b_problem("B2-fringing-square", mesh_level="coarse")
    geometry = replace(problem.case.geometry, nx=nx, ny=ny, nz=nz)
    case = replace(
        problem.case,
        geometry=geometry,
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
    problem = replace(
        problem,
        case=case,
        profile=build_benchmark_b_field_profile(
            "B2-fringing-square", axial_stations=nx
        ),
    )
    initial_bundle = None
    restart_sha256 = None
    if restart_path is not None:
        restart_path = Path(restart_path)
        restart = load_extruded_restart_bundle(restart_path)
        validate_extruded_restart_bundle(restart, case=case)
        initial_bundle = restart.bundle
        restart_sha256 = hashlib.sha256(restart_path.read_bytes()).hexdigest()
    outer_steps = max(2, min(max_steps, max(6, coupling_iterations * 2)))

    timings: list[float] = []
    last_bundle = None
    shard_count = 1
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
            solve_kwargs = {"num_devices": num_devices}
            if initial_bundle is not None:
                solve_kwargs["initial_bundle"] = initial_bundle
            solution = solve_extruded_inductionless(problem, **solve_kwargs)
            jax.block_until_ready(
                tuple(getattr(solution.bundle, name) for name in _BUNDLE_FIELD_NAMES)
            )
        finally:
            if trace_started:
                try:
                    jax.profiler.stop_trace()
                except Exception:
                    pass
        timings.append(time.perf_counter() - start)
        last_bundle = solution.bundle
        shard_counts = {
            len(getattr(solution.bundle, name).addressable_shards)
            for name in ("u", "v", "w", "p", "phi", "jx", "jy", "jz")
        }
        shard_count = min(shard_counts)
        if num_devices > 1 and shard_counts != {num_devices}:
            raise RuntimeError(
                "Production fields returned shard counts "
                f"{sorted(shard_counts)} for {num_devices} devices."
            )

    actual_shape = tuple(int(value) for value in last_bundle.u.shape)
    actual_nx, actual_ny, actual_nz = actual_shape
    total_cells = actual_nx * actual_ny * actual_nz
    executed_steps = int(
        getattr(last_bundle, "iteration_residual_history", np.empty(outer_steps)).size
    )
    cell_updates = total_cells * executed_steps
    warm_seconds = min(timings[1:] or timings)
    velocity_l2 = float(
        np.sqrt(
            np.sum(np.asarray(last_bundle.u) ** 2)
            + np.sum(np.asarray(last_bundle.v) ** 2)
            + np.sum(np.asarray(last_bundle.w) ** 2)
        )
    )
    potential_l2 = float(np.linalg.norm(np.asarray(last_bundle.phi)))
    current_l2 = float(
        np.sqrt(
            np.sum(np.asarray(last_bundle.jx) ** 2)
            + np.sum(np.asarray(last_bundle.jy) ** 2)
            + np.sum(np.asarray(last_bundle.jz) ** 2)
        )
    )
    if not all(
        np.isfinite(value) and value > 0.0
        for value in (velocity_l2, potential_l2, current_l2)
    ):
        raise RuntimeError(
            "Production scaling solve returned a zero or nonfinite physics signature."
        )
    charge_residual = np.asarray(last_bundle.charge_balance_residual, dtype=float)
    boundary_residual = np.asarray(last_bundle.boundary_current_residual, dtype=float)
    electric_history = np.asarray(
        last_bundle.iteration_electric_linear_history, dtype=float
    ).reshape((-1, 6))
    max_charge_residual = float(np.max(np.abs(charge_residual)))
    max_boundary_residual = float(np.max(np.abs(boundary_residual)))
    max_electric_local_residual = float(np.max(np.abs(electric_history[:, 2])))
    electric_solves_converged = bool(np.all(electric_history[:, 4] == 1.0))
    update_history = np.asarray(
        getattr(last_bundle, "iteration_residual_history", ()), dtype=float
    )
    component_history = np.asarray(
        getattr(last_bundle, "iteration_component_residual_history", ()), dtype=float
    ).reshape((-1, 6))
    sustained_count = min(3, len(update_history), len(component_history))
    final_update_residual = (
        float(update_history[-1]) if update_history.size else float("inf")
    )
    steady_state_passed = bool(
        sustained_count == 3
        and np.all(update_history[-sustained_count:] <= case.solver.coupling_tolerance)
        and np.all(component_history[-sustained_count:, 3:] <= 1.0e-3)
    )
    validation_passed = bool(
        np.isfinite(max_charge_residual)
        and np.isfinite(max_boundary_residual)
        and np.isfinite(max_electric_local_residual)
        and max_charge_residual <= 1.0e-3
        and max_boundary_residual <= 1.0e-3
        and max_electric_local_residual <= 1.0e-3
        and electric_solves_converged
        and steady_state_passed
    )
    if strict_validation and not validation_passed:
        raise RuntimeError(
            "Production scaling solve failed conservation/linear-solve validation: "
            f"charge={max_charge_residual:.6e}, "
            f"boundary_current={max_boundary_residual:.6e}, "
            f"electric_local={max_electric_local_residual:.6e}, "
            f"electric_converged={electric_solves_converged}, "
            f"steady={steady_state_passed}."
        )
    return StrongScalingRecord(
        backend=jax.default_backend(),
        device_kind=devices[0].device_kind,
        num_devices=num_devices,
        nx=actual_nx,
        ny=actual_ny,
        nz=actual_nz,
        iterations=executed_steps,
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
        memory_bytes_estimate=_bundle_memory_bytes(last_bundle)
        if last_bundle is not None
        else None,
        profile_path=str(profile_path) if profile_path is not None else None,
        spatially_sharded=num_devices > 1,
        global_shard_count=shard_count,
        velocity_l2=velocity_l2,
        potential_l2=potential_l2,
        current_l2=current_l2,
        max_charge_balance_residual=max_charge_residual,
        max_boundary_current_residual=max_boundary_residual,
        max_electric_local_residual=max_electric_local_residual,
        electric_solves_converged=electric_solves_converged,
        validation_passed=validation_passed,
        initialization="restart" if initial_bundle is not None else "cold_start",
        restart_sha256=restart_sha256,
        final_update_residual=final_update_residual,
        steady_state_passed=steady_state_passed,
        signature_relative_tolerance=0.5 * case.solver.coupling_tolerance,
    )


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


def _gradient_3d_benchmark(
    field: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
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
        u_new = (
            0.935 * u
            + 0.048 * lap_u
            + 0.022 * forcing
            + 0.032 * lorentz_x
            - 0.008 * div_j
            + 0.004 * jnp.sin(2.5 * u)
        )
        v_new = (
            0.94 * v
            + 0.045 * lap_v
            + 0.018 * lorentz_y
            - 0.006 * div_j
            + 0.003 * jnp.cos(2.0 * phi)
        )
        w_new = (
            0.94 * w
            + 0.045 * lap_w
            + 0.018 * lorentz_z
            - 0.006 * div_j
            + 0.003 * jnp.sin(1.5 * phi)
        )
        phi_rhs = (
            0.11 * u_new - 0.035 * forcing - 0.025 * div_j + 0.015 * (jx + jy + jz)
        )
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
