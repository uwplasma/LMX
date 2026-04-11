from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import platform
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from .linear import solve_poisson_jacobi_state
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


def _build_poisson_problem(ny: int, nz: int) -> tuple[jnp.ndarray, ...]:
    mesh = generate_rect_duct_mesh(width=2.0, height=2.0, ny=ny, nz=nz)
    dy = mesh.dy[:, None].astype(jnp.float32)
    dz = mesh.dz[None, :].astype(jnp.float32)
    west = jnp.where(jnp.arange(ny)[:, None] > 0, 1.0 / jnp.maximum(dy**2, 1.0e-12), 0.0)
    east = jnp.where(jnp.arange(ny)[:, None] < ny - 1, 1.0 / jnp.maximum(dy**2, 1.0e-12), 0.0)
    south = jnp.where(jnp.arange(nz)[None, :] > 0, 1.0 / jnp.maximum(dz**2, 1.0e-12), 0.0)
    north = jnp.where(jnp.arange(nz)[None, :] < nz - 1, 1.0 / jnp.maximum(dz**2, 1.0e-12), 0.0)
    diagonal = west + east + south + north
    diagonal = diagonal.at[0, 0].set(1.0)
    y, z = jnp.meshgrid(mesh.y_centers, mesh.z_centers, indexing="ij")
    rhs = jnp.sin(jnp.pi * y / jnp.maximum(jnp.max(jnp.abs(mesh.y_centers)), 1.0e-12))
    rhs = rhs * jnp.cos(jnp.pi * z / jnp.maximum(jnp.max(jnp.abs(mesh.z_centers)), 1.0e-12))
    rhs = rhs - jnp.mean(rhs)
    return diagonal, west, east, south, north, rhs


def _row_or_replicated_sharding(mesh: Mesh, shape: tuple[int, ...], num_devices: int) -> NamedSharding:
    if shape and shape[0] >= num_devices and shape[0] % num_devices == 0:
        return NamedSharding(mesh, P("d", None))
    return NamedSharding(mesh, P())


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

    selected = np.asarray(devices[:num_devices], dtype=object)
    mesh = Mesh(selected, ("d",))
    field_sharding = NamedSharding(mesh, P("d", None))
    diagonal, west, east, south, north, rhs = _build_poisson_problem(ny, nz)
    diagonal_sharding = _row_or_replicated_sharding(mesh, diagonal.shape, num_devices)
    west_sharding = _row_or_replicated_sharding(mesh, west.shape, num_devices)
    east_sharding = _row_or_replicated_sharding(mesh, east.shape, num_devices)
    south_sharding = _row_or_replicated_sharding(mesh, south.shape, num_devices)
    north_sharding = _row_or_replicated_sharding(mesh, north.shape, num_devices)
    rhs_sharding = _row_or_replicated_sharding(mesh, rhs.shape, num_devices)
    diagonal = jax.device_put(diagonal, diagonal_sharding)
    west = jax.device_put(west, west_sharding)
    east = jax.device_put(east, east_sharding)
    south = jax.device_put(south, south_sharding)
    north = jax.device_put(north, north_sharding)
    rhs = jax.device_put(rhs, rhs_sharding)

    kernel = jax.jit(
        lambda d, w, e, s, n, r: solve_poisson_jacobi_state(
            d,
            w,
            e,
            s,
            n,
            r,
            anchor=(0, 0),
            iterations=iterations,
            tolerance=None,
            relaxation=1.0,
        )[0],
        in_shardings=(
            diagonal_sharding,
            west_sharding,
            east_sharding,
            south_sharding,
            north_sharding,
            rhs_sharding,
        ),
        out_shardings=field_sharding,
    )

    timings: list[float] = []
    for _ in range(repeats):
        start = time.perf_counter()
        result = kernel(diagonal, west, east, south, north, rhs)
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
    )


def write_scaling_report(records: list[StrongScalingRecord], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(record) for record in records]
    path.write_text(json.dumps(payload, indent=2))
    return path
