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


def _build_operator_problem(ny: int, nz: int) -> tuple[jnp.ndarray, ...]:
    mesh = generate_rect_duct_mesh(width=2.0, height=2.0, ny=ny, nz=nz)
    y, z = jnp.meshgrid(mesh.y_centers, mesh.z_centers, indexing="ij")
    y_scale = jnp.maximum(jnp.max(jnp.abs(mesh.y_centers)), 1.0e-12)
    z_scale = jnp.maximum(jnp.max(jnp.abs(mesh.z_centers)), 1.0e-12)
    field = jnp.sin(jnp.pi * y / y_scale) * jnp.cos(jnp.pi * z / z_scale)
    potential = jnp.cos(0.5 * jnp.pi * y / y_scale) * jnp.sin(0.5 * jnp.pi * z / z_scale)
    forcing = (
        0.5 * jnp.sin(2.0 * jnp.pi * y / y_scale)
        - 0.35 * jnp.cos(3.0 * jnp.pi * z / z_scale)
        + 0.15 * jnp.sin(jnp.pi * y * z / jnp.maximum(y_scale * z_scale, 1.0e-12))
    )
    return field.astype(jnp.float32), potential.astype(jnp.float32), forcing.astype(jnp.float32)


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
    field, potential, forcing = _build_operator_problem(ny, nz)
    field_sharding = _row_or_replicated_sharding(mesh, field.shape, num_devices)
    potential_sharding = _row_or_replicated_sharding(mesh, potential.shape, num_devices)
    forcing_sharding = _row_or_replicated_sharding(mesh, forcing.shape, num_devices)
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


def write_scaling_report(records: list[StrongScalingRecord], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [asdict(record) for record in records]
    path.write_text(json.dumps(payload, indent=2))
    return path
