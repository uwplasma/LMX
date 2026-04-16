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
    nx: int | None = None
    benchmark_kind: str = "stencil2d"


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
    if nx % num_devices != 0:
        raise ValueError(f"nx={nx} must be divisible by num_devices={num_devices} for x-sharded scaling.")

    selected = np.asarray(devices[:num_devices], dtype=object)
    mesh = Mesh(selected, ("d",))
    sharding = NamedSharding(mesh, P("d", None, None))
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
