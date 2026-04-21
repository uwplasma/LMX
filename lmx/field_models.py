from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np


def make_divergence_free_cross_section_field(
    *,
    width: float,
    height: float,
    base_bz: float,
    perturbation: float = 0.15,
):
    """Return an analytic cross-sectional field with dBy/dy + dBz/dz = 0."""

    def field(y: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        y_hat = 2.0 * y / width
        z_hat = 2.0 * z / height
        phase_y = 0.5 * jnp.pi * y_hat
        phase_z = 0.5 * jnp.pi * z_hat
        by = perturbation * base_bz * jnp.sin(phase_y) * jnp.cos(phase_z)
        bz = base_bz * (1.0 - perturbation * (height / width) * jnp.cos(phase_y) * jnp.sin(phase_z))
        bx = jnp.zeros_like(by)
        return jnp.stack([bx, by, bz], axis=-1)

    return field


def make_localized_divergence_free_obstacle_field(
    *,
    width: float,
    height: float,
    base_bz: float,
    core_fraction_y: float = 0.35,
    core_fraction_z: float = 0.35,
):
    """Return a localized divergence-free field with a central Bz-dominant obstacle."""

    ay = max(0.5 * width * core_fraction_y, 1.0e-6)
    az = max(0.5 * height * core_fraction_z, 1.0e-6)

    def field(y: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        gaussian = jnp.exp(-((y / ay) ** 2 + (z / az) ** 2))
        by = 2.0 * base_bz * y * z * gaussian / (az**2)
        bz = base_bz * gaussian * (1.0 - 2.0 * (y / ay) ** 2)
        bx = jnp.zeros_like(by)
        return jnp.stack([bx, by, bz], axis=-1)

    return field


def sample_cross_section_field(
    field_fn,
    *,
    width: float,
    height: float,
    ny: int,
    nz: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.linspace(-0.5 * width, 0.5 * width, ny)
    z = np.linspace(-0.5 * height, 0.5 * height, nz)
    yy, zz = np.meshgrid(y, z, indexing="ij")
    field = np.asarray(field_fn(jnp.asarray(yy), jnp.asarray(zz)), dtype=float)
    return y, z, field


def cross_section_divergence_metrics(
    field_fn,
    *,
    width: float,
    height: float,
    ny: int = 81,
    nz: int = 81,
) -> dict[str, float]:
    y, z, field = sample_cross_section_field(field_fn, width=width, height=height, ny=ny, nz=nz)
    dy = float(y[1] - y[0]) if len(y) > 1 else 1.0
    dz = float(z[1] - z[0]) if len(z) > 1 else 1.0
    by = field[..., 1]
    bz = field[..., 2]
    dby_dy = np.gradient(by, dy, axis=0)
    dbz_dz = np.gradient(bz, dz, axis=1)
    div = dby_dy + dbz_dz
    magnitude = np.sqrt(by**2 + bz**2)
    return {
        "max_abs_divergence": float(np.max(np.abs(div))),
        "rms_divergence": float(np.sqrt(np.mean(div**2))),
        "mean_field_magnitude": float(np.mean(magnitude)),
    }


def save_cross_section_divergence_report(metrics: dict[str, float], path: str | Path) -> Path:
    import json

    out = Path(path)
    out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return out
