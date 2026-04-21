from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

try:
    from scipy.interpolate import RegularGridInterpolator
except Exception:  # pragma: no cover - SciPy is a shipped dependency for normal environments.
    RegularGridInterpolator = None


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
    out = Path(path)
    out.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    return out


def write_tabulated_field_npz(
    path: str | Path,
    *,
    x: np.ndarray | None = None,
    y: np.ndarray,
    z: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
    bz: np.ndarray,
) -> Path:
    payload: dict[str, np.ndarray] = {
        "y": np.asarray(y, dtype=float),
        "z": np.asarray(z, dtype=float),
        "bx": np.asarray(bx, dtype=float),
        "by": np.asarray(by, dtype=float),
        "bz": np.asarray(bz, dtype=float),
    }
    if x is not None:
        payload["x"] = np.asarray(x, dtype=float)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, **payload)
    return out


def load_tabulated_field(path: str | Path) -> dict[str, np.ndarray]:
    source = Path(path)
    if source.suffix.lower() != ".npz":
        raise ValueError("Tabulated magnetic fields currently use NPZ files with x/y/z and bx/by/bz arrays")
    with np.load(source) as payload:
        data = {key: np.asarray(payload[key], dtype=float) for key in payload.files}
    required_2d = {"y", "z", "bx", "by", "bz"}
    required_3d = {"x", "y", "z", "bx", "by", "bz"}
    keys = set(data)
    if required_3d.issubset(keys):
        return data
    if required_2d.issubset(keys):
        return data
    raise ValueError(
        "Tabulated magnetic-field NPZ must contain either y/z/bx/by/bz or x/y/z/bx/by/bz arrays"
    )


def sample_tabulated_cross_section_field(
    path: str | Path,
    *,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    data = load_tabulated_field(path)
    if "x" in data:
        raise ValueError("3D tabulated field needs an x coordinate; use sample_tabulated_field_volume(...)")
    return _interpolate_tabulated_field_2d(data, y=y, z=z)


def sample_tabulated_field_volume(
    path: str | Path,
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    data = load_tabulated_field(path)
    if "x" not in data:
        sampled = _interpolate_tabulated_field_2d(data, y=y, z=z)
        return sampled
    return _interpolate_tabulated_field_3d(data, x=x, y=y, z=z)


def _component_interpolator(points: tuple[np.ndarray, ...], values: np.ndarray) -> RegularGridInterpolator:
    if RegularGridInterpolator is None:
        raise RuntimeError("SciPy RegularGridInterpolator is required for tabulated magnetic fields")
    return RegularGridInterpolator(points, values, bounds_error=False, fill_value=None)


def _interpolate_tabulated_field_2d(data: dict[str, np.ndarray], *, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    y_axis = np.asarray(data["y"], dtype=float)
    z_axis = np.asarray(data["z"], dtype=float)
    points = np.stack([np.asarray(y, dtype=float).reshape(-1), np.asarray(z, dtype=float).reshape(-1)], axis=-1)
    components: list[np.ndarray] = []
    for key in ("bx", "by", "bz"):
        interpolator = _component_interpolator((y_axis, z_axis), np.asarray(data[key], dtype=float))
        sampled = np.asarray(interpolator(points), dtype=float).reshape(np.asarray(y).shape)
        components.append(sampled)
    return np.stack(components, axis=-1)


def _interpolate_tabulated_field_3d(
    data: dict[str, np.ndarray],
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    x_axis = np.asarray(data["x"], dtype=float)
    y_axis = np.asarray(data["y"], dtype=float)
    z_axis = np.asarray(data["z"], dtype=float)
    points = np.stack(
        [
            np.asarray(x, dtype=float).reshape(-1),
            np.asarray(y, dtype=float).reshape(-1),
            np.asarray(z, dtype=float).reshape(-1),
        ],
        axis=-1,
    )
    components: list[np.ndarray] = []
    for key in ("bx", "by", "bz"):
        interpolator = _component_interpolator((x_axis, y_axis, z_axis), np.asarray(data[key], dtype=float))
        sampled = np.asarray(interpolator(points), dtype=float).reshape(np.asarray(x).shape)
        components.append(sampled)
    return np.stack(components, axis=-1)
