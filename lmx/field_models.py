from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from jax import config as jax_config

jax_config.update("jax_enable_x64", True)

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


def make_maxwell_consistent_fringe_field(
    *, peak_field: float, center: float, transition_width: float, axis: str = "y"
):
    """Return a curl- and divergence-free transverse tanh fringe field."""

    if transition_width <= 0.0:
        raise ValueError("transition_width must be positive")
    if axis not in {"y", "z"}:
        raise ValueError("axis must be 'y' or 'z'")

    def field(x: jnp.ndarray, y: jnp.ndarray, z: jnp.ndarray) -> jnp.ndarray:
        transverse = y if axis == "y" else z
        continued = 0.5 * peak_field * (1.0 - jnp.tanh(((x - center) + 1j * transverse) / transition_width))
        zero = jnp.zeros_like(x)
        components = (
            (jnp.imag(continued), jnp.real(continued), zero)
            if axis == "y"
            else (jnp.imag(continued), zero, jnp.real(continued))
        )
        return jnp.stack(components, axis=-1)

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
    raise ValueError("Tabulated magnetic-field NPZ must contain either y/z/bx/by/bz or x/y/z/bx/by/bz arrays")


def tabulated_field_quality_metrics(
    path: str | Path,
) -> dict[str, float | bool | int | str]:
    """Return interpolation, normalization, and divergence metrics for a field table."""

    data = load_tabulated_field(path)
    has_x = "x" in data
    axes = (data["x"], data["y"], data["z"]) if has_x else (data["y"], data["z"])
    axis_names = ("x", "y", "z") if has_x else ("y", "z")
    axis_monotonic = all(bool(np.all(np.diff(axis) > 0.0)) for axis in axes)
    field = np.stack([data["bx"], data["by"], data["bz"]], axis=-1)
    finite_fraction = float(np.mean(np.isfinite(field)))
    magnitude = np.linalg.norm(field, axis=-1)
    mean_field_magnitude = float(np.mean(magnitude))
    max_field_magnitude = float(np.max(magnitude))
    normalized_magnitude = magnitude / max(max_field_magnitude, 1.0e-12)

    if has_x:
        xx, yy, zz = np.meshgrid(data["x"], data["y"], data["z"], indexing="ij")
        sampled = sample_tabulated_field_volume(path, x=xx, y=yy, z=zz)
        dx = float(data["x"][1] - data["x"][0]) if len(data["x"]) > 1 else 1.0
        dy = float(data["y"][1] - data["y"][0]) if len(data["y"]) > 1 else 1.0
        dz = float(data["z"][1] - data["z"][0]) if len(data["z"]) > 1 else 1.0
        divergence = (
            np.gradient(data["bx"], dx, axis=0)
            + np.gradient(data["by"], dy, axis=1)
            + np.gradient(data["bz"], dz, axis=2)
        )
    else:
        yy, zz = np.meshgrid(data["y"], data["z"], indexing="ij")
        sampled = sample_tabulated_cross_section_field(path, y=yy, z=zz)
        dy = float(data["y"][1] - data["y"][0]) if len(data["y"]) > 1 else 1.0
        dz = float(data["z"][1] - data["z"][0]) if len(data["z"]) > 1 else 1.0
        divergence = np.gradient(data["by"], dy, axis=0) + np.gradient(data["bz"], dz, axis=1)

    node_error = np.asarray(sampled, dtype=float) - field
    interpolation_node_linf_error = float(np.max(np.abs(node_error)))
    interpolation_node_l2_error = float(
        np.linalg.norm(node_error.reshape(-1)) / max(np.linalg.norm(field.reshape(-1)), 1.0e-12)
    )
    max_abs_divergence = float(np.max(np.abs(divergence)))
    rms_divergence = float(np.sqrt(np.mean(divergence**2)))
    divergence_to_field_ratio = rms_divergence / max(mean_field_magnitude, 1.0e-12)
    validation_pass = bool(
        axis_monotonic
        and finite_fraction == 1.0
        and interpolation_node_linf_error <= 1.0e-10
        and divergence_to_field_ratio <= 2.5e-1
        and max_field_magnitude > 0.0
    )
    return {
        "dimension": (3 if has_x else 2),
        "axis_names": ",".join(axis_names),
        "axis_monotonic": axis_monotonic,
        "cell_count": int(np.prod(magnitude.shape)),
        "finite_fraction": finite_fraction,
        "mean_field_magnitude": mean_field_magnitude,
        "max_field_magnitude": max_field_magnitude,
        "normalized_magnitude_min": float(np.min(normalized_magnitude)),
        "normalized_magnitude_max": float(np.max(normalized_magnitude)),
        "interpolation_node_linf_error": interpolation_node_linf_error,
        "interpolation_node_l2_error": interpolation_node_l2_error,
        "max_abs_divergence": max_abs_divergence,
        "rms_divergence": rms_divergence,
        "divergence_to_field_ratio": float(divergence_to_field_ratio),
        "validation_pass": validation_pass,
    }


def tabulated_cross_section_reconstruction_metrics(
    path: str | Path,
    *,
    reference_field_fn: Callable[[jnp.ndarray, jnp.ndarray], jnp.ndarray],
    y: np.ndarray,
    z: np.ndarray,
    relative_l2_tolerance: float = 2.0e-3,
    relative_linf_tolerance: float = 1.0e-2,
) -> dict[str, float | bool | int | str]:
    """Compare a 2D tabulated field against a manufactured reference field.

    The generic table-node interpolation check in
    :func:`tabulated_field_quality_metrics` proves that the NPZ payload is
    internally self-consistent. This check is stricter for validation examples:
    it samples the table at the actual solver or diagnostic points and compares
    those values against the analytic field used to generate the table.
    """

    y_values = np.asarray(y, dtype=float)
    z_values = np.asarray(z, dtype=float)
    yy, zz = np.meshgrid(y_values, z_values, indexing="ij")
    sampled = np.asarray(sample_tabulated_cross_section_field(path, y=yy, z=zz), dtype=float)
    reference = np.asarray(reference_field_fn(jnp.asarray(yy), jnp.asarray(zz)), dtype=float)
    error = sampled - reference
    reference_norm = max(float(np.linalg.norm(reference.reshape(-1))), 1.0e-12)
    reference_abs = max(float(np.max(np.abs(reference))), 1.0e-12)
    relative_l2_error = float(np.linalg.norm(error.reshape(-1)) / reference_norm)
    relative_linf_error = float(np.max(np.abs(error)) / reference_abs)
    magnitude_error = np.linalg.norm(sampled, axis=-1) - np.linalg.norm(reference, axis=-1)
    magnitude_scale = max(float(np.max(np.linalg.norm(reference, axis=-1))), 1.0e-12)
    magnitude_relative_linf_error = float(np.max(np.abs(magnitude_error)) / magnitude_scale)
    component_linf_errors = {
        f"{name}_relative_linf_error": float(
            np.max(np.abs(error[..., index])) / max(float(np.max(np.abs(reference[..., index]))), 1.0e-12)
        )
        for index, name in enumerate(("bx", "by", "bz"))
    }
    validation_pass = bool(
        relative_l2_error <= relative_l2_tolerance
        and relative_linf_error <= relative_linf_tolerance
        and np.all(np.isfinite(sampled))
        and np.all(np.isfinite(reference))
    )
    return {
        "sample_count": int(sampled.shape[0] * sampled.shape[1]),
        "relative_l2_error": relative_l2_error,
        "relative_linf_error": relative_linf_error,
        "magnitude_relative_linf_error": magnitude_relative_linf_error,
        "relative_l2_tolerance": float(relative_l2_tolerance),
        "relative_linf_tolerance": float(relative_linf_tolerance),
        "validation_pass": validation_pass,
        "validation_status": "tabulated_field_matches_manufactured_reference_at_solver_points",
        **component_linf_errors,
    }


def sample_tabulated_cross_section_field(
    path: str | Path,
    *,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    data = load_tabulated_field(path)
    if "x" in data:
        raise ValueError("3D tabulated field needs an x coordinate; use sample_tabulated_field_volume(...)")
    return _interpolate_tabulated_field(data, y=y, z=z)


def sample_tabulated_field_volume(
    path: str | Path,
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    data = load_tabulated_field(path)
    if "x" not in data:
        sampled = _interpolate_tabulated_field(data, y=y, z=z)
        return sampled
    return _interpolate_tabulated_field(data, x=x, y=y, z=z)


def _component_interpolator(points: tuple[np.ndarray, ...], values: np.ndarray) -> RegularGridInterpolator:
    if RegularGridInterpolator is None:
        raise RuntimeError("SciPy RegularGridInterpolator is required for tabulated magnetic fields")
    return RegularGridInterpolator(points, values, bounds_error=False, fill_value=None)


def _interpolate_tabulated_field(data: dict[str, np.ndarray], **coordinates: np.ndarray) -> np.ndarray:
    axes = tuple(np.asarray(data[name], dtype=float) for name in coordinates)
    values = tuple(np.asarray(value, dtype=float) for value in coordinates.values())
    points = np.stack([value.reshape(-1) for value in values], axis=-1)
    components: list[np.ndarray] = []
    for key in ("bx", "by", "bz"):
        interpolator = _component_interpolator(axes, np.asarray(data[key], dtype=float))
        sampled = np.asarray(interpolator(points), dtype=float).reshape(values[0].shape)
        components.append(sampled)
    return np.stack(components, axis=-1)
