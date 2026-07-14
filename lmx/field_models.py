from __future__ import annotations

import ast
from collections.abc import Callable
import json
import re
import warnings
from pathlib import Path

from jax import config as jax_config
import jax.numpy as jnp
import numpy as np

jax_config.update("jax_enable_x64", True)

try:
    from scipy.interpolate import RegularGridInterpolator
except Exception:  # pragma: no cover - SciPy is a shipped dependency for normal environments.
    RegularGridInterpolator = None

try:
    import magpylib_jax as magpy
except Exception:  # pragma: no cover - optional dependency in nonstandard environments.
    magpy = None


def load_wham_coil_model_script(
    path: str | Path,
    *,
    radial_loops: int | None = None,
    axial_loops: int | None = None,
    preserve_ampere_turns: bool = True,
) -> dict[str, float | int | str | bool]:
    """Parse the WHAM coil-model script into LMX field-generation parameters.

    The attached WHAM script defines two high-field coils with circular-current
    loops. This parser extracts the physical dimensions and coil-center
    separation while allowing examples to use a reduced loop count. When
    ``preserve_ampere_turns`` is true, the reduced loop current is scaled to
    keep total ampere-turns approximately fixed.
    """

    source = Path(path)
    text = source.read_text(encoding="utf-8")
    env = _safe_numeric_assignments(text)
    required = ("dz_HF", "r_in_HF", "r_out_HF", "nz", "nr", "I_coil")
    missing = [name for name in required if name not in env]
    if missing:
        raise ValueError(f"WHAM coil model is missing required assignments: {', '.join(missing)}")

    coil_centers = _wham_coil_centers_from_script(text)
    source_radial_loops = int(round(float(env["nr"])))
    source_axial_loops = int(round(float(env["nz"])))
    requested_radial_loops = source_radial_loops if radial_loops is None else max(int(radial_loops), 1)
    requested_axial_loops = source_axial_loops if axial_loops is None else max(int(axial_loops), 1)
    source_loop_current = float(env["I_coil"])
    source_ampere_turns = source_loop_current * source_radial_loops * source_axial_loops
    if preserve_ampere_turns:
        current_scale = source_ampere_turns / (requested_radial_loops * requested_axial_loops)
    else:
        current_scale = source_loop_current
    return {
        "source_path": str(source),
        "coil_axial_thickness": float(env["dz_HF"]),
        "inner_radius": float(env["r_in_HF"]),
        "outer_radius": float(env["r_out_HF"]),
        "source_radial_loops": source_radial_loops,
        "source_axial_loops": source_axial_loops,
        "source_loop_current": source_loop_current,
        "source_ampere_turns": source_ampere_turns,
        "coil_center_negative": float(min(coil_centers)),
        "coil_center_positive": float(max(coil_centers)),
        "coil_separation": float(abs(max(coil_centers) - min(coil_centers))),
        "radial_loops": requested_radial_loops,
        "axial_loops": requested_axial_loops,
        "current_scale": float(current_scale),
        "preserve_ampere_turns": bool(preserve_ampere_turns),
    }


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
        continued = 0.5 * peak_field * (
            1.0 - jnp.tanh(((x - center) + 1j * transverse) / transition_width)
        )
        zero = jnp.zeros_like(x)
        components = ((jnp.imag(continued), jnp.real(continued), zero)
                      if axis == "y" else
                      (jnp.imag(continued), zero, jnp.real(continued)))
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


def sample_wham_mirror_field(
    x: np.ndarray | jnp.ndarray,
    y: np.ndarray | jnp.ndarray,
    z: np.ndarray | jnp.ndarray,
    *,
    coil_separation: float | jnp.ndarray = 1.96,
    current_scale: float = 2000.0 * 17.0 / 17.51,
    inner_radius: float = 0.5 * 86.0e-3,
    outer_radius: float = 0.5 * 730.0e-3,
    coil_axial_thickness: float = 14.3e-3 * 8.0,
    radial_loops: int = 24,
    axial_loops: int = 8,
    x_offset: float | jnp.ndarray = 0.0,
    y_offset: float | jnp.ndarray = 0.0,
    z_offset: float | jnp.ndarray = 0.0,
) -> jnp.ndarray:
    """Sample a differentiable WHAM-like mirror field with magpylib_jax.

    The mirror axis is aligned with ``z`` following the attached WHAM coil-model
    script. This is the right orientation for a pipe flowing along ``x`` and
    crossing the mirror field transversely.
    """

    if magpy is None:
        raise RuntimeError("magpylib_jax is required for WHAM mirror field generation")

    x_arr = jnp.asarray(x, dtype=jnp.float32)
    y_arr = jnp.asarray(y, dtype=jnp.float32)
    z_arr = jnp.asarray(z, dtype=jnp.float32)
    if x_arr.shape != y_arr.shape or x_arr.shape != z_arr.shape:
        raise ValueError("x, y, and z arrays must share the same shape")

    radial_loops = max(int(radial_loops), 1)
    axial_loops = max(int(axial_loops), 1)
    separation = jnp.asarray(coil_separation, dtype=jnp.float32)
    current = jnp.asarray(current_scale, dtype=jnp.float32)
    radii = jnp.linspace(inner_radius, outer_radius, radial_loops, dtype=jnp.float32)
    axial_offsets = jnp.linspace(
        -0.5 * coil_axial_thickness,
        0.5 * coil_axial_thickness,
        axial_loops,
        dtype=jnp.float32,
    )
    x_shifted = x_arr + jnp.asarray(x_offset, dtype=jnp.float32)
    y_shifted = y_arr + jnp.asarray(y_offset, dtype=jnp.float32)
    z_shifted = z_arr + jnp.asarray(z_offset, dtype=jnp.float32)
    points = jnp.stack([x_shifted.reshape(-1), y_shifted.reshape(-1), z_shifted.reshape(-1)], axis=-1)

    sources = []
    for coil_center in (-0.5 * separation, 0.5 * separation):
        for axial_offset in axial_offsets:
            z_center = coil_center + axial_offset
            for radius in radii:
                sources.append(
                    magpy.current.Circle(
                        current=current,
                        diameter=2.0 * radius,
                        position=(0.0, 0.0, z_center),
                    )
                )
    collection = magpy.Collection(*sources)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        field = jnp.asarray(magpy.getB(collection, points), dtype=jnp.float32)
    return field.reshape(x_arr.shape + (3,))


def sample_wham_mirror_axis_profile(
    x: np.ndarray | jnp.ndarray,
    *,
    coil_separation: float | jnp.ndarray = 1.96,
    current_scale: float = 2000.0 * 17.0 / 17.51,
    inner_radius: float = 0.5 * 86.0e-3,
    outer_radius: float = 0.5 * 730.0e-3,
    coil_axial_thickness: float = 14.3e-3 * 8.0,
    radial_loops: int = 24,
    axial_loops: int = 8,
    x_offset: float | jnp.ndarray = 0.0,
    y_offset: float | jnp.ndarray = 0.0,
    z_offset: float | jnp.ndarray = 0.0,
) -> jnp.ndarray:
    x_arr = jnp.asarray(x, dtype=jnp.float32)
    zeros = jnp.zeros_like(x_arr)
    field = sample_wham_mirror_field(
        x_arr,
        zeros,
        zeros,
        coil_separation=coil_separation,
        current_scale=current_scale,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        coil_axial_thickness=coil_axial_thickness,
        radial_loops=radial_loops,
        axial_loops=axial_loops,
        x_offset=x_offset,
        y_offset=y_offset,
        z_offset=z_offset,
    )
    return field[..., 2]


def wham_mirror_station_scale(
    x: np.ndarray | jnp.ndarray,
    *,
    coil_separation: float | jnp.ndarray = 1.96,
    current_scale: float = 2000.0 * 17.0 / 17.51,
    inner_radius: float = 0.5 * 86.0e-3,
    outer_radius: float = 0.5 * 730.0e-3,
    coil_axial_thickness: float = 14.3e-3 * 8.0,
    radial_loops: int = 24,
    axial_loops: int = 8,
    x_offset: float | jnp.ndarray = 0.0,
    y_offset: float | jnp.ndarray = 0.0,
    z_offset: float | jnp.ndarray = 0.0,
) -> jnp.ndarray:
    profile = sample_wham_mirror_axis_profile(
        x,
        coil_separation=coil_separation,
        current_scale=current_scale,
        inner_radius=inner_radius,
        outer_radius=outer_radius,
        coil_axial_thickness=coil_axial_thickness,
        radial_loops=radial_loops,
        axial_loops=axial_loops,
        x_offset=x_offset,
        y_offset=y_offset,
        z_offset=z_offset,
    )
    magnitude = jnp.abs(profile)
    return magnitude / jnp.maximum(jnp.max(magnitude), 1.0e-12)


def write_wham_mirror_field_npz(
    path: str | Path,
    *,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    coil_separation: float = 1.96,
    current_scale: float = 2000.0 * 17.0 / 17.51,
    inner_radius: float = 0.5 * 86.0e-3,
    outer_radius: float = 0.5 * 730.0e-3,
    coil_axial_thickness: float = 14.3e-3 * 8.0,
    radial_loops: int = 24,
    axial_loops: int = 8,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    z_offset: float = 0.0,
) -> Path:
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    field = np.asarray(
        sample_wham_mirror_field(
            xx,
            yy,
            zz,
            coil_separation=coil_separation,
            current_scale=current_scale,
            inner_radius=inner_radius,
            outer_radius=outer_radius,
            coil_axial_thickness=coil_axial_thickness,
            radial_loops=radial_loops,
            axial_loops=axial_loops,
            x_offset=x_offset,
            y_offset=y_offset,
            z_offset=z_offset,
        ),
        dtype=float,
    )
    return write_tabulated_field_npz(
        path,
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        z=np.asarray(z, dtype=float),
        bx=field[..., 0],
        by=field[..., 1],
        bz=field[..., 2],
    )


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


def tabulated_field_quality_metrics(path: str | Path) -> dict[str, float | bool | int | str]:
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
    interpolation_node_l2_error = float(np.linalg.norm(node_error.reshape(-1)) / max(np.linalg.norm(field.reshape(-1)), 1.0e-12))
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
        "dimension": int(3 if has_x else 2),
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


def _safe_numeric_assignments(text: str) -> dict[str, float]:
    tree = ast.parse(text)
    env: dict[str, float] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        try:
            env[node.targets[0].id] = _eval_numeric_ast(node.value, env)
        except ValueError:
            continue
    return env


def _eval_numeric_ast(node: ast.AST, env: dict[str, float]) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.Name) and node.id in env:
        return float(env[node.id])
    if isinstance(node, ast.UnaryOp):
        value = _eval_numeric_ast(node.operand, env)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return value
    if isinstance(node, ast.BinOp):
        left = _eval_numeric_ast(node.left, env)
        right = _eval_numeric_ast(node.right, env)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
    raise ValueError("unsupported nonnumeric expression")


def _wham_coil_centers_from_script(text: str) -> tuple[float, float]:
    direct = re.search(r"HF1\.position\s*=\s*\([^,]+,\s*[^,]+,\s*([^)]+)\)", text)
    copied = re.search(r"HF2\s*=\s*HF1\.copy\(position\s*=\s*\([^,]+,\s*[^,]+,\s*([^)]+)\)\)", text)
    if not direct or not copied:
        raise ValueError("WHAM coil model must define HF1.position and HF2 copy position")
    return float(direct.group(1).strip()), float(copied.group(1).strip())
