from __future__ import annotations

import json
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
    points = jnp.stack([x_arr.reshape(-1), y_arr.reshape(-1), z_arr.reshape(-1)], axis=-1)

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
