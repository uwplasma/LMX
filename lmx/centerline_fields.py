"""Field sampling and local-frame projections on centerline pipe meshes."""

from __future__ import annotations

from collections.abc import Callable
import csv
import json
from pathlib import Path

import numpy as np

from .mesh import StructuredMesh


FieldSampler = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def centerline_pipe_frames(mesh: StructuredMesh) -> dict[str, np.ndarray]:
    """Recover stationwise orthonormal pipe frames from a mapped centerline mesh."""

    points = _centerline_points(mesh)
    if points.shape[1] < 2:
        raise ValueError("centerline pipe frame recovery requires at least one nonzero radial face")
    station = np.asarray(mesh.x_faces, dtype=float)
    center = points[:, 0, 0, :]
    tangent = np.gradient(center, station, axis=0)
    tangent = _unit_vector(tangent)

    normal = points[:, 1, 0, :] - center
    normal = normal - np.sum(normal * tangent, axis=1, keepdims=True) * tangent
    normal = _unit_vector(normal)
    binormal = np.cross(tangent, normal)
    binormal = _unit_vector(binormal)
    normal = _unit_vector(np.cross(binormal, tangent))
    return {
        "station": station,
        "center": center,
        "tangent": tangent,
        "normal": normal,
        "binormal": binormal,
    }


def sample_field_on_centerline_pipe_mesh(
    mesh: StructuredMesh,
    field_sampler: FieldSampler,
    *,
    max_points_per_call: int | None = None,
) -> dict[str, np.ndarray | StructuredMesh | str]:
    """Sample a vector magnetic field on a mapped pipe mesh and project it locally.

    The returned components are aligned with the pipe frame:

    - ``B_s``: streamwise component along the centerline tangent
    - ``B_n``: first cross-sectional normal component
    - ``B_b``: second cross-sectional binormal component
    - ``B_perp``: magnitude of the transverse magnetic field
    """

    points = _centerline_points(mesh)
    frames = centerline_pipe_frames(mesh)
    flat = points.reshape(-1, 3)
    field = _sample_in_chunks(field_sampler, flat, max_points_per_call=max_points_per_call)
    field = field.reshape(points.shape)
    tangent = frames["tangent"][:, None, None, :]
    normal = frames["normal"][:, None, None, :]
    binormal = frames["binormal"][:, None, None, :]
    b_s = np.sum(field * tangent, axis=-1)
    b_n = np.sum(field * normal, axis=-1)
    b_b = np.sum(field * binormal, axis=-1)
    b_perp = np.sqrt(b_n**2 + b_b**2)
    b_mag = np.linalg.norm(field, axis=-1)
    return {
        "case": "centerline_pipe_field_sample",
        "mesh": mesh,
        "frames": frames,
        "points": points,
        "field": field,
        "B_s": b_s,
        "B_n": b_n,
        "B_b": b_b,
        "B_perp": b_perp,
        "B_magnitude": b_mag,
    }


def centerline_field_quality_metrics(sample: dict[str, object]) -> dict[str, float | int | bool | str]:
    """Return finite-value, local-component, and cross-section variation metrics."""

    station = np.asarray(sample["frames"]["station"], dtype=float)  # type: ignore[index]
    b_s = np.asarray(sample["B_s"], dtype=float)
    b_perp = np.asarray(sample["B_perp"], dtype=float)
    b_mag = np.asarray(sample["B_magnitude"], dtype=float)
    center_b_s = b_s[:, 0, 0]
    center_b_perp = b_perp[:, 0, 0]
    center_b_mag = b_mag[:, 0, 0]
    finite_fraction = float(np.mean(np.isfinite(b_mag)))
    section = b_mag[:, 1:, :-1] if b_mag.shape[1] > 1 and b_mag.shape[2] > 1 else b_mag
    section_mean = np.mean(np.abs(section), axis=(1, 2))
    section_span = np.max(section, axis=(1, 2)) - np.min(section, axis=(1, 2))
    relative_span = section_span / np.maximum(section_mean, 1.0e-30)
    peak_index = int(np.argmax(center_b_mag))
    peak_perp_index = int(np.argmax(center_b_perp))
    streamwise_fraction = np.abs(center_b_s) / np.maximum(center_b_mag, 1.0e-30)
    transverse_fraction = center_b_perp / np.maximum(center_b_mag, 1.0e-30)
    validation_pass = bool(
        finite_fraction == 1.0
        and float(np.max(center_b_mag)) > 0.0
        and float(np.max(center_b_perp)) > 0.0
        and np.all(np.isfinite(relative_span))
    )
    return {
        "case": str(sample.get("case", "centerline_pipe_field_sample")),
        "station_count": int(station.size),
        "finite_fraction": finite_fraction,
        "peak_centerline_b_magnitude": float(center_b_mag[peak_index]),
        "peak_centerline_b_perp": float(center_b_perp[peak_perp_index]),
        "peak_b_magnitude_station": float(station[peak_index]),
        "peak_b_perp_station": float(station[peak_perp_index]),
        "mean_centerline_b_magnitude": float(np.mean(center_b_mag)),
        "mean_centerline_b_perp": float(np.mean(center_b_perp)),
        "mean_abs_centerline_b_s": float(np.mean(np.abs(center_b_s))),
        "max_streamwise_field_fraction": float(np.max(streamwise_fraction)),
        "max_transverse_field_fraction": float(np.max(transverse_fraction)),
        "max_cross_section_relative_b_span": float(np.max(relative_span)),
        "mean_cross_section_relative_b_span": float(np.mean(relative_span)),
        "validation_pass": validation_pass,
    }


def write_centerline_field_preview(
    sample: dict[str, object],
    out_dir: str | Path,
    *,
    filename_stem: str = "centerline_pipe_field_preview",
    title: str = "Mapped pipe magnetic-field handoff",
) -> list[Path]:
    """Write a QA panel and machine-readable summaries for a local field sample."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _set_field_plot_style()
    metrics = centerline_field_quality_metrics(sample)
    frames = sample["frames"]  # type: ignore[assignment]
    station = np.asarray(frames["station"], dtype=float)  # type: ignore[index]
    center = np.asarray(frames["center"], dtype=float)  # type: ignore[index]
    b_s = np.asarray(sample["B_s"], dtype=float)
    b_perp = np.asarray(sample["B_perp"], dtype=float)
    b_mag = np.asarray(sample["B_magnitude"], dtype=float)
    points = np.asarray(sample["points"], dtype=float)
    peak_index = int(np.argmax(b_perp[:, 0, 0]))

    fig = plt.figure(figsize=(13.6, 7.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.25, 1.0, 0.86])
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_components = fig.add_subplot(gs[0, 1])
    ax_section = fig.add_subplot(gs[1, 1])
    ax_text = fig.add_subplot(gs[:, 2])

    _plot_centerline_colored_by_field(ax3d, center, b_perp[:, 0, 0], label=r"$B_\perp$ [T]")
    ax_components.plot(station, b_perp[:, 0, 0], color="#0f766e", linewidth=2.0, label=r"$B_\perp$")
    ax_components.plot(station, np.abs(b_s[:, 0, 0]), color="#b91c1c", linestyle="--", linewidth=1.8, label=r"$|B_s|$")
    ax_components.plot(station, b_mag[:, 0, 0], color="#334155", linewidth=1.2, alpha=0.75, label=r"$|B|$")
    ax_components.axvline(station[peak_index], color="#64748b", linestyle=":", linewidth=1.0)
    ax_components.set_xlabel("station s [m]")
    ax_components.set_ylabel("centerline field [T]")
    ax_components.set_title("Local field components on pipe centerline")
    ax_components.legend(loc="upper right", fontsize=9, frameon=True)

    _plot_peak_section(ax_section, points, b_mag, peak_index, station[peak_index])
    _plot_field_metrics(ax_text, metrics)
    fig.suptitle(title, fontsize=17)

    png = out / f"{filename_stem}.png"
    pdf = out / f"{filename_stem}.pdf"
    summary = out / f"{filename_stem}_summary.json"
    csv_path = out / f"{filename_stem}_centerline.csv"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    summary.write_text(
        json.dumps({"metrics": metrics, "artifacts": [png.name, pdf.name, csv_path.name]}, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_centerline_csv(csv_path, station, b_s[:, 0, 0], b_perp[:, 0, 0], b_mag[:, 0, 0])
    return [png, pdf, summary, csv_path]


def solve_centerline_pipe_current_closure(
    sample: dict[str, object],
    *,
    mean_velocity: float = 0.2,
    conductivity: float = 7.9e5,
    potential_iterations: int = 250,
    potential_tolerance: float = 1.0e-10,
) -> dict[str, object]:
    """Solve the conservative inductionless current-closure diagnostic.

    This diagnostic reuses the finite-volume pipe current operators from the
    current extruded solver path. It treats the mapped centerline mesh as a
    locally circular pipe with station coordinate ``s`` and solves for the
    potential that cancels ``div(sigma u x B)`` for a prescribed streamwise
    velocity profile. It is a current-closure gate, not a full momentum solve.
    """

    from .fringing import (
        _pipe_conservative_current_diagnostics_3d,
        _pipe_conservative_current_fluxes_3d,
        _pipe_conservative_emf_rhs_3d,
        _pipe_poisson_sparse_3d,
        _station_axial_current_from_fluxes,
    )
    import jax.numpy as jnp

    mesh = sample["mesh"]
    if not isinstance(mesh, StructuredMesh):
        raise ValueError("current closure sample must include the source StructuredMesh")
    station_faces = np.asarray(mesh.x_faces, dtype=float)
    r_faces = np.asarray(mesh.y_faces, dtype=float)
    theta_faces = np.asarray(mesh.z_faces, dtype=float)
    if station_faces.size < 3 or r_faces.size < 3 or theta_faces.size < 3:
        raise ValueError("current closure requires at least two cells in each mapped direction")
    ds = float(np.mean(np.diff(station_faces)))
    dtheta = float(np.mean(np.diff(theta_faces)))
    r_centers = 0.5 * (r_faces[:-1] + r_faces[1:])
    theta_centers = 0.5 * (theta_faces[:-1] + theta_faces[1:])

    b_s = _face_sample_to_cells(np.asarray(sample["B_s"], dtype=float))
    b_n = _face_sample_to_cells(np.asarray(sample["B_n"], dtype=float))
    b_b = _face_sample_to_cells(np.asarray(sample["B_b"], dtype=float))
    theta = theta_centers[None, None, :]
    b_r = b_n * np.cos(theta) + b_b * np.sin(theta)
    b_theta = -b_n * np.sin(theta) + b_b * np.cos(theta)
    velocity = _parabolic_pipe_velocity_profile(
        r_centers,
        theta_centers,
        station_count=station_faces.size - 1,
        mean_velocity=mean_velocity,
        pipe_radius=float(r_faces[-1]),
    )
    uxb_s = np.zeros_like(velocity)
    uxb_r = -velocity * b_theta
    uxb_theta = velocity * b_r

    sigma = jnp.full(velocity.shape, float(conductivity), dtype=float)
    r_faces_j = jnp.asarray(r_faces, dtype=float)
    r_centers_j = jnp.asarray(r_centers, dtype=float)
    uxb_s_j = jnp.asarray(uxb_s, dtype=float)
    uxb_r_j = jnp.asarray(uxb_r, dtype=float)
    uxb_theta_j = jnp.asarray(uxb_theta, dtype=float)
    emf_rhs = _pipe_conservative_emf_rhs_3d(
        sigma,
        uxb_s_j,
        uxb_r_j,
        uxb_theta_j,
        dx=ds,
        r_faces=r_faces_j,
        r_centers=r_centers_j,
        dtheta=dtheta,
    )
    phi, residual, iteration_count, initial_residual = _pipe_poisson_sparse_3d(
        -emf_rhs,
        sigma,
        dx=ds,
        r_faces=r_faces_j,
        r_centers=r_centers_j,
        dtheta=dtheta,
        iterations=int(potential_iterations),
        tolerance=float(potential_tolerance),
    )
    flux_s, flux_r, flux_theta = _pipe_conservative_current_fluxes_3d(
        sigma,
        phi,
        uxb_s_j,
        uxb_r_j,
        uxb_theta_j,
        dx=ds,
        r_faces=r_faces_j,
        r_centers=r_centers_j,
        dtheta=dtheta,
    )
    div_j, wall_current_leakage, boundary_current_residual = _pipe_conservative_current_diagnostics_3d(
        sigma,
        phi,
        uxb_s_j,
        uxb_r_j,
        uxb_theta_j,
        dx=ds,
        r_faces=r_faces_j,
        r_centers=r_centers_j,
        dtheta=dtheta,
    )
    dr = jnp.diff(r_faces_j)
    cell_area = r_centers_j[:, None] * dr[:, None] * dtheta
    axial_current = _station_axial_current_from_fluxes(flux_s, cell_area)
    j_s = 0.5 * (flux_s[1:] + flux_s[:-1])
    j_r = 0.5 * (flux_r[:, 1:, :] + flux_r[:, :-1, :])
    j_theta = 0.5 * (flux_theta + jnp.roll(flux_theta, 1, axis=2))
    station_centers = 0.5 * (station_faces[:-1] + station_faces[1:])
    closure = {
        "case": "centerline_pipe_current_closure",
        "mesh": mesh,
        "station": station_centers,
        "r": r_centers,
        "theta": theta_centers,
        "velocity": velocity,
        "B_s": b_s,
        "B_r": b_r,
        "B_theta": b_theta,
        "phi": np.asarray(phi, dtype=float),
        "J_s": np.asarray(j_s, dtype=float),
        "J_r": np.asarray(j_r, dtype=float),
        "J_theta": np.asarray(j_theta, dtype=float),
        "div_J": np.asarray(div_j, dtype=float),
        "cell_area": np.asarray(cell_area, dtype=float),
        "axial_current": np.asarray(axial_current, dtype=float),
        "wall_current_leakage": np.asarray(wall_current_leakage, dtype=float),
        "boundary_current_residual": np.asarray(boundary_current_residual, dtype=float),
        "potential_residual": float(residual),
        "potential_initial_residual": float(initial_residual),
        "potential_iterations": int(iteration_count),
    }
    closure["metrics"] = centerline_current_closure_metrics(closure)
    closure["pressure_metrics"] = centerline_current_pressure_metrics(closure)
    return closure


def centerline_current_closure_metrics(closure: dict[str, object]) -> dict[str, float | int | bool | str]:
    """Return scalar gates for the mapped-pipe current-closure diagnostic."""

    div_j = np.asarray(closure["div_J"], dtype=float)
    j_s = np.asarray(closure["J_s"], dtype=float)
    j_r = np.asarray(closure["J_r"], dtype=float)
    j_theta = np.asarray(closure["J_theta"], dtype=float)
    wall = np.asarray(closure["wall_current_leakage"], dtype=float)
    boundary = np.asarray(closure["boundary_current_residual"], dtype=float)
    axial = np.asarray(closure["axial_current"], dtype=float)
    max_current = float(np.max(np.sqrt(j_s**2 + j_r**2 + j_theta**2))) if j_s.size else 0.0
    max_div = float(np.max(np.abs(div_j))) if div_j.size else 0.0
    residual = float(closure["potential_residual"])
    initial_residual = float(closure["potential_initial_residual"])
    relative_charge_balance = max_div / max(abs(initial_residual), 1.0e-30)
    r_scale = float(np.max(np.asarray(closure["r"], dtype=float))) if np.asarray(closure["r"]).size else 1.0
    charge_to_current_scale = max_div * max(r_scale, 1.0e-30) / max(max_current, 1.0e-30)
    validation_pass = bool(
        np.isfinite(max_div)
        and np.isfinite(max_current)
        and relative_charge_balance <= 1.0e-8
        and charge_to_current_scale <= 1.0e-7
        and float(np.max(np.abs(boundary))) <= 1.0e-7
    )
    return {
        "case": str(closure.get("case", "centerline_pipe_current_closure")),
        "station_count": int(np.asarray(closure["station"]).size),
        "max_charge_balance_residual": max_div,
        "mean_charge_balance_residual": float(np.mean(np.abs(div_j))) if div_j.size else 0.0,
        "relative_charge_balance_residual": float(relative_charge_balance),
        "charge_balance_to_current_scale": float(charge_to_current_scale),
        "max_current_magnitude": max_current,
        "mean_current_magnitude": float(np.mean(np.sqrt(j_s**2 + j_r**2 + j_theta**2))) if j_s.size else 0.0,
        "max_wall_current_leakage": float(np.max(np.abs(wall))) if wall.size else 0.0,
        "net_boundary_current_residual": float(np.max(np.abs(boundary))) if boundary.size else 0.0,
        "axial_current_span": float(np.max(axial) - np.min(axial)) if axial.size else 0.0,
        "potential_initial_residual": initial_residual,
        "potential_residual": residual,
        "potential_iterations": int(closure["potential_iterations"]),
        "validation_pass": validation_pass,
    }


def centerline_current_pressure_metrics(closure: dict[str, object]) -> dict[str, float | int | bool | str]:
    """Return a streamwise pressure proxy from conservative ``J x B`` fields."""

    station = np.asarray(closure["station"], dtype=float)
    j_r = np.asarray(closure["J_r"], dtype=float)
    j_theta = np.asarray(closure["J_theta"], dtype=float)
    b_r = np.asarray(closure["B_r"], dtype=float)
    b_theta = np.asarray(closure["B_theta"], dtype=float)
    cell_area = np.asarray(closure["cell_area"], dtype=float)
    lorentz_s = j_r * b_theta - j_theta * b_r
    area = np.maximum(np.sum(cell_area), 1.0e-30)
    area_mean_lorentz_s = np.sum(lorentz_s * cell_area[None, :, :], axis=(1, 2)) / area
    pressure_gradient = -area_mean_lorentz_s
    if station.size > 1:
        positive_pressure_gradient = np.maximum(pressure_gradient, 0.0)
        pressure_drop = float(np.trapezoid(positive_pressure_gradient, station))
        signed_pressure_drop = float(np.trapezoid(pressure_gradient, station))
    else:
        pressure_drop = 0.0
        signed_pressure_drop = 0.0
    return {
        "case": "centerline_pipe_jxb_pressure_proxy",
        "station_count": int(station.size),
        "max_abs_lorentz_s": float(np.max(np.abs(lorentz_s))) if lorentz_s.size else 0.0,
        "max_abs_area_mean_lorentz_s": float(np.max(np.abs(area_mean_lorentz_s))) if area_mean_lorentz_s.size else 0.0,
        "mean_pressure_gradient": float(np.mean(pressure_gradient)) if pressure_gradient.size else 0.0,
        "mhd_pressure_drop_proxy_pa": pressure_drop,
        "signed_mhd_pressure_drop_proxy_pa": signed_pressure_drop,
        "mhd_pressure_drop_proxy_kpa": pressure_drop / 1000.0,
        "validation_pass": bool(np.all(np.isfinite(pressure_gradient))),
    }


def write_centerline_current_closure_preview(
    closure: dict[str, object],
    out_dir: str | Path,
    *,
    filename_stem: str = "centerline_pipe_current_closure",
    title: str = "Mapped pipe conservative current-closure diagnostic",
) -> list[Path]:
    """Write current-closure plots and JSON/CSV diagnostics."""

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _set_field_plot_style()
    metrics = centerline_current_closure_metrics(closure)
    pressure_metrics = centerline_current_pressure_metrics(closure)
    station = np.asarray(closure["station"], dtype=float)
    r = np.asarray(closure["r"], dtype=float)
    theta = np.asarray(closure["theta"], dtype=float)
    div_j = np.asarray(closure["div_J"], dtype=float)
    phi = np.asarray(closure["phi"], dtype=float)
    current = np.sqrt(
        np.asarray(closure["J_s"], dtype=float) ** 2
        + np.asarray(closure["J_r"], dtype=float) ** 2
        + np.asarray(closure["J_theta"], dtype=float) ** 2
    )
    charge_by_station = np.max(np.abs(div_j), axis=(1, 2))
    relative_charge_by_station = charge_by_station / max(abs(float(closure["potential_initial_residual"])), 1.0e-30)
    peak_index = int(np.argmax(current.max(axis=(1, 2))))

    fig = plt.figure(figsize=(13.2, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.05, 1.0, 0.86])
    ax_charge = fig.add_subplot(gs[0, 0])
    ax_axial = fig.add_subplot(gs[1, 0])
    ax_phi = fig.add_subplot(gs[0, 1])
    ax_current = fig.add_subplot(gs[1, 1])
    ax_text = fig.add_subplot(gs[:, 2])

    ax_charge.semilogy(station, np.maximum(relative_charge_by_station, 1.0e-30), color="#b91c1c", linewidth=2.0)
    ax_charge.set_xlabel("station s [m]")
    ax_charge.set_ylabel(r"relative max $|\nabla\cdot J|$")
    ax_charge.set_title("Local charge-balance residual relative to EMF source")

    axial = np.asarray(closure["axial_current"], dtype=float)
    ax_axial.plot(station, axial, color="#0f766e", linewidth=2.0)
    ax_axial.set_xlabel("station s [m]")
    ax_axial.set_ylabel(r"$\int J_s\,dA$")
    ax_axial.set_title("Stationwise axial current")

    _plot_polar_cell_contour(ax_phi, r, theta, phi[peak_index], title=f"Potential at peak current, s = {station[peak_index]:.2f} m", label=r"$\phi$")
    _plot_polar_cell_contour(ax_current, r, theta, current[peak_index], title="Current magnitude at peak current", label=r"$|J|$")
    _plot_current_metrics(ax_text, metrics, pressure_metrics)
    fig.suptitle(title, fontsize=17)

    png = out / f"{filename_stem}.png"
    pdf = out / f"{filename_stem}.pdf"
    summary = out / f"{filename_stem}_summary.json"
    csv_path = out / f"{filename_stem}_station_data.csv"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    summary.write_text(
        json.dumps(
            {"metrics": metrics, "pressure_metrics": pressure_metrics, "artifacts": [png.name, pdf.name, csv_path.name]},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_current_closure_csv(csv_path, station, charge_by_station, axial)
    return [png, pdf, summary, csv_path]


def _centerline_points(mesh: StructuredMesh) -> np.ndarray:
    if mesh.point_coordinates is None:
        raise ValueError("centerline field sampling requires mesh.point_coordinates")
    points = np.asarray(mesh.point_coordinates, dtype=float)
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError("point_coordinates must have shape (nx+1, nr+1, ntheta+1, 3)")
    if points.shape[0] != int(mesh.x_faces.size):
        raise ValueError("point coordinate station count must match mesh.x_faces")
    return points


def _unit_vector(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1.0e-14)


def _sample_in_chunks(
    field_sampler: FieldSampler,
    points: np.ndarray,
    *,
    max_points_per_call: int | None,
) -> np.ndarray:
    chunk_size = points.shape[0] if max_points_per_call is None else max(int(max_points_per_call), 1)
    chunks = []
    for start in range(0, points.shape[0], chunk_size):
        chunk = points[start : start + chunk_size]
        sampled = np.asarray(field_sampler(chunk[:, 0], chunk[:, 1], chunk[:, 2]), dtype=float)
        if sampled.shape != (chunk.shape[0], 3):
            raise ValueError("field_sampler must return an array with shape (n_points, 3)")
        chunks.append(sampled)
    return np.concatenate(chunks, axis=0)


def _face_sample_to_cells(values: np.ndarray) -> np.ndarray:
    if values.ndim != 3:
        raise ValueError("face-sampled field component must be three-dimensional")
    return 0.125 * (
        values[:-1, :-1, :-1]
        + values[1:, :-1, :-1]
        + values[:-1, 1:, :-1]
        + values[:-1, :-1, 1:]
        + values[1:, 1:, :-1]
        + values[1:, :-1, 1:]
        + values[:-1, 1:, 1:]
        + values[1:, 1:, 1:]
    )


def _parabolic_pipe_velocity_profile(
    r_centers: np.ndarray,
    theta_centers: np.ndarray,
    *,
    station_count: int,
    mean_velocity: float,
    pipe_radius: float,
) -> np.ndarray:
    radius = float(pipe_radius)
    base = 2.0 * float(mean_velocity) * np.maximum(1.0 - (r_centers / max(radius, 1.0e-30)) ** 2, 0.0)
    return np.broadcast_to(base[None, :, None], (station_count, r_centers.size, theta_centers.size)).copy()


def _set_field_plot_style() -> None:
    global plt, Line3DCollection
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Line3DCollection

    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.5,
        }
    )


def _plot_centerline_colored_by_field(ax, center: np.ndarray, values: np.ndarray, *, label: str) -> None:
    segments = np.stack([center[:-1], center[1:]], axis=1)
    line_values = 0.5 * (values[:-1] + values[1:])
    collection = Line3DCollection(segments, cmap="viridis", linewidth=4.0)
    collection.set_array(line_values)
    collection.set_clim(float(np.min(values)), float(np.max(values)))
    ax.add_collection3d(collection)
    ax.scatter(center[0, 0], center[0, 1], center[0, 2], color="#2563eb", s=32, label="inlet")
    ax.scatter(center[-1, 0], center[-1, 1], center[-1, 2], color="#b91c1c", s=32, label="outlet")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("Blanket pipe route colored by local transverse field")
    ax.view_init(elev=24, azim=-58)
    _equal_3d_axes(ax, center)
    ax.legend(loc="upper left", fontsize=8)
    cbar = plt.colorbar(collection, ax=ax, shrink=0.66, pad=0.04)
    cbar.set_label(label)


def _plot_peak_section(ax, points: np.ndarray, b_mag: np.ndarray, peak_index: int, station: float) -> None:
    section_points = points[peak_index, :, :-1, :]
    values = b_mag[peak_index, :, :-1]
    local_y = np.linalg.norm(section_points - points[peak_index, 0, 0, :][None, None, :], axis=-1)
    theta = np.asarray(np.linspace(0.0, 2.0 * np.pi, values.shape[1], endpoint=False))
    signed_y = local_y * np.cos(theta)[None, :]
    signed_z = local_y * np.sin(theta)[None, :]
    contour = ax.contourf(signed_y, signed_z, values, levels=18, cmap="coolwarm")
    ax.set_aspect("equal")
    ax.set_xlabel("local normal radius [m]")
    ax.set_ylabel("local binormal radius [m]")
    ax.set_title(f"Field magnitude across pipe at peak B_perp, s = {station:.2f} m")
    cbar = plt.colorbar(contour, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label(r"$|B|$ [T]")


def _plot_polar_cell_contour(
    ax,
    r: np.ndarray,
    theta: np.ndarray,
    values: np.ndarray,
    *,
    title: str,
    label: str,
) -> None:
    rr = r[:, None]
    tt = theta[None, :]
    y = rr * np.cos(tt)
    z = rr * np.sin(tt)
    contour = ax.contourf(y, z, values, levels=18, cmap="coolwarm")
    ax.set_aspect("equal")
    ax.set_xlabel("local normal radius [m]")
    ax.set_ylabel("local binormal radius [m]")
    ax.set_title(title)
    cbar = plt.colorbar(contour, ax=ax, shrink=0.9, pad=0.02)
    cbar.set_label(label)


def _plot_field_metrics(ax, metrics: dict[str, float | int | bool | str]) -> None:
    ax.axis("off")
    lines = [
        "Quality gates",
        f"finite fraction: {metrics['finite_fraction']:.3f}",
        f"peak |B|: {metrics['peak_centerline_b_magnitude']:.3e} T",
        f"peak B_perp: {metrics['peak_centerline_b_perp']:.3e} T",
        f"peak B_perp station: {metrics['peak_b_perp_station']:.3f} m",
        f"mean B_perp: {metrics['mean_centerline_b_perp']:.3e} T",
        f"max |B_s|/|B|: {metrics['max_streamwise_field_fraction']:.3f}",
        f"max B_perp/|B|: {metrics['max_transverse_field_fraction']:.3f}",
        f"max section |B| span: {metrics['max_cross_section_relative_b_span']:.3e}",
        f"validation pass: {metrics['validation_pass']}",
        "",
        "Interpretation",
        "This is the solver-facing field handoff:",
        "global B(x,y,z) is sampled on the",
        "mapped pipe mesh and projected into",
        "local streamwise and transverse",
        "components before phi/J assembly.",
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
    )


def _plot_current_metrics(
    ax,
    metrics: dict[str, float | int | bool | str],
    pressure_metrics: dict[str, float | int | bool | str],
) -> None:
    ax.axis("off")
    lines = [
        "Current-closure gates",
        f"max |div J|: {metrics['max_charge_balance_residual']:.3e}",
        f"mean |div J|: {metrics['mean_charge_balance_residual']:.3e}",
        f"relative |div J|: {metrics['relative_charge_balance_residual']:.3e}",
        f"scaled |div J|: {metrics['charge_balance_to_current_scale']:.3e}",
        f"max |J|: {metrics['max_current_magnitude']:.3e}",
        f"wall leakage: {metrics['max_wall_current_leakage']:.3e}",
        f"boundary residual: {metrics['net_boundary_current_residual']:.3e}",
        f"axial-current span: {metrics['axial_current_span']:.3e}",
        f"potential residual: {metrics['potential_residual']:.3e}",
        f"potential iterations: {metrics['potential_iterations']}",
        f"JxB dp proxy: {pressure_metrics['mhd_pressure_drop_proxy_kpa']:.3e} kPa",
        f"validation pass: {metrics['validation_pass']}",
        "",
        "Interpretation",
        "This solves the conservative",
        "inductionless potential equation for",
        "a prescribed streamwise pipe profile.",
        "It gates phi/J assembly before any",
        "curved-pipe momentum validation claim.",
    ]
    ax.text(
        0.02,
        0.98,
        "\n".join(lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
    )


def _equal_3d_axes(ax, xyz: np.ndarray) -> None:
    mins = np.min(xyz, axis=0)
    maxs = np.max(xyz, axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.55 * float(np.max(maxs - mins))
    if radius <= 0.0:
        radius = 1.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def _write_centerline_csv(path: Path, station: np.ndarray, b_s: np.ndarray, b_perp: np.ndarray, b_mag: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["station_m", "B_s_T", "B_perp_T", "B_magnitude_T"])
        for row in zip(station, b_s, b_perp, b_mag, strict=True):
            writer.writerow([f"{value:.12e}" for value in row])


def _write_current_closure_csv(
    path: Path,
    station: np.ndarray,
    charge_by_station: np.ndarray,
    axial_current: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["station_m", "max_abs_div_j", "axial_current"])
        for row in zip(station, charge_by_station, axial_current, strict=True):
            writer.writerow([f"{value:.12e}" for value in row])
