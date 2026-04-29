"""Field sampling and local-frame projections on centerline pipe meshes."""

from __future__ import annotations

from collections.abc import Callable
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np

from .field_models import sample_wham_mirror_field
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


def sample_wham_field_on_centerline_pipe_mesh(
    mesh: StructuredMesh,
    *,
    coil_parameters: dict[str, float | int] | None = None,
    field_scale: float = 1.0,
    max_points_per_call: int | None = None,
) -> dict[str, np.ndarray | StructuredMesh | str]:
    """Sample the WHAM-like mirror field on a mapped pipe mesh in local coordinates."""

    params = dict(coil_parameters or {})

    def sampler(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        field = sample_wham_mirror_field(
            x,
            y,
            z,
            coil_separation=float(params.get("coil_separation", 1.96)),
            current_scale=float(params.get("current_scale", 100323.62459546926)),
            inner_radius=float(params.get("inner_radius", 0.043)),
            outer_radius=float(params.get("outer_radius", 0.365)),
            coil_axial_thickness=float(params.get("coil_axial_thickness", 0.1144)),
            radial_loops=int(params.get("radial_loops", 12)),
            axial_loops=int(params.get("axial_loops", 4)),
            x_offset=float(params.get("x_offset", 0.0)),
            y_offset=float(params.get("y_offset", 0.0)),
            z_offset=float(params.get("z_offset", 0.0)),
        )
        return field_scale * np.asarray(field, dtype=float)

    sample = sample_field_on_centerline_pipe_mesh(
        mesh,
        sampler,
        max_points_per_call=max_points_per_call,
    )
    sample["case"] = "wham_centerline_pipe_field_sample"
    sample["field_scale"] = float(field_scale)
    sample["coil_parameters"] = params
    return sample


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


def _set_field_plot_style() -> None:
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
