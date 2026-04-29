"""Geometry helpers for WHAM liquid-metal blanket routing previews."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from .mesh import StructuredMesh, centerline_pipe_mesh_quality_metrics


@dataclass(frozen=True)
class WhamBlanketLoop:
    """Dimensional parameters for a circular pipe routed around WHAM's central cell."""

    pipe_radius: float = 0.12
    bend_radius: float = 0.90
    entry_length: float = 1.35
    central_cell_radius: float = 0.42
    coil_separation: float = 1.96
    coil_inner_radius: float = 0.043
    coil_outer_radius: float = 0.365
    coil_axial_thickness: float = 0.1144
    z_offset: float = 0.0


def build_wham_blanket_centerline(
    geometry: WhamBlanketLoop | None = None,
    *,
    straight_points: int = 72,
    bend_points: int = 144,
) -> dict[str, np.ndarray]:
    """Return a U-shaped blanket pipe centerline around the mirror central cell.

    The WHAM mirror axis is the global ``z`` axis. The preview routes a circular
    pipe in the midplane: it enters from negative ``x`` on the lower side of the
    central cell, bends around the outboard side, and returns on the upper side.
    This is a geometry-review path, not yet a solver mesh.
    """

    spec = geometry or WhamBlanketLoop()
    if spec.pipe_radius <= 0.0:
        raise ValueError("pipe_radius must be positive")
    if spec.bend_radius <= spec.central_cell_radius + spec.pipe_radius:
        raise ValueError("bend_radius must exceed central_cell_radius + pipe_radius")
    if spec.entry_length <= 0.0:
        raise ValueError("entry_length must be positive")
    straight_count = max(int(straight_points), 2)
    bend_count = max(int(bend_points), 5)

    x_start = -(spec.entry_length + spec.bend_radius)
    x_in = np.linspace(x_start, 0.0, straight_count)
    y_in = -spec.bend_radius * np.ones_like(x_in)
    z_in = spec.z_offset * np.ones_like(x_in)

    theta = np.linspace(-0.5 * np.pi, 0.5 * np.pi, bend_count)
    x_bend = spec.bend_radius * np.cos(theta)
    y_bend = spec.bend_radius * np.sin(theta)
    z_bend = spec.z_offset * np.ones_like(theta)

    x_out = np.linspace(0.0, x_start, straight_count)
    y_out = spec.bend_radius * np.ones_like(x_out)
    z_out = spec.z_offset * np.ones_like(x_out)

    x = np.concatenate([x_in, x_bend[1:-1], x_out])
    y = np.concatenate([y_in, y_bend[1:-1], y_out])
    z = np.concatenate([z_in, z_bend[1:-1], z_out])
    points = np.column_stack([x, y, z])
    segment_lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    station = np.concatenate([[0.0], np.cumsum(segment_lengths)])
    return {"x": x, "y": y, "z": z, "station": station}


def tube_surface_from_centerline(
    centerline: dict[str, np.ndarray],
    *,
    radius: float,
    circumferential_points: int = 40,
) -> dict[str, np.ndarray]:
    """Build a circular tube surface around a sampled 3D centerline."""

    if radius <= 0.0:
        raise ValueError("radius must be positive")
    n_theta = max(int(circumferential_points), 8)
    centers = np.column_stack(
        [
            np.asarray(centerline["x"], dtype=float),
            np.asarray(centerline["y"], dtype=float),
            np.asarray(centerline["z"], dtype=float),
        ]
    )
    if centers.ndim != 2 or centers.shape[0] < 3:
        raise ValueError("centerline must contain at least three points")

    tangent = np.gradient(centers, axis=0)
    tangent_norm = np.linalg.norm(tangent, axis=1, keepdims=True)
    tangent = tangent / np.maximum(tangent_norm, 1.0e-14)

    reference = np.tile(np.array([0.0, 0.0, 1.0]), (centers.shape[0], 1))
    near_parallel = np.abs(np.sum(tangent * reference, axis=1)) > 0.94
    reference[near_parallel] = np.array([0.0, 1.0, 0.0])
    normal = np.cross(reference, tangent)
    normal = normal / np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1.0e-14)
    binormal = np.cross(tangent, normal)

    phi = np.linspace(0.0, 2.0 * np.pi, n_theta, endpoint=True)
    cos_phi = np.cos(phi)[None, :, None]
    sin_phi = np.sin(phi)[None, :, None]
    surface = centers[:, None, :] + radius * (normal[:, None, :] * cos_phi + binormal[:, None, :] * sin_phi)
    return {"x": surface[..., 0], "y": surface[..., 1], "z": surface[..., 2]}


def wham_blanket_clearance_metrics(
    centerline: dict[str, np.ndarray],
    geometry: WhamBlanketLoop | None = None,
) -> dict[str, float]:
    """Return simple length and clearance metrics for the previewed blanket pipe."""

    spec = geometry or WhamBlanketLoop()
    x = np.asarray(centerline["x"], dtype=float)
    y = np.asarray(centerline["y"], dtype=float)
    station = np.asarray(centerline["station"], dtype=float)
    radial_distance = np.sqrt(x**2 + y**2)
    centerline_to_cell_clearance = float(np.min(radial_distance - spec.central_cell_radius))
    tube_to_cell_clearance = centerline_to_cell_clearance - spec.pipe_radius
    return {
        "path_length": float(station[-1]),
        "pipe_radius": float(spec.pipe_radius),
        "bend_radius": float(spec.bend_radius),
        "entry_length": float(spec.entry_length),
        "central_cell_radius": float(spec.central_cell_radius),
        "centerline_to_cell_clearance": centerline_to_cell_clearance,
        "tube_to_cell_clearance": float(tube_to_cell_clearance),
        "coil_separation": float(spec.coil_separation),
    }


def write_wham_blanket_geometry_preview(
    centerline: dict[str, np.ndarray],
    out_dir: str | Path,
    *,
    geometry: WhamBlanketLoop | None = None,
    filename_stem: str = "wham_blanket_geometry_preview",
) -> list[Path]:
    """Write a publication-style WHAM blanket geometry preview panel."""

    spec = geometry or WhamBlanketLoop()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tube = tube_surface_from_centerline(centerline, radius=spec.pipe_radius)
    metrics = wham_blanket_clearance_metrics(centerline, spec)

    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.5,
        }
    )

    fig = plt.figure(figsize=(12.5, 7.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.35, 0.9, 0.82])
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_top = fig.add_subplot(gs[0, 1])
    ax_side = fig.add_subplot(gs[1, 1])
    ax_text = fig.add_subplot(gs[:, 2])

    _plot_blanket_3d(ax3d, centerline, tube, spec)
    _plot_blanket_top_view(ax_top, centerline, spec)
    _plot_blanket_side_view(ax_side, centerline, spec)
    _plot_blanket_metrics(ax_text, spec, metrics)

    fig.suptitle("LMX WHAM liquid-metal blanket pipe geometry preview", fontsize=17)
    png = out / f"{filename_stem}.png"
    pdf = out / f"{filename_stem}.pdf"
    summary = out / f"{filename_stem}_summary.json"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    summary.write_text(
        json.dumps(
            {
                "case": "wham_blanket_geometry_preview",
                "geometry": asdict(spec),
                "metrics": metrics,
                "artifacts": [png.name, pdf.name],
                "notes": (
                    "Geometry-only design review: a circular pipe enters from the negative-x side, "
                    "wraps around the WHAM central-cell envelope in the z=0 midplane, and returns. "
                    "No MHD solve is claimed by this artifact."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return [png, pdf, summary]


def write_centerline_pipe_mesh_preview(
    mesh: StructuredMesh,
    out_dir: str | Path,
    *,
    filename_stem: str = "centerline_pipe_mesh_preview",
    title: str = "Centerline pipe mapped mesh preview",
) -> list[Path]:
    """Write a mesh-QA panel for a mapped pipe following a 3D centerline."""

    if mesh.point_coordinates is None:
        raise ValueError("centerline pipe mesh preview requires point_coordinates")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    points = np.asarray(mesh.point_coordinates, dtype=float)
    center = points[:, 0, 0, :]
    outer = points[:, -1, :, :]
    metrics = centerline_pipe_mesh_quality_metrics(mesh)

    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.5,
        }
    )
    fig = plt.figure(figsize=(12.8, 5.8), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.35, 1.0, 0.75])
    ax3d = fig.add_subplot(gs[0, 0], projection="3d")
    ax_spacing = fig.add_subplot(gs[0, 1])
    ax_text = fig.add_subplot(gs[0, 2])

    ax3d.plot_surface(
        outer[:, :, 0],
        outer[:, :, 1],
        outer[:, :, 2],
        color="#0f766e",
        alpha=0.24,
        linewidth=0.0,
        shade=False,
    )
    ax3d.plot(center[:, 0], center[:, 1], center[:, 2], color="#111827", linewidth=2.2)
    for station_index, color in ((0, "#2563eb"), (points.shape[0] // 2, "#f97316"), (-1, "#dc2626")):
        section = points[station_index, -1, :, :]
        ax3d.plot(section[:, 0], section[:, 1], section[:, 2], color=color, linewidth=1.4)
    ax3d.scatter([center[0, 0]], [center[0, 1]], [center[0, 2]], color="#2563eb", s=32, label="inlet")
    ax3d.scatter([center[-1, 0]], [center[-1, 1]], [center[-1, 2]], color="#dc2626", s=32, label="outlet")
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("")
    ax3d.set_title("Mapped pipe surface and stations")
    ax3d.view_init(elev=24, azim=-42)
    _set_mesh_equal_3d(ax3d, center, float(metrics["target_radius"]))
    ax3d.legend(loc="upper left", fontsize=9)

    station_edges = np.asarray(mesh.x_faces, dtype=float)
    station_spacing = np.diff(station_edges)
    outer_radius = np.linalg.norm(outer - center[:, None, :], axis=-1)
    radius_error = np.max(np.abs(outer_radius - float(metrics["target_radius"])), axis=1)
    ax_spacing.plot(station_edges[:-1], station_spacing, color="#0f766e", marker="o", markersize=3, label="station spacing")
    ax_spacing.set_xlabel("station s [m]")
    ax_spacing.set_ylabel("Δs [m]")
    ax_spacing.ticklabel_format(axis="y", style="plain", useOffset=False)
    if station_spacing.size:
        spacing_pad = max(1.0e-4, 0.05 * float(np.ptp(station_spacing)))
        ax_spacing.set_ylim(float(np.min(station_spacing)) - spacing_pad, float(np.max(station_spacing)) + spacing_pad)
    ax_spacing_error = ax_spacing.twinx()
    ax_spacing_error.plot(station_edges, 1.0e15 * radius_error, color="#b91c1c", linestyle="--", label="radius error")
    ax_spacing_error.set_ylabel(r"radius error [$10^{-15}$ m]")
    ax_spacing.set_title("Mesh spacing and radius preservation")
    lines = ax_spacing.get_lines() + ax_spacing_error.get_lines()
    ax_spacing.legend(lines, [line.get_label() for line in lines], loc="upper right", fontsize=8)

    ax_text.axis("off")
    text = "\n".join(
        [
            "Mesh QA",
            f"geometry = {metrics['geometry']}",
            f"stations = {metrics['station_count']}",
            f"radial faces = {metrics['radial_face_count']}",
            f"theta faces = {metrics['theta_face_count']}",
            f"cells = {metrics['cell_count']}",
            f"path length = {metrics['path_length']:.2f} m",
            f"target radius = {metrics['target_radius']:.3f} m",
            f"min Δs = {metrics['min_station_spacing']:.3f} m",
            f"max Δs = {metrics['max_station_spacing']:.3f} m",
            f"max radius error = {metrics['max_radius_error']:.2e} m",
            f"theta closure = {metrics['max_theta_closure_error']:.2e} m",
            f"validation pass = {metrics['validation_pass']}",
        ]
    )
    ax_text.text(
        0.02,
        0.98,
        text,
        va="top",
        ha="left",
        fontsize=11,
        linespacing=1.32,
        bbox={"boxstyle": "round,pad=0.55", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
    )
    fig.suptitle(title, fontsize=16)
    png = out / f"{filename_stem}.png"
    pdf = out / f"{filename_stem}.pdf"
    summary = out / f"{filename_stem}_summary.json"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    summary.write_text(
        json.dumps(
            {
                "case": filename_stem,
                "metrics": metrics,
                "artifacts": [png.name, pdf.name],
                "notes": "Mapped centerline-pipe mesh QA artifact; no MHD solve is claimed by this mesh preview.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return [png, pdf, summary]


def _plot_blanket_3d(ax, centerline: dict[str, np.ndarray], tube: dict[str, np.ndarray], spec: WhamBlanketLoop) -> None:
    x = np.asarray(centerline["x"], dtype=float)
    y = np.asarray(centerline["y"], dtype=float)
    z = np.asarray(centerline["z"], dtype=float)
    ax.plot_surface(
        tube["x"],
        tube["y"],
        tube["z"],
        rstride=3,
        cstride=3,
        color="#0f766e",
        alpha=0.84,
        linewidth=0.15,
        edgecolor="#064e3b",
        shade=True,
    )
    ax.plot(x, y, z, color="#022c22", linewidth=1.8, label="pipe centerline")
    _draw_wham_coils(ax, spec)
    _draw_central_cell(ax, spec)
    ax.scatter([x[0]], [y[0]], [z[0]], color="#2563eb", s=38, label="inlet")
    ax.scatter([x[-1]], [y[-1]], [z[-1]], color="#dc2626", s=38, label="outlet")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title("3D route around mirror central cell")
    ax.view_init(elev=24, azim=-42)
    _set_equal_3d(ax, x, y, z, spec)
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=9)


def _plot_blanket_top_view(ax, centerline: dict[str, np.ndarray], spec: WhamBlanketLoop) -> None:
    x = np.asarray(centerline["x"], dtype=float)
    y = np.asarray(centerline["y"], dtype=float)
    ax.plot(x, y, color="#0f766e", linewidth=2.4)
    ax.fill(
        spec.central_cell_radius * np.cos(np.linspace(0.0, 2.0 * np.pi, 160)),
        spec.central_cell_radius * np.sin(np.linspace(0.0, 2.0 * np.pi, 160)),
        color="#f59e0b",
        alpha=0.15,
        label="central-cell clearance envelope",
    )
    ax.add_patch(plt.Circle((0.0, 0.0), spec.bend_radius, fill=False, color="#94a3b8", linestyle="--", linewidth=1.0))
    ax.annotate("flow in", xy=(x[14], y[14]), xytext=(x[14] - 0.7, y[14] - 0.32), arrowprops={"arrowstyle": "->", "color": "#1d4ed8"}, color="#1d4ed8")
    ax.annotate("return", xy=(x[-16], y[-16]), xytext=(x[-16] - 0.85, y[-16] + 0.32), arrowprops={"arrowstyle": "->", "color": "#b91c1c"}, color="#b91c1c")
    ax.set_title("Top view: pump-loop routing")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8)


def _plot_blanket_side_view(ax, centerline: dict[str, np.ndarray], spec: WhamBlanketLoop) -> None:
    x = np.asarray(centerline["x"], dtype=float)
    z = np.asarray(centerline["z"], dtype=float)
    ax.plot(x, z, color="#0f766e", linewidth=2.2, label="pipe centerline")
    for coil_z, label in ((-0.5 * spec.coil_separation, "mirror coil"), (0.5 * spec.coil_separation, None)):
        ax.axhspan(
            coil_z - 0.5 * spec.coil_axial_thickness,
            coil_z + 0.5 * spec.coil_axial_thickness,
            color="#64748b",
            alpha=0.18,
            label=label,
        )
    ax.axhline(0.0, color="#111827", linewidth=0.9, linestyle=":", label="midplane")
    ax.set_title("Side view: midplane routing between mirror coils")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_ylim(-0.65 * spec.coil_separation, 0.65 * spec.coil_separation)
    ax.legend(loc="upper right", fontsize=8)


def _plot_blanket_metrics(ax, spec: WhamBlanketLoop, metrics: dict[str, float]) -> None:
    ax.axis("off")
    text = "\n".join(
        [
            "Design parameters",
            f"pipe radius = {spec.pipe_radius:.2f} m",
            f"bend radius = {spec.bend_radius:.2f} m",
            f"straight entry length = {spec.entry_length:.2f} m",
            f"central-cell envelope radius = {spec.central_cell_radius:.2f} m",
            f"coil separation = {spec.coil_separation:.2f} m",
            "",
            "Preview metrics",
            f"path length = {metrics['path_length']:.2f} m",
            f"tube-to-cell clearance = {metrics['tube_to_cell_clearance']:.2f} m",
            "",
            "Assumption",
            "The pipe is circular and lies in the",
            "WHAM midplane for this first design",
            "review. Simulation setup follows after",
            "the route and clearances are approved.",
        ]
    )
    ax.text(
        0.02,
        0.98,
        text,
        va="top",
        ha="left",
        fontsize=12,
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.6", "facecolor": "#f8fafc", "edgecolor": "#cbd5e1"},
    )


def _draw_wham_coils(ax, spec: WhamBlanketLoop) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 160)
    for zc in (-0.5 * spec.coil_separation, 0.5 * spec.coil_separation):
        for radius, alpha, linewidth in (
            (spec.coil_inner_radius, 0.85, 1.1),
            (spec.coil_outer_radius, 0.85, 1.5),
        ):
            ax.plot(radius * np.cos(theta), radius * np.sin(theta), zc * np.ones_like(theta), color="#7f1d1d", alpha=alpha, linewidth=linewidth)
        verts = []
        for start, stop in zip(theta[:-1], theta[1:]):
            verts.append(
                [
                    (spec.coil_inner_radius * np.cos(start), spec.coil_inner_radius * np.sin(start), zc),
                    (spec.coil_outer_radius * np.cos(start), spec.coil_outer_radius * np.sin(start), zc),
                    (spec.coil_outer_radius * np.cos(stop), spec.coil_outer_radius * np.sin(stop), zc),
                    (spec.coil_inner_radius * np.cos(stop), spec.coil_inner_radius * np.sin(stop), zc),
                ]
            )
        collection = Poly3DCollection(verts, facecolor="#ef4444", edgecolor="none", alpha=0.10)
        ax.add_collection3d(collection)


def _draw_central_cell(ax, spec: WhamBlanketLoop) -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 72)
    z = np.linspace(-0.5 * spec.coil_separation, 0.5 * spec.coil_separation, 8)
    tt, zz = np.meshgrid(theta, z, indexing="ij")
    xx = spec.central_cell_radius * np.cos(tt)
    yy = spec.central_cell_radius * np.sin(tt)
    ax.plot_surface(xx, yy, zz, color="#f59e0b", alpha=0.075, linewidth=0.0, shade=False)
    ax.plot(
        spec.central_cell_radius * np.cos(theta),
        spec.central_cell_radius * np.sin(theta),
        np.zeros_like(theta),
        color="#d97706",
        linestyle="--",
        linewidth=1.2,
    )


def _set_equal_3d(ax, x: np.ndarray, y: np.ndarray, z: np.ndarray, spec: WhamBlanketLoop) -> None:
    xlim = (min(float(np.min(x)), -spec.central_cell_radius) - 0.25, max(float(np.max(x)), spec.central_cell_radius) + 0.25)
    ylim = (min(float(np.min(y)), -spec.bend_radius) - 0.35, max(float(np.max(y)), spec.bend_radius) + 0.35)
    zlim = (-0.65 * spec.coil_separation, 0.65 * spec.coil_separation)
    ranges = np.array([xlim[1] - xlim[0], ylim[1] - ylim[0], zlim[1] - zlim[0]])
    centers = np.array([sum(xlim) / 2.0, sum(ylim) / 2.0, sum(zlim) / 2.0])
    radius = 0.5 * float(np.max(ranges))
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    try:
        ax.set_box_aspect((1.0, 1.0, 0.7))
    except Exception:  # pragma: no cover - compatibility with older Matplotlib.
        pass


def _set_mesh_equal_3d(ax, center: np.ndarray, radius_padding: float) -> None:
    xyz_min = np.min(center, axis=0) - 2.0 * radius_padding
    xyz_max = np.max(center, axis=0) + 2.0 * radius_padding
    ranges = xyz_max - xyz_min
    plot_radius = 0.5 * float(np.max(ranges))
    midpoint = 0.5 * (xyz_min + xyz_max)
    ax.set_xlim(midpoint[0] - plot_radius, midpoint[0] + plot_radius)
    ax.set_ylim(midpoint[1] - plot_radius, midpoint[1] + plot_radius)
    ax.set_zlim(midpoint[2] - plot_radius, midpoint[2] + plot_radius)
    try:
        ax.set_box_aspect((1.0, 1.0, 0.55))
    except Exception:  # pragma: no cover - compatibility with older Matplotlib.
        pass
