from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import ScalarFormatter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

from .core import Solution
from .mesh import StructuredMesh
from .validation import extract_midplane_profile


def _set_plot_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.5,
            "grid.color": "#4f4f4f",
            "legend.frameon": True,
            "legend.framealpha": 0.92,
            "legend.facecolor": "white",
            "legend.edgecolor": "#cbd5e1",
            "legend.fontsize": 12,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "lines.linewidth": 2.0,
        }
    )


def _plot_field(ax: plt.Axes, solution: Solution, field: jnp.ndarray, *, title: str, cmap: str) -> None:
    mesh = solution.mesh
    field_min = float(jnp.min(field))
    field_max = float(jnp.max(field))
    if field_min >= 0.0:
        cmap = "magma"
        norm = colors.Normalize(vmin=field_min, vmax=max(field_max, field_min + 1e-12))
    elif field_max <= 0.0:
        cmap = "magma_r"
        norm = colors.Normalize(vmin=min(field_min, field_max - 1e-12), vmax=field_max)
    else:
        vmax = float(jnp.max(jnp.abs(field)))
        vmax = max(vmax, 1e-12)
        norm = colors.TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    image = ax.pcolormesh(
        mesh.z_faces,
        mesh.y_faces,
        field,
        shading="auto",
        cmap=cmap,
        norm=norm,
    )
    ax.set_title(title)
    ax.set_xlabel("z")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)


def _plot_profile(
    ax: plt.Axes,
    coordinate: jnp.ndarray,
    values: jnp.ndarray,
    *,
    axis_name: str,
    title: str,
    reference_coordinate: jnp.ndarray | None = None,
    reference_values: jnp.ndarray | None = None,
    reference_label: str | None = None,
) -> None:
    coord_scale = float(jnp.max(jnp.abs(coordinate)))
    coord_scale = coord_scale if coord_scale > 0.0 else 1.0
    value_scale = float(jnp.max(jnp.abs(values)))
    value_scale = value_scale if value_scale > 0.0 else 1.0
    ax.plot(coordinate / coord_scale, values / value_scale, color="#0f766e", label="LMX")
    if reference_coordinate is not None and reference_values is not None:
        ref_coord_scale = float(jnp.max(jnp.abs(reference_coordinate)))
        ref_coord_scale = ref_coord_scale if ref_coord_scale > 0.0 else 1.0
        ref_value_scale = float(jnp.max(jnp.abs(reference_values)))
        ref_value_scale = ref_value_scale if ref_value_scale > 0.0 else 1.0
        ax.plot(
            reference_coordinate / ref_coord_scale,
            reference_values / ref_value_scale,
            color="#b45309",
            linestyle="--",
            label=reference_label or "Reference",
        )
    ax.set_title(title)
    ax.set_xlabel(f"Normalized {axis_name}")
    ax.set_ylabel("Normalized velocity")
    ax.set_xlim(-1.02, 1.02)
    ax.legend()


def write_case_overview_plots(
    solution: Solution,
    out_dir: str | Path,
    *,
    case_title: str,
    y_reference_coordinate: jnp.ndarray | None = None,
    y_reference_values: jnp.ndarray | None = None,
    z_reference_coordinate: jnp.ndarray | None = None,
    z_reference_values: jnp.ndarray | None = None,
    reference_label: str = "Reference",
) -> list[Path]:
    _set_plot_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_profile = extract_midplane_profile(solution, axis="y", fluid_only=True)
    z_profile = extract_midplane_profile(solution, axis="z", fluid_only=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
    fig.suptitle(case_title, fontsize=16, y=1.02)

    _plot_field(axes[0, 0], solution, solution.state.u, title="Velocity u", cmap="RdBu_r")
    _plot_field(axes[0, 1], solution, solution.state.phi, title="Electric potential φ", cmap="PuOr_r")
    _plot_profile(
        axes[1, 0],
        y_profile["y"],
        y_profile["u"],
        axis_name="y",
        title="Midplane y profile",
        reference_coordinate=y_reference_coordinate,
        reference_values=y_reference_values,
        reference_label=reference_label,
    )
    _plot_profile(
        axes[1, 1],
        z_profile["z"],
        z_profile["u"],
        axis_name="z",
        title="Midplane z profile",
        reference_coordinate=z_reference_coordinate,
        reference_values=z_reference_values,
        reference_label=reference_label,
    )

    png_path = out_dir / "overview.png"
    pdf_path = out_dir / "overview.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    diagnostics_paths: list[Path] = []
    if solution.diagnostics.time_history.size > 0:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
        time_history = solution.diagnostics.time_history
        axes[0].plot(time_history, solution.diagnostics.u_max_history, color="#1d4ed8", label="max |u|")
        if solution.diagnostics.current_max_history.size:
            axes[0].plot(time_history, solution.diagnostics.current_max_history, color="#b91c1c", label="max |J|")
        if solution.diagnostics.lorentz_max_history.size:
            axes[0].plot(time_history, solution.diagnostics.lorentz_max_history, color="#6d28d9", label="max |J×B|")
        axes[0].set_title("Trace magnitudes")
        axes[0].set_xlabel("time")
        axes[0].set_ylabel("magnitude")
        axes[0].legend()

        axes[1].plot(time_history, solution.diagnostics.residual_history, color="#0f766e", label="velocity residual")
        if solution.diagnostics.potential_residual_history.size:
            axes[1].plot(
                time_history,
                solution.diagnostics.potential_residual_history,
                color="#b45309",
                label="potential residual",
            )
        axes[1].set_title("Solver residuals")
        axes[1].set_xlabel("time")
        axes[1].set_ylabel("residual")
        axes[1].set_yscale("log")
        axes[1].legend()

        diag_png = out_dir / "diagnostics.png"
        diag_pdf = out_dir / "diagnostics.pdf"
        fig.savefig(diag_png, bbox_inches="tight")
        fig.savefig(diag_pdf, bbox_inches="tight")
        plt.close(fig)
        diagnostics_paths.extend([diag_png, diag_pdf])

    return [png_path, pdf_path, *diagnostics_paths]


def _safe_writer_candidates() -> list[tuple[str, str]]:
    return [("gif", "pillow")]


def _format_time_with_units(value: float) -> str:
    magnitude = abs(value)
    if magnitude < 1.0e-3:
        return f"{value * 1.0e6:.1f} μs"
    if magnitude < 1.0:
        return f"{value * 1.0e3:.1f} ms"
    return f"{value:.2f} s"


def _add_fluid_outline(ax: plt.Axes, mesh: StructuredMesh, fluid_mask: jnp.ndarray | None) -> None:
    if fluid_mask is None:
        return
    mask = np.asarray(fluid_mask, dtype=float)
    try:
        ax.contour(
            np.asarray(mesh.z_centers),
            np.asarray(mesh.y_centers),
            mask,
            levels=[0.5],
            colors="#111827",
            linewidths=1.1,
            alpha=0.9,
        )
    except ValueError:
        return


def _add_layer_annotations(ax: plt.Axes, mesh: StructuredMesh, fluid_mask: jnp.ndarray | None, *, show_side_layers: bool) -> None:
    y0, y1 = float(mesh.y_faces[0]), float(mesh.y_faces[-1])
    z0, z1 = float(mesh.z_faces[0]), float(mesh.z_faces[-1])
    annotation_style = {
        "bbox": {"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.92},
        "arrowprops": {"arrowstyle": "->", "color": "#111827", "lw": 1.0},
        "fontsize": 11,
        "color": "#111827",
    }
    ax.annotate(
        "Hartmann\nlayers",
        xy=(0.5, 0.94),
        xycoords="axes fraction",
        xytext=(0.5, 0.985),
        textcoords="axes fraction",
        ha="center",
        va="top",
        **annotation_style,
    )
    ax.annotate(
        "",
        xy=(0.5, 0.06),
        xycoords="axes fraction",
        xytext=(0.5, 0.015),
        textcoords="axes fraction",
        **{key: value for key, value in annotation_style.items() if key in {"arrowprops"}},
    )
    if show_side_layers:
        ax.annotate(
            "Side layers",
            xy=(0.93, 0.5),
            xycoords="axes fraction",
            xytext=(1.03, 0.5),
            textcoords="axes fraction",
            ha="left",
            va="center",
            **annotation_style,
        )
        ax.annotate(
            "",
            xy=(0.07, 0.5),
            xycoords="axes fraction",
            xytext=(-0.03, 0.5),
            textcoords="axes fraction",
            **{key: value for key, value in annotation_style.items() if key in {"arrowprops"}},
        )
    _add_fluid_outline(ax, mesh, fluid_mask)


def _movie_field_stack(
    frames: list[dict[str, object]],
    *,
    field_mode: str,
) -> tuple[list[np.ndarray], list[float], str, str]:
    frame_u = [jnp.asarray(frame["u"]) for frame in frames]
    frame_peaks = [max(float(jnp.max(jnp.abs(field))), 1e-12) for field in frame_u]
    if field_mode == "raw":
        return [np.asarray(field) for field in frame_u], frame_peaks, "Velocity u", "u"
    if field_mode == "bulk_deviation":
        display_fields: list[np.ndarray] = []
        for frame, field in zip(frames, frame_u, strict=True):
            fluid_mask = jnp.asarray(frame.get("fluid_mask", jnp.ones_like(field, dtype=bool))).astype(bool)
            fluid_values = jnp.where(fluid_mask, field, 0.0)
            fluid_count = jnp.maximum(jnp.sum(fluid_mask), 1)
            bulk_mean = jnp.sum(fluid_values) / fluid_count
            display_fields.append(np.asarray(jnp.where(fluid_mask, field - bulk_mean, 0.0)))
        return display_fields, frame_peaks, "Velocity deviation", "u - <u>_fluid"
    raise ValueError(f"Unsupported field_mode {field_mode!r}")


def write_transient_movies(
    frames: list[dict[str, object]],
    out_dir: str | Path,
    *,
    case_title: str,
    fps: int = 6,
    field_mode: str = "raw",
    output_stem: str = "hunt_velocity",
    include_2d: bool = True,
    include_3d: bool = True,
) -> list[Path]:
    if not frames:
        return []
    _set_plot_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = frames[0]["mesh"]
    display_fields, frame_peaks, movie_label, colorbar_label = _movie_field_stack(frames, field_mode=field_mode)
    u_stack = jnp.asarray(np.stack(display_fields))
    stack_min = float(jnp.min(u_stack))
    stack_max = float(jnp.max(u_stack))
    stack_abs_max = max(float(jnp.max(jnp.abs(u_stack))), 1e-12)
    use_normalized_positive = stack_min >= 0.0 or stack_max <= 0.0
    if use_normalized_positive:
        cmap = "magma"
        norm = colors.Normalize(vmin=0.0, vmax=1.0)
    else:
        cmap = "RdBu_r"
        norm = colors.TwoSlopeNorm(vmin=-stack_abs_max, vcenter=0.0, vmax=stack_abs_max)

    times = [float(frame["time"]) for frame in frames]
    case = frames[0].get("case")
    case_name = str(getattr(case, "name", "")).lower()
    show_side_layers = "hunt" in case_name or "shercliff" in case_name or bool(getattr(getattr(case, "geometry", None), "target_side_layer", None))

    def _movie_field(index: int) -> np.ndarray:
        field = np.asarray(display_fields[index])
        if use_normalized_positive:
            return field / frame_peaks[index]
        return field

    effective_label = f"Normalized {movie_label.lower()}" if use_normalized_positive else movie_label
    effective_colorbar_label = f"{colorbar_label} / max|u(t)|" if use_normalized_positive else colorbar_label
    outputs: list[Path] = []

    fig2d = None
    anim2d = None
    if include_2d:
        fig2d, ax2d = plt.subplots(figsize=(6.1, 5.2), constrained_layout=True)
        image = ax2d.pcolormesh(mesh.z_faces, mesh.y_faces, _movie_field(0), shading="auto", cmap=cmap, norm=norm)
        ax2d.set_xlabel("z")
        ax2d.set_ylabel("y")
        ax2d.set_title(f"{case_title}\n2D {effective_label.lower()}")
        ax2d.set_aspect("equal")
        _add_layer_annotations(ax2d, mesh, frames[0].get("fluid_mask"), show_side_layers=show_side_layers)
        annotation_bbox = {"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.85}
        time_text = ax2d.text(0.02, 0.98, "", transform=ax2d.transAxes, ha="left", va="top", bbox=annotation_bbox)
        peak_text = ax2d.text(0.98, 0.98, "", transform=ax2d.transAxes, ha="right", va="top", bbox=annotation_bbox)
        plt.colorbar(image, ax=ax2d, fraction=0.046, pad=0.04, label=effective_colorbar_label)
        contour_state: list[object] = []

        def update_2d(index: int):
            field = _movie_field(index)
            image.set_array(field.ravel())
            if contour_state:
                previous_contour = contour_state.pop()
                if hasattr(previous_contour, "remove"):
                    previous_contour.remove()
                elif hasattr(previous_contour, "collections"):
                    for collection in previous_contour.collections:
                        collection.remove()
            if use_normalized_positive:
                contour_levels = np.linspace(0.2, 0.95, 4)
            else:
                contour_levels = np.linspace(-0.8 * stack_abs_max, 0.8 * stack_abs_max, 5)
            contour = ax2d.contour(
                np.asarray(mesh.z_centers),
                np.asarray(mesh.y_centers),
                field,
                levels=contour_levels,
                colors="white",
                linewidths=0.55,
                alpha=0.55,
            )
            contour_state.append(contour)
            time_text.set_text(f"t = {_format_time_with_units(times[index])}")
            peak_text.set_text(f"max|u| = {frame_peaks[index]:.2e}")
            return image, time_text, peak_text

        anim2d = animation.FuncAnimation(fig2d, update_2d, frames=len(frames), interval=1000 / fps, blit=False)
        update_2d(len(frames) - 1)
        poster_2d = out_dir / f"{output_stem}_2d_poster.png"
        fig2d.savefig(poster_2d, bbox_inches="tight")
        poster_2d_pdf = out_dir / f"{output_stem}_2d_poster.pdf"
        fig2d.savefig(poster_2d_pdf, bbox_inches="tight")
        outputs.extend([poster_2d, poster_2d_pdf])

    y_centers = mesh.y_centers
    z_centers = mesh.z_centers
    zz, yy = np.meshgrid(np.asarray(z_centers), np.asarray(y_centers))
    fig3d = None
    anim3d = None
    if include_3d:
        fig3d = plt.figure(figsize=(6.8, 5.2), constrained_layout=True)
        ax3d = fig3d.add_subplot(111, projection="3d")

        def update_3d(index: int):
            ax3d.cla()
            field = _movie_field(index)
            boundary_y = np.asarray([mesh.y_faces[0], mesh.y_faces[-1], mesh.y_faces[-1], mesh.y_faces[0], mesh.y_faces[0]], dtype=float)
            boundary_z = np.asarray([mesh.z_faces[0], mesh.z_faces[0], mesh.z_faces[-1], mesh.z_faces[-1], mesh.z_faces[0]], dtype=float)
            surface = ax3d.plot_surface(
                zz,
                yy,
                field,
                cmap=cmap,
                norm=norm,
                linewidth=0,
                antialiased=True,
            )
            ax3d.plot(boundary_z, boundary_y, np.zeros_like(boundary_y), color="#111827", linewidth=1.2, alpha=0.9)
            ax3d.set_xlabel("z")
            ax3d.set_ylabel("y")
            ax3d.set_zlabel(effective_colorbar_label)
            ax3d.set_title(
                f"{case_title} | 3D {effective_label.lower()}\n"
                f"t = {_format_time_with_units(times[index])} | max|u| = {frame_peaks[index]:.2e}"
            )
            if use_normalized_positive:
                ax3d.set_zlim(0.0, 1.05)
            else:
                ax3d.set_zlim(-stack_abs_max, stack_abs_max)
            ax3d.view_init(elev=26, azim=38)
            return (surface,)

        anim3d = animation.FuncAnimation(fig3d, update_3d, frames=len(frames), interval=1000 / fps, blit=False)
        update_3d(len(frames) - 1)
        poster_3d = out_dir / f"{output_stem}_3d_poster.png"
        fig3d.savefig(poster_3d, bbox_inches="tight")
        poster_3d_pdf = out_dir / f"{output_stem}_3d_poster.pdf"
        fig3d.savefig(poster_3d_pdf, bbox_inches="tight")
        outputs.extend([poster_3d, poster_3d_pdf])

    for suffix, writer_name in _safe_writer_candidates():
        if include_2d and anim2d is not None:
            writer = animation.writers[writer_name](fps=fps)
            path2d = out_dir / f"{output_stem}_2d.{suffix}"
            anim2d.save(path2d, writer=writer, dpi=72)
            outputs.append(path2d)
        if include_3d and anim3d is not None:
            path3d = out_dir / f"{output_stem}_3d.{suffix}"
            writer3d = animation.writers[writer_name](fps=fps)
            anim3d.save(path3d, writer=writer3d, dpi=72)
            outputs.append(path3d)

    if fig2d is not None:
        plt.close(fig2d)
    if fig3d is not None:
        plt.close(fig3d)
    return outputs


def write_geometry_preview_plots(
    mesh: StructuredMesh,
    out_dir: str | Path,
    *,
    case_title: str,
    fluid_mask: jnp.ndarray | None = None,
) -> list[Path]:
    _set_plot_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 5.8), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.15))
    ax2d = fig.add_subplot(grid[0, 0])
    ax3d = fig.add_subplot(grid[0, 1], projection="3d")
    fig.suptitle(case_title, fontsize=16)
    _draw_geometry_preview(ax2d, ax3d, mesh, case_title=case_title, fluid_mask=fluid_mask)

    png_path = out_dir / "geometry_preview.png"
    pdf_path = out_dir / "geometry_preview.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def write_geometry_gallery_plots(
    items: list[tuple[str, StructuredMesh, jnp.ndarray | None]],
    out_dir: str | Path,
    *,
    title: str,
) -> list[Path]:
    _set_plot_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(15.5, 8.8), constrained_layout=True)
    grid = fig.add_gridspec(2, len(items), height_ratios=(1.0, 1.15))
    fig.suptitle(title, fontsize=18)

    for column, (item_title, mesh, fluid_mask) in enumerate(items):
        ax2d = fig.add_subplot(grid[0, column])
        ax3d = fig.add_subplot(grid[1, column], projection="3d")
        _draw_geometry_preview(ax2d, ax3d, mesh, case_title=item_title, fluid_mask=fluid_mask)

    png_path = out_dir / "geometry_gallery.png"
    pdf_path = out_dir / "geometry_gallery.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def _centers_to_edges(values: np.ndarray) -> np.ndarray:
    if values.size <= 1:
        delta = 0.5
        return np.asarray([values[0] - delta, values[0] + delta], dtype=float)
    midpoints = 0.5 * (values[1:] + values[:-1])
    first = values[0] - 0.5 * (values[1] - values[0])
    last = values[-1] + 0.5 * (values[-1] - values[-2])
    return np.concatenate([[first], midpoints, [last]])


def _draw_geometry_preview(
    ax2d: plt.Axes,
    ax3d: plt.Axes,
    mesh: StructuredMesh,
    *,
    case_title: str,
    fluid_mask: jnp.ndarray | None = None,
) -> None:
    mask_source = fluid_mask if fluid_mask is not None else mesh.fluid_mask
    mask = jnp.asarray(mask_source) if mask_source is not None else jnp.ones(mesh.yz_shape, dtype=bool)
    if mask.size == 0:
        mask = jnp.ones(mesh.yz_shape, dtype=bool)
    mask_image = np.asarray(mask.astype(float))
    view_elev = 22
    view_azim = 34

    if mesh.point_coordinates is not None:
        points = np.asarray(mesh.point_coordinates)
        section = points[0]
        stride_r = max(section.shape[0] // 12, 1)
        stride_theta = max(section.shape[1] // 18, 1)
        radial = np.sqrt(section[:, :, 1] ** 2 + section[:, :, 2] ** 2)
        radial_scale = max(float(np.max(radial)), 1.0e-12)
        ax2d.scatter(
            section[:, :, 1].ravel(),
            section[:, :, 2].ravel(),
            c=(radial / radial_scale).ravel(),
            cmap="viridis",
            s=10,
            alpha=0.28,
            linewidths=0.0,
            zorder=0,
        )
        for radial_index in range(0, section.shape[0], stride_r):
            ax2d.plot(section[radial_index, :, 1], section[radial_index, :, 2], color="#1d4ed8", linewidth=0.8, alpha=0.9)
        for theta_index in range(0, section.shape[1], stride_theta):
            ax2d.plot(section[:, theta_index, 1], section[:, theta_index, 2], color="#b45309", linewidth=0.7, alpha=0.8)
        ax2d.set_title(f"{case_title}\nPipe cross-section")
        ax2d.set_xlabel("y")
        ax2d.set_ylabel("z")
        ax2d.set_aspect("equal")
        ax2d.scatter([0.0], [0.0], color="#111827", s=16, zorder=5)
        ax2d.text(
            0.02,
            0.98,
            "O-grid lines follow r and θ",
            transform=ax2d.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.92},
        )

        x_indices = (0, points.shape[0] // 2, -1)
        for offset, (slice_index, color) in enumerate(zip(x_indices, ("#1d4ed8", "#0f766e", "#b45309"), strict=True)):
            section = points[slice_index]
            ax3d.plot_wireframe(
                section[:, :, 1],
                section[:, :, 2],
                section[:, :, 0],
                rstride=max(section.shape[0] // 10, 1),
                cstride=max(section.shape[1] // 14, 1),
                color=color,
                linewidth=0.7,
                alpha=0.8 if offset else 0.5,
            )
        middle = points[points.shape[0] // 2]
        middle_radial = np.sqrt(middle[:, :, 1] ** 2 + middle[:, :, 2] ** 2)
        ax3d.plot_surface(
            middle[:, :, 1],
            middle[:, :, 2],
            middle[:, :, 0],
            facecolors=plt.cm.viridis(middle_radial / max(float(np.max(middle_radial)), 1.0e-12)),
            shade=False,
            linewidth=0.0,
            antialiased=True,
            alpha=0.92,
        )
        shell_r = np.asarray(points[:, -1, :, 1], dtype=float)
        shell_z = np.asarray(points[:, -1, :, 2], dtype=float)
        shell_x = np.asarray(points[:, -1, :, 0], dtype=float)
        ax3d.plot_surface(
            shell_r,
            shell_z,
            shell_x,
            color="#93c5fd",
            linewidth=0.0,
            antialiased=False,
            alpha=0.12,
            shade=False,
        )
        centerline_y = points[:, 0, 0, 1]
        centerline_z = points[:, 0, 0, 2]
        centerline_x = points[:, 0, 0, 0]
        ax3d.plot(centerline_y, centerline_z, centerline_x, color="#111827", linewidth=2.0)
        ax3d.quiver(
            float(np.max(points[:, 0, 0, 1]) + 0.12 * radial_scale),
            0.0,
            float(np.min(points[:, 0, 0, 0])),
            0.0,
            0.0,
            float(np.max(points[:, 0, 0, 0]) - np.min(points[:, 0, 0, 0])),
            color="#111827",
            arrow_length_ratio=0.08,
            linewidth=1.4,
        )
        ax3d.text(
            float(np.max(points[:, 0, 0, 1]) + 0.15 * radial_scale),
            0.0,
            float(np.mean(points[:, 0, 0, 0])),
            "flow",
            color="#111827",
        )
        ax3d.set_title(f"{case_title}\nPipe domain")
        view_elev = 18
        view_azim = -52
    else:
        ax2d.set_facecolor("#f8fafc")
        z0, z1 = float(mesh.z_faces[0]), float(mesh.z_faces[-1])
        y0, y1 = float(mesh.y_faces[0]), float(mesh.y_faces[-1])
        for z_face in np.asarray(mesh.z_faces, dtype=float):
            ax2d.plot([z_face, z_face], [y0, y1], color="#cbd5e1", linewidth=0.55, alpha=0.8, zorder=1)
        for y_face in np.asarray(mesh.y_faces, dtype=float):
            ax2d.plot([z0, z1], [y_face, y_face], color="#cbd5e1", linewidth=0.55, alpha=0.8, zorder=1)
        if np.all(mask_image > 0.5):
            ax2d.add_patch(
                Rectangle(
                    (z0, y0),
                    z1 - z0,
                    y1 - y0,
                    facecolor="#ccfbf1",
                    edgecolor="#0f766e",
                    linewidth=1.6,
                    alpha=0.85,
                    zorder=0,
                )
            )
            legend_handles = [Patch(facecolor="#ccfbf1", edgecolor="#0f766e", label="Fluid cells")]
            ax2d.set_title(f"{case_title}\nFluid cross-section")
        else:
            region_map = np.where(mask_image > 0.5, 1.0, 0.0)
            cmap = colors.ListedColormap(["#e5e7eb", "#99f6e4"])
            ax2d.pcolormesh(mesh.z_faces, mesh.y_faces, region_map, shading="auto", cmap=cmap, vmin=0.0, vmax=1.0, zorder=0)
            legend_handles = [
                Patch(facecolor="#99f6e4", edgecolor="none", label="Fluid region"),
                Patch(facecolor="#e5e7eb", edgecolor="none", label="Wall / exterior cells"),
            ]
            ax2d.set_title(f"{case_title}\nFluid and wall regions")
        ax2d.set_xlabel("z")
        ax2d.set_ylabel("y")
        ax2d.set_aspect("equal")
        ax2d.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=max(1, len(legend_handles)))
        x0, x1 = float(mesh.x_faces[0]), float(mesh.x_faces[-1])
        corners = np.asarray(
            [
                [x0, y0, z0],
                [x0, y0, z1],
                [x0, y1, z0],
                [x0, y1, z1],
                [x1, y0, z0],
                [x1, y0, z1],
                [x1, y1, z0],
                [x1, y1, z1],
            ]
        )
        edges = (
            (0, 1), (0, 2), (1, 3), (2, 3),
            (4, 5), (4, 6), (5, 7), (6, 7),
            (0, 4), (1, 5), (2, 6), (3, 7),
        )
        for start, end in edges:
            ax3d.plot(
                [corners[start, 1], corners[end, 1]],
                [corners[start, 2], corners[end, 2]],
                [corners[start, 0], corners[end, 0]],
                color="#1d4ed8",
                linewidth=1.0,
                alpha=0.65,
            )
        x_slices = np.linspace(x0, x1, 5)
        for slice_index, x_slice in enumerate(x_slices):
            if np.all(mask_image > 0.5):
                facecolors = np.zeros(mask_image.shape + (4,), dtype=float)
                facecolors[..., :] = colors.to_rgba("#14b8a6", alpha=0.24 if slice_index not in (1, 3) else 0.44)
            else:
                facecolors = np.zeros(mask_image.shape + (4,), dtype=float)
                facecolors[mask_image > 0.5, :] = colors.to_rgba(
                    "#14b8a6", alpha=0.22 if slice_index not in (1, 3) else 0.42
                )
                facecolors[mask_image <= 0.5, :] = colors.to_rgba(
                    "#cbd5e1", alpha=0.10 if slice_index not in (1, 3) else 0.22
                )
            ax3d.plot_surface(
                np.full_like(mask_image, x_slice, dtype=float),
                np.broadcast_to(np.asarray(mesh.y_centers)[:, None], mask_image.shape),
                np.broadcast_to(np.asarray(mesh.z_centers)[None, :], mask_image.shape),
                facecolors=facecolors,
                rstride=1,
                cstride=1,
                shade=False,
                linewidth=0.0,
                antialiased=False,
            )
        if np.any(mask_image > 0.5):
            fluid_rows = np.where(np.any(mask_image > 0.5, axis=1))[0]
            fluid_cols = np.where(np.any(mask_image > 0.5, axis=0))[0]
            fy0 = float(mesh.y_faces[fluid_rows[0]])
            fy1 = float(mesh.y_faces[fluid_rows[-1] + 1])
            fz0 = float(mesh.z_faces[fluid_cols[0]])
            fz1 = float(mesh.z_faces[fluid_cols[-1] + 1])
            fluid_corners = np.asarray(
                [
                    [x0, fy0, fz0],
                    [x0, fy0, fz1],
                    [x0, fy1, fz0],
                    [x0, fy1, fz1],
                    [x1, fy0, fz0],
                    [x1, fy0, fz1],
                    [x1, fy1, fz0],
                    [x1, fy1, fz1],
                ]
            )
            for start, end in edges:
                ax3d.plot(
                    [fluid_corners[start, 1], fluid_corners[end, 1]],
                    [fluid_corners[start, 2], fluid_corners[end, 2]],
                    [fluid_corners[start, 0], fluid_corners[end, 0]],
                    color="#b45309",
                    linewidth=1.25,
                    alpha=0.85,
                )
        ax3d.quiver(
            y1 + 0.08 * (y1 - y0),
            z0,
            x0,
            0.0,
            0.0,
            x1 - x0,
            color="#111827",
            arrow_length_ratio=0.08,
            linewidth=1.4,
        )
        ax3d.text(y1 + 0.1 * (y1 - y0), z0, 0.5 * (x0 + x1), "flow", color="#111827")
        ax3d.set_title(f"{case_title}\nExtruded domain")

    ax3d.set_xlabel("y")
    ax3d.set_ylabel("z")
    ax3d.set_zlabel("x")
    ax3d.view_init(elev=view_elev, azim=view_azim)
    ax3d.set_box_aspect(
        (
            float(mesh.y_faces[-1] - mesh.y_faces[0]),
            float(mesh.z_faces[-1] - mesh.z_faces[0]),
            max(0.75 * float(mesh.x_faces[-1] - mesh.x_faces[0]), 1.2 * float(mesh.y_faces[-1] - mesh.y_faces[0])),
        )
    )
    ax3d.set_xticks([])
    ax3d.set_yticks([])
    ax3d.set_zticks([])


def write_extruded_overview_plots(
    solution,
    out_dir: str | Path,
    *,
    case_title: str,
) -> list[Path]:
    _set_plot_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = solution.bundle
    validation = solution.validation
    x = np.asarray(bundle.x, dtype=float)
    field_scale = np.asarray(bundle.field_scale, dtype=float)
    mean_velocity = np.asarray(bundle.mean_velocity, dtype=float)
    current_proxy = np.asarray(bundle.current_scaled_pressure_proxy, dtype=float)
    charge_balance = np.maximum(np.asarray(bundle.charge_balance_residual, dtype=float), 1.0e-16)
    boundary_current = np.maximum(np.asarray(bundle.boundary_current_residual, dtype=float), 1.0e-16)
    wall_leakage = np.maximum(np.asarray(bundle.wall_current_leakage, dtype=float), 1.0e-16)
    axial_current = np.asarray(bundle.axial_current, dtype=float)

    peak_index = int(np.argmax(np.abs(field_scale))) if field_scale.size else 0
    y = np.asarray(bundle.y, dtype=float)
    z = np.asarray(bundle.z, dtype=float)
    y_edges = _centers_to_edges(y)
    z_edges = _centers_to_edges(z)
    coord_x_label = "r" if bundle.geometry_kind == "pipe_ogrid" else "y"
    coord_y_label = r"$\\theta$" if bundle.geometry_kind == "pipe_ogrid" else "z"

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    fig.suptitle(case_title, fontsize=16)

    axes[0, 0].plot(x, mean_velocity, color="#0f766e", label="Mean velocity")
    axes[0, 0].plot(x, current_proxy, color="#b45309", linestyle="--", label="Current proxy")
    axes[0, 0].plot(x, field_scale, color="#1d4ed8", alpha=0.7, label="Field scale")
    axes[0, 0].set_title("Station response")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].legend()

    axes[0, 1].semilogy(x, charge_balance, color="#7c3aed", label="Charge balance")
    axes[0, 1].semilogy(x, wall_leakage, color="#dc2626", linestyle="--", label="Wall leakage")
    axes[0, 1].semilogy(x, boundary_current, color="#0891b2", linestyle=":", label="Boundary residual")
    axes[0, 1].plot(x, np.maximum(np.abs(axial_current), 1.0e-16), color="#111827", alpha=0.6, label="|Axial current|")
    axes[0, 1].set_title(
        "Conservation audit\n"
        f"max|div J|={validation.max_charge_balance_residual:.2e}, "
        f"net boundary={validation.net_boundary_current_residual:.2e}"
    )
    axes[0, 1].set_xlabel("x")
    axes[0, 1].legend()

    u_station = np.asarray(bundle.u[peak_index], dtype=float)
    phi_station = np.asarray(bundle.phi[peak_index], dtype=float)
    u_im = axes[1, 0].pcolormesh(z_edges, y_edges, u_station, shading="auto", cmap="RdBu_r")
    plt.colorbar(u_im, ax=axes[1, 0], fraction=0.046, pad=0.04)
    axes[1, 0].set_title(f"u at peak field station (x={x[peak_index]:.2f})")
    axes[1, 0].set_xlabel(coord_y_label)
    axes[1, 0].set_ylabel(coord_x_label)

    phi_im = axes[1, 1].pcolormesh(z_edges, y_edges, phi_station, shading="auto", cmap="PuOr_r")
    plt.colorbar(phi_im, ax=axes[1, 1], fraction=0.046, pad=0.04)
    axes[1, 1].set_title("Electric potential at peak field station")
    axes[1, 1].set_xlabel(coord_y_label)
    axes[1, 1].set_ylabel(coord_x_label)

    png_path = out_dir / "extruded_overview.png"
    pdf_path = out_dir / "extruded_overview.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def write_strong_scaling_plots(
    records: list[dict[str, object]],
    out_dir: str | Path,
    *,
    case_title: str,
) -> list[Path]:
    _set_plot_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups: dict[str, list[dict[str, object]]] = {}
    for record in records:
        groups.setdefault(str(record["platform"]), []).append(record)
    for values in groups.values():
        values.sort(key=lambda item: int(item["num_devices"]))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    fig.suptitle(case_title, fontsize=16)
    palette = ["#0f766e", "#1d4ed8", "#b45309", "#7c3aed"]

    for color, (platform_name, values) in zip(palette, groups.items(), strict=False):
        device_counts = np.asarray([int(item["num_devices"]) for item in values], dtype=float)
        runtimes = np.asarray([float(item.get("warm_seconds", item["mean_seconds"])) for item in values], dtype=float)
        baseline = runtimes[0]
        speedup = baseline / np.maximum(runtimes, 1.0e-12)
        ny_value = values[0].get("ny")
        nz_value = values[0].get("nz")
        if ny_value is None or nz_value is None:
            label = str(platform_name)
        else:
            ny = int(ny_value)
            nz = int(nz_value)
            label = f"{platform_name} ({ny}×{nz})"

        axes[0].plot(device_counts, runtimes, marker="o", color=color, label=label)
        axes[1].plot(device_counts, speedup, marker="o", color=color, label=label)

    ideal_device_counts = np.asarray(sorted({int(item["num_devices"]) for item in records}), dtype=float)
    axes[1].plot(
        ideal_device_counts,
        ideal_device_counts / ideal_device_counts[0],
        linestyle="--",
        color="#64748b",
        alpha=0.85,
        label="Ideal linear speedup",
    )

    axes[0].set_title("Warm runtime")
    axes[0].set_xlabel("Device count")
    axes[0].set_ylabel("Runtime [s]")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(sorted({int(item["num_devices"]) for item in records}))
    axes[0].get_xaxis().set_major_formatter(ScalarFormatter())
    axes[0].legend()
    axes[0].text(
        0.02,
        0.02,
        "Fixed global problem per platform.\nWarm runtime excludes first-call compilation.",
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.9},
    )

    axes[1].set_title("Strong-scaling speedup")
    axes[1].set_xlabel("Device count")
    axes[1].set_ylabel("Warm-runtime speedup")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(sorted({int(item["num_devices"]) for item in records}))
    axes[1].get_xaxis().set_major_formatter(ScalarFormatter())
    axes[1].legend()

    png_path = out_dir / "strong_scaling.png"
    pdf_path = out_dir / "strong_scaling.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def write_autodiff_plots(
    sensitivity_scan: list[dict[str, float]],
    optimization_history: list[dict[str, float]],
    out_dir: str | Path,
    *,
    case_title: str,
    target_parameter: float,
    parameter_key: str = "hartmann_number",
    parameter_label: str = "Recovered parameter",
    target_label: str = "Target parameter",
) -> list[Path]:
    _set_plot_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), constrained_layout=True)
    fig.suptitle(case_title, fontsize=16)

    ha_values = np.asarray([item["hartmann_number"] for item in sensitivity_scan], dtype=float)
    mean_velocity = np.asarray([item["mean_velocity"] for item in sensitivity_scan], dtype=float)
    d_mean_velocity = np.asarray([item["d_mean_velocity_d_ha"] for item in sensitivity_scan], dtype=float)

    axes[0].plot(ha_values, mean_velocity, color="#0f766e", label=r"$\bar{u}(Ha)$")
    axes[0].plot(ha_values, d_mean_velocity, color="#b45309", linestyle="--", label=r"$d\bar{u}/dHa$")
    axes[0].set_title("Sensitivity scan")
    axes[0].set_xlabel("Hartmann number")
    axes[0].set_ylabel("Response")
    axes[0].legend()
    axes[0].text(
        0.03,
        0.03,
        "Mean throughput drops as Hartmann layers strengthen.\nThe dashed line is the autodiff sensitivity.",
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.9},
    )

    iteration = np.asarray([item["iteration"] for item in optimization_history], dtype=float)
    objective = np.asarray([item["loss"] for item in optimization_history], dtype=float)
    parameter = np.asarray([item[parameter_key] for item in optimization_history], dtype=float)

    axes[1].plot(iteration, objective, color="#1d4ed8", label="Loss")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Gradient step")
    axes[1].set_ylabel("Profile misfit")
    twin = axes[1].twinx()
    twin.plot(iteration, parameter, color="#7c3aed", linestyle="--", label=parameter_label)
    twin.axhline(target_parameter, color="#111827", linestyle=":", linewidth=1.2, label=target_label)
    axes[1].set_title("Inverse design")
    if parameter.size:
        axes[1].text(
            0.03,
            0.03,
            f"{parameter_label} = {parameter[-1]:.3f}\nTarget = {target_parameter:.3f}",
            transform=axes[1].transAxes,
            ha="left",
            va="bottom",
            fontsize=11,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.9},
        )

    lines_left, labels_left = axes[1].get_legend_handles_labels()
    lines_right, labels_right = twin.get_legend_handles_labels()
    axes[1].legend(lines_left + lines_right, labels_left + labels_right, loc="upper left")

    png_path = out_dir / "autodiff_summary.png"
    pdf_path = out_dir / "autodiff_summary.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]
