"""Plots and compressed media for solver and validation results."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .core import Solution
from .mesh import StructuredMesh, generate_bent_pipe_mesh
from .validation import hartmann_analytic_profile
from .validation import extract_midplane_profile


def _load_matplotlib() -> None:
    global plt, animation, colors, Line2D, Patch, Rectangle, ScalarFormatter, inset_axes, mark_inset
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import animation, colors
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.ticker import ScalarFormatter
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset


def _set_plot_style() -> None:
    _load_matplotlib()
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
            "legend.fontsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "lines.linewidth": 2.0,
        }
    )


def _prepare_plot_output(out_dir: str | Path) -> Path:
    """Load plotting dependencies, apply the house style, and create output."""

    _set_plot_style()
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    return output


def _save_figure_pair(
    fig,
    out_dir: Path,
    stem: str,
    *,
    dpi: int | None = None,
    tight: bool = True,
) -> list[Path]:
    """Save one figure as PNG and PDF, then release its Matplotlib state."""

    import matplotlib.pyplot as pyplot

    save_options = {"bbox_inches": "tight"} if tight else {}
    if dpi is not None:
        save_options["dpi"] = dpi
    paths = [out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"]
    for path in paths:
        fig.savefig(path, **save_options)
    pyplot.close(fig)
    return paths


def _draw_duct_wireframe(
    ax,
    *,
    length: float,
    y_faces: np.ndarray,
    z_faces: np.ndarray,
    face_alpha: float = 0.09,
) -> None:
    y0, y1 = float(y_faces[0]), float(y_faces[-1])
    z0, z1 = float(z_faces[0]), float(z_faces[-1])
    x0, x1 = 0.0, float(length)
    wall_style = {"color": "#d1d5db", "alpha": face_alpha, "linewidth": 0.0, "shade": False}
    line_color = "#6b7280"
    xx = np.asarray([[x0, x1], [x0, x1]], dtype=float)
    yy_span = np.asarray([[y0, y0], [y1, y1]], dtype=float)
    zz_span = np.asarray([[z0, z1], [z0, z1]], dtype=float)
    ax.plot_surface(xx, np.full_like(xx, z0), yy_span, **wall_style)
    ax.plot_surface(xx, np.full_like(xx, z1), yy_span, **wall_style)
    ax.plot_surface(xx, zz_span, np.full_like(xx, y0), **wall_style)
    ax.plot_surface(xx, zz_span, np.full_like(xx, y1), **wall_style)
    boundary_y = np.asarray([y0, y1, y1, y0, y0], dtype=float)
    boundary_z = np.asarray([z0, z0, z1, z1, z0], dtype=float)
    for x in (x0, x1):
        ax.plot(
            np.full_like(boundary_y, x),
            boundary_z,
            boundary_y,
            color=line_color,
            linewidth=1.2,
            alpha=0.85,
        )
    for y in (y0, y1):
        for z in (z0, z1):
            ax.plot([x0, x1], [z, z], [y, y], color=line_color, linewidth=1.2, alpha=0.85)


def _draw_profile_slab(
    ax,
    *,
    field: np.ndarray,
    y_display: np.ndarray,
    z_display: np.ndarray,
    cmap_obj,
    norm,
    x_plane: float,
    amplitude: float,
    use_normalized_positive: bool,
) -> None:
    field_display = np.asarray(field, dtype=float)
    dense_y_count = max(3 * y_display.size, y_display.size)
    dense_z_count = max(3 * z_display.size, z_display.size)
    y_dense = np.linspace(float(y_display[0]), float(y_display[-1]), dense_y_count)
    z_dense = np.linspace(float(z_display[0]), float(z_display[-1]), dense_z_count)
    field_z_dense = np.vstack([np.interp(z_dense, z_display, row) for row in field_display])
    field_dense = np.column_stack([np.interp(y_dense, y_display, field_z_dense[:, j]) for j in range(field_z_dense.shape[1])])
    yy_surface, zz_surface = np.meshgrid(y_dense, z_dense, indexing="ij")
    if use_normalized_positive:
        displacement_field = np.clip(field_dense, 0.0, None)
        peak = max(float(np.max(displacement_field)), 1.0e-12)
        normalized_displacement = displacement_field / peak
    else:
        peak = max(float(np.max(np.abs(field_dense))), 1.0e-12)
        normalized_displacement = 0.5 * (field_dense / peak + 1.0)
    x_surface = x_plane + amplitude * normalized_displacement
    color_rgba = cmap_obj(norm(field_dense))
    ax.plot_surface(
        x_surface,
        zz_surface,
        yy_surface,
        facecolors=color_rgba,
        shade=False,
        linewidth=0.0,
        edgecolor="none",
        antialiased=True,
    )
    base_plane = np.full_like(yy_surface, x_plane)
    ax.plot_surface(
        base_plane,
        zz_surface,
        yy_surface,
        color="#cbd5e1",
        alpha=0.08,
        linewidth=0.0,
        shade=False,
    )
    edge_alpha = 0.86
    side_color_top = cmap_obj(norm(field_dense[-1, :]))
    side_color_bottom = cmap_obj(norm(field_dense[0, :]))
    top_x = np.vstack([base_plane[-1, :], x_surface[-1, :]])
    top_z = np.vstack([zz_surface[-1, :], zz_surface[-1, :]])
    top_y = np.vstack([yy_surface[-1, :], yy_surface[-1, :]])
    bottom_x = np.vstack([base_plane[0, :], x_surface[0, :]])
    bottom_z = np.vstack([zz_surface[0, :], zz_surface[0, :]])
    bottom_y = np.vstack([yy_surface[0, :], yy_surface[0, :]])
    left_x = np.column_stack([base_plane[:, 0], x_surface[:, 0]])
    left_z = np.column_stack([zz_surface[:, 0], zz_surface[:, 0]])
    left_y = np.column_stack([yy_surface[:, 0], yy_surface[:, 0]])
    right_x = np.column_stack([base_plane[:, -1], x_surface[:, -1]])
    right_z = np.column_stack([zz_surface[:, -1], zz_surface[:, -1]])
    right_y = np.column_stack([yy_surface[:, -1], yy_surface[:, -1]])
    for side_x, side_z, side_y, side_color in (
        (top_x, top_z, top_y, np.tile(side_color_top[None, :, :], (2, 1, 1))),
        (bottom_x, bottom_z, bottom_y, np.tile(side_color_bottom[None, :, :], (2, 1, 1))),
        (left_x, left_z, left_y, np.tile(cmap_obj(norm(field_dense[:, 0]))[:, None, :], (1, 2, 1))),
        (right_x, right_z, right_y, np.tile(cmap_obj(norm(field_dense[:, -1]))[:, None, :], (1, 2, 1))),
    ):
        ax.plot_surface(
            side_x,
            side_z,
            side_y,
            facecolors=side_color,
            shade=False,
            linewidth=0.0,
            edgecolor="none",
            antialiased=True,
            alpha=edge_alpha,
        )


def _plot_field(ax: plt.Axes, solution: Solution, field: jnp.ndarray, *, title: str, cmap: str) -> None:
    _load_matplotlib()
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
    ax.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98))


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
    out_dir = _prepare_plot_output(out_dir)

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

    overview_paths = _save_figure_pair(fig, out_dir, "overview")

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
        axes[0].legend(loc="upper left", bbox_to_anchor=(0.02, 0.98))

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
        axes[1].legend(loc="upper left", bbox_to_anchor=(0.02, 0.98))

        diagnostics_paths = _save_figure_pair(fig, out_dir, "diagnostics")

    return [*overview_paths, *diagnostics_paths]


def _preferred_animation_writer() -> tuple[str, str]:
    _load_matplotlib()
    available = set(animation.writers.list())
    if "pillow" in available:
        return "gif", "pillow"
    if "imagemagick" in available:
        return "gif", "imagemagick"
    return "gif", "pillow"


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


def _add_layer_annotations(
    ax: plt.Axes,
    mesh: StructuredMesh,
    fluid_mask: jnp.ndarray | None,
    *,
    show_side_layers: bool,
    case_hint: str = "",
) -> None:
    case_hint = case_hint.lower()
    side_label = "Side layers"
    hartmann_vertical = "hunt" in case_hint or "shercliff" in case_hint
    if "hunt" in case_hint:
        side_label = "Hunt / side\nlayers"
    elif "shercliff" in case_hint:
        side_label = "Shercliff\nlayers"
    annotation_style = {
        "bbox": {"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.92},
        "arrowprops": {"arrowstyle": "->", "color": "#111827", "lw": 1.0},
        "fontsize": 11,
        "color": "#111827",
    }
    if hartmann_vertical:
        ax.annotate(
            "Hartmann\nlayers",
            xy=(0.93, 0.50),
            xycoords="axes fraction",
            xytext=(1.03, 0.50),
            textcoords="axes fraction",
            ha="left",
            va="center",
            **annotation_style,
        )
        ax.annotate(
            "",
            xy=(0.07, 0.50),
            xycoords="axes fraction",
            xytext=(-0.03, 0.50),
            textcoords="axes fraction",
            **{key: value for key, value in annotation_style.items() if key in {"arrowprops"}},
        )
    else:
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
        if hartmann_vertical:
            ax.annotate(
                side_label,
                xy=(0.50, 0.93),
                xycoords="axes fraction",
                xytext=(0.50, 0.985),
                textcoords="axes fraction",
                ha="center",
                va="top",
                **annotation_style,
            )
            ax.annotate(
                "",
                xy=(0.50, 0.07),
                xycoords="axes fraction",
                xytext=(0.50, 0.015),
                textcoords="axes fraction",
                **{key: value for key, value in annotation_style.items() if key in {"arrowprops"}},
            )
        else:
            ax.annotate(
                side_label,
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
    symmetry_average_axes: tuple[str, ...] = (),
    profile_fluid_only: bool = True,
    view_elev: float = 18.0,
    view_azim: float = -68.0,
) -> list[Path]:
    if not frames:
        return []
    out_dir = _prepare_plot_output(out_dir)

    mesh = frames[0]["mesh"]
    display_fields, frame_peaks, movie_label, colorbar_label = _movie_field_stack(frames, field_mode=field_mode)
    raw_u_fields = [np.asarray(frame["u"], dtype=float) for frame in frames]
    fluid_mask = jnp.asarray(frames[0].get("fluid_mask", jnp.ones_like(display_fields[0], dtype=bool)))

    def _symmetrize(field: np.ndarray) -> np.ndarray:
        result = np.asarray(field, dtype=float)
        if "y" in symmetry_average_axes:
            result = 0.5 * (result + result[::-1, :])
        if "z" in symmetry_average_axes:
            result = 0.5 * (result + result[:, ::-1])
        return result

    if symmetry_average_axes:
        display_fields = [_symmetrize(field) for field in display_fields]
        raw_u_fields = [_symmetrize(field) for field in raw_u_fields]
        frame_peaks = [max(float(np.max(np.abs(field[np.asarray(fluid_mask, dtype=bool)]))), 1.0e-12) for field in raw_u_fields]

    fluid_values = jnp.asarray(np.stack([jnp.where(fluid_mask, jnp.asarray(field), jnp.nan) for field in display_fields]))
    stack_min = float(jnp.nanmin(fluid_values))
    stack_max = float(jnp.nanmax(fluid_values))
    stack_abs_max = max(float(jnp.nanmax(jnp.abs(fluid_values))), 1e-12)
    negative_tolerance = 5.0e-2 * stack_abs_max
    negative_fraction = float(jnp.nanmean((fluid_values < -negative_tolerance).astype(float)))
    use_normalized_positive = (
        stack_max <= negative_tolerance
        or stack_min >= -negative_tolerance
        or negative_fraction <= 0.02
    )
    if use_normalized_positive:
        cmap = "viridis"
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

    fluid_mask_np = np.asarray(fluid_mask, dtype=bool)

    def _profile_data(index: int, axis: str) -> tuple[np.ndarray, np.ndarray]:
        field = raw_u_fields[index]
        if axis == "y":
            mid_z = int(len(mesh.z_centers) // 2)
            values = np.asarray(field[:, mid_z], dtype=float)
            coords = np.asarray(mesh.y_centers, dtype=float)
            if profile_fluid_only:
                mask = fluid_mask_np[:, mid_z]
                values = values[mask]
                coords = coords[mask]
            return coords, values
        if axis == "z":
            mid_y = int(len(mesh.y_centers) // 2)
            values = np.asarray(field[mid_y, :], dtype=float)
            coords = np.asarray(mesh.z_centers, dtype=float)
            if profile_fluid_only:
                mask = fluid_mask_np[mid_y, :]
                values = values[mask]
                coords = coords[mask]
            return coords, values
        raise ValueError(f"Unsupported profile axis {axis}")

    fig2d = None
    anim2d = None
    if include_2d:
        fig2d = plt.figure(figsize=(8.6, 5.6), constrained_layout=True)
        grid2d = fig2d.add_gridspec(2, 2, width_ratios=(1.0, 0.38), height_ratios=(1.0, 1.0))
        ax2d = fig2d.add_subplot(grid2d[:, 0])
        profile_y_ax = fig2d.add_subplot(grid2d[0, 1])
        profile_z_ax = fig2d.add_subplot(grid2d[1, 1])
        image = ax2d.pcolormesh(
            np.asarray(mesh.z_faces),
            np.asarray(mesh.y_faces),
            _movie_field(0),
            shading="auto",
            cmap=cmap,
            norm=norm,
        )
        ax2d.set_xlabel("z")
        ax2d.set_ylabel("y")
        ax2d.set_title(f"{case_title}\n2D {effective_label.lower()}")
        ax2d.set_aspect("equal")
        _add_layer_annotations(
            ax2d,
            mesh,
            frames[0].get("fluid_mask"),
            show_side_layers=show_side_layers,
            case_hint=case_title,
        )
        y_profile_coord, current_y_profile = _profile_data(0, "y")
        z_profile_coord, current_z_profile = _profile_data(0, "z")
        peak_profile = max(frame_peaks[0], 1.0e-12)
        hartmann_reference = None
        if "hartmann" in case_name and getattr(getattr(case, "geometry", None), "target_ha", None) is not None:
            hartmann_reference = np.asarray(
                hartmann_analytic_profile(
                    y_profile_coord,
                    float(case.geometry.target_ha),
                )
            )

        profile_y_ax.set_title("y-centerline", fontsize=11)
        profile_y_ax.set_xlabel("u / max u(t)", fontsize=11)
        profile_y_ax.set_ylabel("y", fontsize=11)
        profile_y_ax.tick_params(labelsize=10)
        profile_y_line, = profile_y_ax.plot(
            current_y_profile / peak_profile,
            y_profile_coord,
            color="#111827",
            linewidth=1.7,
            label="LMX transient",
        )
        if hartmann_reference is not None:
            reference_scale = max(float(np.max(np.abs(hartmann_reference))), 1.0e-12)
            profile_y_ax.plot(
                hartmann_reference / reference_scale,
                y_profile_coord,
                color="#b45309",
                linestyle="--",
                linewidth=1.5,
                label="steady analytic",
            )
        profile_y_ax.set_xlim(-0.02, 1.05)
        profile_y_ax.set_ylim(float(np.min(y_profile_coord)), float(np.max(y_profile_coord)))
        profile_y_ax.legend(loc="lower right", fontsize=10)

        profile_z_ax.set_title("z-centerline", fontsize=11)
        profile_z_ax.set_xlabel("z", fontsize=11)
        profile_z_ax.set_ylabel("u / max u(t)", fontsize=11)
        profile_z_ax.tick_params(labelsize=10)
        profile_z_line, = profile_z_ax.plot(
            z_profile_coord,
            current_z_profile / peak_profile,
            color="#0f766e",
            linewidth=1.7,
        )
        profile_z_ax.set_xlim(float(np.min(z_profile_coord)), float(np.max(z_profile_coord)))
        profile_z_ax.set_ylim(-0.02, 1.05)
        annotation_bbox = {"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.85}
        time_text = ax2d.text(0.02, 0.98, "", transform=ax2d.transAxes, ha="left", va="top", bbox=annotation_bbox)
        peak_text = ax2d.text(0.98, 0.98, "", transform=ax2d.transAxes, ha="right", va="top", bbox=annotation_bbox)
        plt.colorbar(image, ax=ax2d, fraction=0.046, pad=0.04, label=effective_colorbar_label)
        contour_state: list[object] = []

        def update_2d(index: int):
            field = _movie_field(index)
            image.set_array(np.asarray(field).ravel())
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
            y_profile_coord, y_profile_field = _profile_data(index, "y")
            z_profile_coord, z_profile_field = _profile_data(index, "z")
            profile_span = max(frame_peaks[index], 1.0e-12)
            profile_y_line.set_data(y_profile_field / profile_span, y_profile_coord)
            profile_z_line.set_data(z_profile_coord, z_profile_field / profile_span)
            profile_y_ax.set_xlim(-0.02, 1.05)
            profile_z_ax.set_ylim(-0.02, 1.05)
            time_text.set_text(f"t = {_format_time_with_units(times[index])}")
            peak_text.set_text(f"max|u| = {frame_peaks[index]:.2e}")
            return image, time_text, peak_text, profile_y_line, profile_z_line

        anim2d = animation.FuncAnimation(fig2d, update_2d, frames=len(frames), interval=1000 / fps, blit=False)
        update_2d(len(frames) - 1)
        poster_2d = out_dir / f"{output_stem}_2d_poster.png"
        fig2d.savefig(poster_2d, bbox_inches="tight")
        poster_2d_pdf = out_dir / f"{output_stem}_2d_poster.pdf"
        fig2d.savefig(poster_2d_pdf, bbox_inches="tight")
        outputs.extend([poster_2d, poster_2d_pdf])

    y_centers = np.asarray(mesh.y_centers, dtype=float)
    z_centers = np.asarray(mesh.z_centers, dtype=float)
    if np.any(fluid_mask_np):
        fluid_y_indices = np.where(np.any(fluid_mask_np, axis=1))[0]
        fluid_z_indices = np.where(np.any(fluid_mask_np, axis=0))[0]
        y_centers_3d = y_centers[fluid_y_indices]
        z_centers_3d = z_centers[fluid_z_indices]
    else:
        fluid_y_indices = np.arange(y_centers.size, dtype=int)
        fluid_z_indices = np.arange(z_centers.size, dtype=int)
        y_centers_3d = y_centers
        z_centers_3d = z_centers
    max_display_points = 49
    if y_centers_3d.size <= max_display_points:
        y_display = y_centers_3d
        display_y_indices = fluid_y_indices
    else:
        display_y_local = np.unique(np.round(np.linspace(0, y_centers_3d.size - 1, max_display_points)).astype(int))
        display_y_indices = fluid_y_indices[display_y_local]
        y_display = y_centers[display_y_indices]
    if z_centers_3d.size <= max_display_points:
        z_display = z_centers_3d
        display_z_indices = fluid_z_indices
    else:
        display_z_local = np.unique(np.round(np.linspace(0, z_centers_3d.size - 1, max_display_points)).astype(int))
        display_z_indices = fluid_z_indices[display_z_local]
        z_display = z_centers[display_z_indices]
    x_extent = max(float(mesh.z_faces[-1] - mesh.z_faces[0]), float(mesh.y_faces[-1] - mesh.y_faces[0]), 1.0)
    fig3d = None
    anim3d = None
    if include_3d:
        fig3d = plt.figure(figsize=(7.4, 5.8), constrained_layout=True)
        ax3d = fig3d.add_subplot(111, projection="3d")
        cmap_obj = plt.get_cmap(cmap)

        def update_3d(index: int):
            ax3d.cla()
            field = _movie_field(index)
            field_display = np.asarray(field[np.ix_(display_y_indices, display_z_indices)], dtype=float)
            amplitude = 0.32 * x_extent
            _draw_duct_wireframe(
                ax3d,
                length=x_extent,
                y_faces=np.asarray(mesh.y_faces, dtype=float),
                z_faces=np.asarray(mesh.z_faces, dtype=float),
                face_alpha=0.11,
            )
            _draw_profile_slab(
                ax3d,
                field=field_display,
                y_display=y_display,
                z_display=z_display,
                cmap_obj=cmap_obj,
                norm=norm,
                x_plane=0.38 * x_extent,
                amplitude=amplitude,
                use_normalized_positive=use_normalized_positive,
            )

            ax3d.quiver(
                0.10 * x_extent,
                float(mesh.z_faces[0]) - 0.08 * (mesh.z_faces[-1] - mesh.z_faces[0]),
                float(mesh.y_faces[0]),
                0.72 * x_extent,
                0.0,
                0.0,
                color="#111827",
                linewidth=1.6,
                arrow_length_ratio=0.08,
            )
            ax3d.text(
                0.86 * x_extent,
                float(mesh.z_faces[0]) - 0.10 * (mesh.z_faces[-1] - mesh.z_faces[0]),
                float(mesh.y_faces[0]),
                "flow",
                color="#111827",
                fontsize=11,
            )
            ax3d.set_title(
                f"{case_title} | 3D streamwise-velocity profile\n"
                f"t = {_format_time_with_units(times[index])} | max|u| = {frame_peaks[index]:.2e}"
            )
            ax3d.set_xlim(0.0, 1.01 * x_extent)
            ax3d.set_ylim(float(mesh.z_faces[0]) - 0.14 * (mesh.z_faces[-1] - mesh.z_faces[0]), float(mesh.z_faces[-1]))
            ax3d.set_zlim(float(mesh.y_faces[0]), float(mesh.y_faces[-1]))
            ax3d.view_init(elev=view_elev, azim=view_azim)
            ax3d.set_box_aspect((5.6, 1.25, 1.25))
            ax3d.grid(False)
            ax3d.set_xticks([])
            ax3d.set_yticks([])
            ax3d.set_zticks([])
            ax3d.set_axis_off()
            return ()

        anim3d = animation.FuncAnimation(fig3d, update_3d, frames=len(frames), interval=1000 / fps, blit=False)
        update_3d(len(frames) - 1)
        poster_3d = out_dir / f"{output_stem}_3d_poster.png"
        fig3d.savefig(poster_3d, bbox_inches="tight")
        poster_3d_pdf = out_dir / f"{output_stem}_3d_poster.pdf"
        fig3d.savefig(poster_3d_pdf, bbox_inches="tight")
        outputs.extend([poster_3d, poster_3d_pdf])

    suffix, writer_name = _preferred_animation_writer()
    if include_2d and anim2d is not None:
        writer = animation.writers[writer_name](fps=fps)
        path2d = out_dir / f"{output_stem}_2d.{suffix}"
        anim2d.save(path2d, writer=writer, dpi=110)
        outputs.append(path2d)
    if include_3d and anim3d is not None:
        path3d = out_dir / f"{output_stem}_3d.{suffix}"
        writer3d = animation.writers[writer_name](fps=fps)
        anim3d.save(path3d, writer=writer3d, dpi=110)
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
    out_dir = _prepare_plot_output(out_dir)

    fig = plt.figure(figsize=(12, 5.8), constrained_layout=True)
    grid = fig.add_gridspec(1, 2, width_ratios=(1.0, 1.15))
    ax2d = fig.add_subplot(grid[0, 0])
    ax3d = fig.add_subplot(grid[0, 1], projection="3d")
    fig.suptitle(case_title, fontsize=16)
    _draw_geometry_preview(ax2d, ax3d, mesh, case_title=case_title, fluid_mask=fluid_mask)

    return _save_figure_pair(fig, out_dir, "geometry_preview")


def write_geometry_gallery_plots(
    items: list[tuple[str, StructuredMesh, jnp.ndarray | None]],
    out_dir: str | Path,
    *,
    title: str,
) -> list[Path]:
    out_dir = _prepare_plot_output(out_dir)

    fig = plt.figure(figsize=(14.2, 7.6), constrained_layout=True)
    grid = fig.add_gridspec(2, len(items), height_ratios=(1.0, 1.15))
    fig.suptitle(title, fontsize=18)

    for column, (item_title, mesh, fluid_mask) in enumerate(items):
        ax2d = fig.add_subplot(grid[0, column])
        ax3d = fig.add_subplot(grid[1, column], projection="3d")
        _draw_geometry_preview(ax2d, ax3d, mesh, case_title=item_title, fluid_mask=fluid_mask)

    return _save_figure_pair(fig, out_dir, "geometry_gallery")


def write_cross_section_field_plots(
    *,
    y: np.ndarray,
    z: np.ndarray,
    field: np.ndarray,
    out_dir: str | Path,
    title: str,
) -> list[Path]:
    out_dir = _prepare_plot_output(out_dir)
    yy, zz = np.meshgrid(y, z, indexing="ij")
    by = field[..., 1]
    bz = field[..., 2]
    bmag = np.sqrt(np.sum(field**2, axis=-1))

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), constrained_layout=True)
    fig.suptitle(title, fontsize=18)
    panels = [
        (by, r"$B_y$", "PuOr_r"),
        (bz, r"$B_z$", "RdBu_r"),
        (bmag, r"$|B|$", "viridis"),
    ]
    for ax, (values, label, cmap) in zip(axes.ravel()[:3], panels, strict=True):
        image = ax.pcolormesh(z, y, values, shading="auto", cmap=cmap)
        ax.set_xlabel("z")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        ax.set_title(label)
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    axq = axes.ravel()[3]
    stride_y = max(1, len(y) // 20)
    stride_z = max(1, len(z) // 20)
    axq.quiver(
        zz[::stride_y, ::stride_z],
        yy[::stride_y, ::stride_z],
        bz[::stride_y, ::stride_z],
        by[::stride_y, ::stride_z],
        color="#0f172a",
        pivot="mid",
        scale=max(float(np.max(bmag)), 1.0e-12) * 18.0,
    )
    axq.set_title("Transverse field directions")
    axq.set_xlabel("z")
    axq.set_ylabel("y")
    axq.set_aspect("equal")
    return _save_figure_pair(fig, out_dir, "field_preview")


def write_tabulated_field_reconstruction_plots(
    *,
    y: np.ndarray,
    z: np.ndarray,
    reference_field: np.ndarray,
    tabulated_field: np.ndarray,
    out_dir: str | Path,
    title: str = "Tabulated-field reconstruction against analytic reference",
) -> list[Path]:
    """Write a diagnostic panel comparing table interpolation with reference values."""

    out_dir = _prepare_plot_output(out_dir)
    y_values = np.asarray(y, dtype=float)
    z_values = np.asarray(z, dtype=float)
    reference = np.asarray(reference_field, dtype=float)
    sampled = np.asarray(tabulated_field, dtype=float)
    if reference.shape != sampled.shape or reference.ndim != 3 or reference.shape[-1] != 3:
        raise ValueError("reference_field and tabulated_field must have matching (..., 3) shapes")
    if reference.shape[:2] != (y_values.size, z_values.size):
        raise ValueError("field shapes must match the y/z coordinate lengths")

    ref_bmag = np.linalg.norm(reference, axis=-1)
    sampled_bmag = np.linalg.norm(sampled, axis=-1)
    magnitude_error = sampled_bmag - ref_bmag
    component_error = sampled[..., 2] - reference[..., 2]
    bmag_scale = max(float(np.max(np.abs(ref_bmag))), 1.0e-12)
    err_scale = max(float(np.max(np.abs(magnitude_error))), 1.0e-12)
    bz_err_scale = max(float(np.max(np.abs(component_error))), 1.0e-12)

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.8), constrained_layout=True)
    fig.suptitle(title, fontsize=18)

    panels = [
        (axes[0, 0], ref_bmag, r"analytic $|B|$", "viridis", None),
        (axes[0, 1], sampled_bmag, r"tabulated $|B|$ at solver points", "viridis", None),
        (axes[1, 0], magnitude_error / bmag_scale, r"relative $|B|$ error", "RdBu_r", max(err_scale / bmag_scale, 1.0e-12)),
        (axes[1, 1], component_error / bmag_scale, r"relative $B_z$ error", "RdBu_r", max(bz_err_scale / bmag_scale, 1.0e-12)),
    ]
    for ax, values, label, cmap, symmetric_scale in panels:
        if symmetric_scale is None:
            image = ax.pcolormesh(z_values, y_values, values, shading="auto", cmap=cmap)
        else:
            scale = max(float(symmetric_scale), 1.0e-12)
            image = ax.pcolormesh(z_values, y_values, values, shading="auto", cmap=cmap, vmin=-scale, vmax=scale)
        ax.set_title(label)
        ax.set_xlabel("z")
        ax.set_ylabel("y")
        ax.set_aspect("equal")
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    return _save_figure_pair(fig, out_dir, "tabulated_field_reconstruction")


def write_bent_pipe_overview_plots(
    solution,
    out_dir: Path,
    *,
    straight_solution=None,
    title: str = "LMX bent-pipe inductionless baseline",
) -> list[Path]:
    geometry = solution.problem.case.geometry
    if geometry.kind != "bent_pipe":
        raise ValueError("Bent-pipe overview plots require a bent_pipe solution")
    out_dir = _prepare_plot_output(out_dir)

    mesh = generate_bent_pipe_mesh(
        tube_radius=geometry.radius or 0.5 * geometry.width,
        bend_radius=geometry.bend_radius or max(geometry.length, geometry.width),
        bend_angle=geometry.bend_angle or (0.5 * np.pi),
        nx=geometry.nx,
        nr=geometry.nr or geometry.ny,
        ntheta=geometry.ntheta or geometry.nz,
    )
    points = np.asarray(mesh.point_coordinates, dtype=float)
    bundle = solution.bundle
    mid_index = int(bundle.u.shape[0] // 2)
    u_mid = np.asarray(bundle.u[mid_index], dtype=float)
    x_hist = np.asarray(bundle.x, dtype=float)
    field_scale = np.asarray(bundle.field_scale, dtype=float)
    mean_velocity = np.asarray(bundle.mean_velocity, dtype=float)
    charge_balance = np.asarray(bundle.charge_balance_residual, dtype=float)
    r_faces = np.asarray(mesh.y_faces, dtype=float)
    theta_faces = np.asarray(mesh.z_faces, dtype=float)
    yy = r_faces[:, None] * np.cos(theta_faces[None, :])
    zz = r_faces[:, None] * np.sin(theta_faces[None, :])
    r_centers = np.asarray(bundle.y, dtype=float)
    theta_centers = np.asarray(bundle.z, dtype=float)
    yyc = r_centers[:, None] * np.cos(theta_centers[None, :])
    zzc = r_centers[:, None] * np.sin(theta_centers[None, :])
    u_peak = max(float(np.max(np.abs(u_mid))), 1.0e-12)
    u_mid_display = u_mid / u_peak
    mean_velocity_display = mean_velocity / max(float(np.max(np.abs(mean_velocity))), 1.0e-12)
    norm = colors.Normalize(vmin=0.0, vmax=1.0)
    cmap = plt.get_cmap("coolwarm")

    fig = plt.figure(figsize=(13.4, 9.4), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, height_ratios=(1.05, 1.0))
    ax3d = fig.add_subplot(grid[0, 0], projection="3d")
    ax_cross = fig.add_subplot(grid[0, 1])
    ax_hist = fig.add_subplot(grid[1, 0])
    ax_cmp = fig.add_subplot(grid[1, 1])
    fig.suptitle(title, fontsize=18)

    shell = points[:, -1, :, :]
    ax3d.plot_surface(
        shell[:, :, 0],
        shell[:, :, 1],
        shell[:, :, 2],
        color="#cbd5e1",
        alpha=0.14,
        linewidth=0.0,
        antialiased=False,
        shade=False,
    )
    centerline = points[:, 0, 0, :]
    ax3d.plot(centerline[:, 0], centerline[:, 1], centerline[:, 2], color="#111827", linewidth=2.0)
    section = points[mid_index]
    ax3d.plot_surface(
        section[:, :, 0],
        section[:, :, 1],
        section[:, :, 2],
        facecolors=cmap(norm(u_mid_display)),
        linewidth=0.0,
        antialiased=False,
        shade=False,
    )
    ax3d.text(
        float(centerline[mid_index, 0]),
        float(centerline[mid_index, 1]),
        float(centerline[mid_index, 2] + 1.35 * (geometry.radius or 0.5 * geometry.width)),
        "mid-bend profile",
        color="#111827",
        fontsize=11,
    )
    ax3d.set_title("Curved centerline and mid-bend profile slab")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.view_init(elev=22, azim=-55)

    image = ax_cross.pcolormesh(yy, zz, u_mid_display, shading="auto", cmap="coolwarm", vmin=0.0, vmax=1.0)
    ax_cross.contour(yyc, zzc, u_mid_display, levels=np.linspace(0.2, 0.95, 5), colors="white", linewidths=0.6, alpha=0.75)
    ax_cross.plot(r_faces * 0.0, r_faces, color="#111827", linewidth=1.1, linestyle="--", alpha=0.85, label="local z-cut")
    ax_cross.set_title("Mid-bend axial velocity")
    ax_cross.set_xlabel("local y")
    ax_cross.set_ylabel("local z")
    ax_cross.set_aspect("equal")
    ax_cross.legend(loc="lower left")
    fig.colorbar(image, ax=ax_cross, fraction=0.046, pad=0.04, label=r"$u/u_{peak}$")

    line_field, = ax_hist.plot(x_hist, field_scale, color="#1d4ed8", label=r"$B/B_{max}$")
    line_mean, = ax_hist.plot(x_hist, mean_velocity_display, color="#0f766e", label=r"$\bar{u}/\bar{u}_{max}$")
    ax_hist.set_title("Arc-length response")
    ax_hist.set_xlabel("s")
    ax_hist.set_ylabel("Response")
    ax_hist_right = ax_hist.twinx()
    line_charge, = ax_hist_right.semilogy(
        x_hist,
        np.maximum(charge_balance, 1.0e-16),
        color="#7c3aed",
        linestyle="--",
        label="Charge balance",
    )
    ax_hist_right.set_ylabel("Residual")
    ax_hist.legend(
        [line_field, line_mean, line_charge],
        [line_field.get_label(), line_mean.get_label(), r"Charge balance residual"],
        loc="upper left",
    )

    theta_index = 0
    opposite = (theta_index + len(bundle.z) // 2) % len(bundle.z)
    signed_r = np.concatenate([-r_centers[::-1], r_centers[1:]])
    bent_cut = np.concatenate([u_mid[:, opposite][::-1], u_mid[1:, theta_index]])
    bent_norm = bent_cut / u_peak
    ax_cmp.plot(signed_r, bent_norm, color="#b91c1c", label="Bent-pipe baseline")
    if straight_solution is not None:
        straight_mid = np.asarray(straight_solution.bundle.u[mid_index], dtype=float)
        straight_cut = np.concatenate([straight_mid[:, opposite][::-1], straight_mid[1:, theta_index]])
        straight_norm = straight_cut / u_peak
        ax_cmp.plot(signed_r, straight_norm, color="#111827", linestyle="--", label="Straight-pipe limit")
    else:
        straight_norm = bent_norm
    ax_cmp.set_title("Local centerline cut")
    ax_cmp.set_xlabel("signed local radius")
    ax_cmp.set_ylabel(r"$u/u_{peak}$")
    ax_cmp.legend(loc="lower center")

    inset = inset_axes(ax_cmp, width="45%", height="45%", loc="lower left", borderpad=1.2)
    edge_window = max(3, len(signed_r) // 8)
    inset.plot(signed_r[-edge_window:], bent_norm[-edge_window:], color="#b91c1c")
    if straight_solution is not None:
        inset.plot(signed_r[-edge_window:], straight_norm[-edge_window:], color="#111827", linestyle="--")
    inset.set_xlim(float(signed_r[-edge_window]), float(signed_r[-1]))
    inset.set_ylim(float(min(np.min(bent_norm[-edge_window:]), np.min(straight_norm[-edge_window:]))) - 0.02, 1.02)
    inset.set_title("wall-layer zoom", fontsize=10)
    inset.tick_params(labelsize=9)
    mark_inset(ax_cmp, inset, loc1=2, loc2=4, fc="none", ec="#6b7280", linewidth=0.8)

    return _save_figure_pair(fig, out_dir, "bent_pipe_overview")


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
        cross_section_label = "Bent-pipe cross-section" if mesh.geometry == "bent_pipe" else "Pipe cross-section"
        ax2d.set_title(f"{case_title}\n{cross_section_label}")
        ax2d.set_xlabel("y")
        ax2d.set_ylabel("z")
        ax2d.set_aspect("equal")
        ax2d.scatter([0.0], [0.0], color="#111827", s=16, zorder=5)
        ax2d.text(
            0.02,
            0.98,
            "O-grid lines follow r and θ" if mesh.geometry != "bent_pipe" else "Mapped bend section in local r, θ",
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
        domain_label = "Bent-pipe domain" if mesh.geometry == "bent_pipe" else "Pipe domain"
        ax3d.set_title(f"{case_title}\n{domain_label}")
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
    out_dir = _prepare_plot_output(out_dir)

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
    coord_y_label = r"$\theta$" if bundle.geometry_kind == "pipe_ogrid" else "z"

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
    fig.suptitle(case_title, fontsize=16)

    axes[0, 0].plot(x, mean_velocity, color="#0f766e", label="Mean velocity")
    axes[0, 0].plot(x, current_proxy, color="#b45309", linestyle="--", label="Current proxy")
    axes[0, 0].plot(x, field_scale, color="#1d4ed8", alpha=0.7, label="Field scale")
    axes[0, 0].set_title("Station response")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].legend(loc="upper left", bbox_to_anchor=(0.02, 0.98))

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
    axes[0, 1].legend(loc="upper left", bbox_to_anchor=(0.02, 0.98))

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

    return _save_figure_pair(fig, out_dir, "extruded_overview")


def write_magnetic_obstacle_benchmark_plots(
    solution,
    reference_solution,
    out_dir: str | Path,
    *,
    case_title: str,
) -> list[Path]:
    out_dir = _prepare_plot_output(out_dir)

    bundle = solution.bundle
    reference_bundle = reference_solution.bundle
    x = np.asarray(bundle.x, dtype=float)
    field_scale = np.asarray(bundle.field_scale, dtype=float)
    mean_velocity = np.asarray(bundle.mean_velocity, dtype=float)
    reference_mean_velocity = np.asarray(reference_bundle.mean_velocity, dtype=float)
    current_proxy = np.asarray(bundle.current_scaled_pressure_proxy, dtype=float)
    pressure_span = np.max(np.asarray(bundle.p, dtype=float), axis=(1, 2)) - np.min(np.asarray(bundle.p, dtype=float), axis=(1, 2))
    reference_pressure_span = np.max(np.asarray(reference_bundle.p, dtype=float), axis=(1, 2)) - np.min(np.asarray(reference_bundle.p, dtype=float), axis=(1, 2))
    velocity_ratio = mean_velocity / np.maximum(reference_mean_velocity, 1.0e-12)
    pressure_excess = np.maximum(pressure_span - reference_pressure_span, 0.0)
    peak_index = int(np.argmax(field_scale)) if field_scale.size else 0
    mid_y = int(bundle.u.shape[1] // 2)
    mid_z = int(bundle.u.shape[2] // 2)
    y = np.asarray(bundle.y, dtype=float)
    z = np.asarray(bundle.z, dtype=float)
    y_edges = _centers_to_edges(y)
    z_edges = _centers_to_edges(z)

    u_peak = np.asarray(bundle.u[peak_index], dtype=float)
    y_cut = np.asarray(bundle.u[peak_index, :, mid_z], dtype=float)
    y_cut_ref = np.asarray(reference_bundle.u[peak_index, :, mid_z], dtype=float)
    z_cut = np.asarray(bundle.u[peak_index, mid_y, :], dtype=float)
    z_cut_ref = np.asarray(reference_bundle.u[peak_index, mid_y, :], dtype=float)
    shared_cut_scale = max(
        float(np.max(np.abs(y_cut_ref))),
        float(np.max(np.abs(z_cut_ref))),
        float(np.max(np.abs(y_cut))),
        float(np.max(np.abs(z_cut))),
        1.0e-12,
    )
    peak_u = max(float(np.max(np.abs(u_peak))), 1.0e-12)
    center_velocity = np.asarray(bundle.u[:, mid_y, mid_z], dtype=float)
    ref_center_velocity = np.asarray(reference_bundle.u[:, mid_y, mid_z], dtype=float)
    center_deficit = np.maximum(
        (ref_center_velocity - center_velocity) / np.maximum(np.abs(ref_center_velocity), 1.0e-12),
        0.0,
    )
    peak_centerline_deficit = float(np.max(center_deficit)) if center_deficit.size else 0.0
    recovery_station = float(x[-1]) if x.size else 0.0
    if center_deficit.size:
        threshold = max(0.1 * peak_centerline_deficit, 1.0e-6)
        tail = np.where(center_deficit[peak_index:] <= threshold)[0]
        if tail.size:
            recovery_station = float(x[peak_index + int(tail[0])])

    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.8), constrained_layout=True)
    fig.suptitle(case_title, fontsize=16)

    ax = axes[0, 0]
    ax.plot(x, field_scale / max(float(np.max(field_scale)), 1.0e-12), color="#1d4ed8", label=r"$B/B_{max}$")
    ax.plot(x, velocity_ratio, color="#0f766e", label=r"$\bar{u}/\bar{u}_{ref}$")
    ax.plot(x, 1.0 - velocity_ratio, color="#b45309", linestyle="--", label="velocity deficit ratio")
    ax.plot(x, center_deficit, color="#dc2626", linestyle="-.", label="centerline deficit ratio")
    ax.axvline(x[peak_index], color="#64748b", linestyle=":", linewidth=1.0, label="peak-field station")
    ax.axvline(recovery_station, color="#7c3aed", linestyle=":", linewidth=1.0, label="recovery station")
    ax.set_title("Obstacle response along x")
    ax.set_xlabel("x")
    ax.set_ylabel("Normalized response")
    ax.legend(loc="lower left", fontsize=9)

    ax = axes[0, 1]
    ax.plot(x, pressure_excess, color="#7c3aed", label="pressure excess")
    ax.plot(x, current_proxy, color="#dc2626", linestyle="--", label="current proxy")
    ax.set_title("Pressure and current response")
    ax.set_xlabel("x")
    ax.set_ylabel("Response")
    ax.legend(loc="upper left", fontsize=9)

    ax = axes[1, 0]
    im = ax.pcolormesh(z_edges, y_edges, u_peak / peak_u, shading="auto", cmap="RdBu_r", vmin=0.0, vmax=1.0)
    ax.contour(z, y, u_peak / peak_u, levels=np.linspace(0.2, 0.95, 5), colors="white", linewidths=0.8, alpha=0.75)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$u/u_{peak}$")
    ax.set_title(f"Peak-field cross-section (x={x[peak_index]:.2f})")
    ax.set_xlabel("z")
    ax.set_ylabel("y")

    ax = axes[1, 1]
    ax.plot(y, y_cut_ref / shared_cut_scale, color="#64748b", label="reference y-cut")
    ax.plot(y, y_cut / shared_cut_scale, color="#1d4ed8", linestyle="--", label="field y-cut")
    ax.plot(z, z_cut_ref / shared_cut_scale, color="#94a3b8", label="reference z-cut")
    ax.plot(z, z_cut / shared_cut_scale, color="#b45309", linestyle="--", label="field z-cut")
    ax.set_title("Peak-field centerline cuts")
    ax.set_xlabel("local coordinate")
    ax.set_ylabel(r"$u/u_{ref,peak}$")
    ax.legend(loc="lower center", ncol=2, fontsize=9)

    return _save_figure_pair(fig, out_dir, "magnetic_obstacle_benchmark")


def write_magnetic_obstacle_schematic_plots(
    solution,
    reference_solution,
    out_dir: str | Path,
    *,
    case_title: str = "Magnetic-obstacle localized-field setup",
) -> list[Path]:
    """Write a setup-first magnetic-obstacle panel for docs and README use."""

    out_dir = _prepare_plot_output(out_dir)

    bundle = solution.bundle
    reference_bundle = reference_solution.bundle
    x = np.asarray(bundle.x, dtype=float)
    y = np.asarray(bundle.y, dtype=float)
    z = np.asarray(bundle.z, dtype=float)
    field_scale = np.asarray(bundle.field_scale, dtype=float)
    if x.size == 0 or y.size == 0 or z.size == 0:
        raise ValueError("Magnetic-obstacle schematic requires non-empty x/y/z coordinates")
    y_edges = _centers_to_edges(y)
    z_edges = _centers_to_edges(z)
    peak_index = int(np.argmax(field_scale))
    peak_x = float(x[peak_index])

    yy, zz = np.meshgrid(y, z, indexing="ij")
    field_fn = solution.problem.case.magnetic_field.fn
    if field_fn is None:
        bmag_cross = np.ones_like(yy)
    else:
        field_cross = np.asarray(field_fn(jnp.asarray(yy), jnp.asarray(zz)), dtype=float)
        bmag_cross = np.linalg.norm(field_cross, axis=-1)
    bmag_cross = bmag_cross / max(float(np.max(np.abs(bmag_cross))), 1.0e-12)

    x_dense = np.linspace(float(x[0]), float(x[-1]), 120)
    z_dense = np.linspace(float(z_edges[0]), float(z_edges[-1]), 80)
    xx_dense, zz_dense = np.meshgrid(x_dense, z_dense, indexing="ij")
    axial_scale = np.interp(x_dense, x, field_scale / max(float(np.max(field_scale)), 1.0e-12))
    obstacle_sheet = axial_scale[:, None] * np.exp(-((zz_dense / max(0.32 * (z_edges[-1] - z_edges[0]), 1.0e-12)) ** 2))

    u_peak = np.asarray(bundle.u[peak_index], dtype=float)
    u_ref_peak = np.asarray(reference_bundle.u[peak_index], dtype=float)
    u_norm = u_peak / max(float(np.max(np.abs(u_peak))), 1.0e-12)
    deficit = np.maximum((u_ref_peak - u_peak) / np.maximum(np.abs(u_ref_peak), 1.0e-12), 0.0)
    deficit_scale = max(float(np.max(deficit)), 1.0e-12)
    center_y = int(u_peak.shape[0] // 2)
    center_z = int(u_peak.shape[1] // 2)
    center_deficit = np.maximum(
        (np.asarray(reference_bundle.u[:, center_y, center_z], dtype=float) - np.asarray(bundle.u[:, center_y, center_z], dtype=float))
        / np.maximum(np.abs(np.asarray(reference_bundle.u[:, center_y, center_z], dtype=float)), 1.0e-12),
        0.0,
    )

    fig = plt.figure(figsize=(14.2, 8.6), constrained_layout=True)
    grid = fig.add_gridspec(2, 2, width_ratios=[1.25, 1.0], height_ratios=[1.0, 1.0])
    ax3d = fig.add_subplot(grid[0, 0], projection="3d")
    ax_diagram = fig.add_subplot(grid[0, 1])
    ax_deficit = fig.add_subplot(grid[1, 0])
    ax_response = fig.add_subplot(grid[1, 1])
    fig.suptitle(case_title, fontsize=18)

    cmap_field = plt.cm.viridis
    cmap_velocity = plt.cm.magma
    norm_unit = colors.Normalize(vmin=0.0, vmax=1.0)

    # Transparent duct shell.
    x0, x1 = float(x[0]), float(x[-1])
    y0, y1 = float(y_edges[0]), float(y_edges[-1])
    z0, z1 = float(z_edges[0]), float(z_edges[-1])
    for y_face in (y0, y1):
        xx_box, zz_box = np.meshgrid([x0, x1], [z0, z1], indexing="ij")
        ax3d.plot_surface(xx_box, np.full_like(xx_box, y_face), zz_box, color="#cbd5e1", alpha=0.08, linewidth=0.0, shade=False)
    for z_face in (z0, z1):
        xx_box, yy_box = np.meshgrid([x0, x1], [y0, y1], indexing="ij")
        ax3d.plot_surface(xx_box, yy_box, np.full_like(xx_box, z_face), color="#cbd5e1", alpha=0.08, linewidth=0.0, shade=False)
    for yy_edge in (y0, y1):
        for zz_edge in (z0, z1):
            ax3d.plot([x0, x1], [yy_edge, yy_edge], [zz_edge, zz_edge], color="#64748b", linewidth=0.8, alpha=0.8)
    for xx_edge in (x0, x1):
        for yy_edge in (y0, y1):
            ax3d.plot([xx_edge, xx_edge], [yy_edge, yy_edge], [z0, z1], color="#64748b", linewidth=0.8, alpha=0.8)
        for zz_edge in (z0, z1):
            ax3d.plot([xx_edge, xx_edge], [y0, y1], [zz_edge, zz_edge], color="#64748b", linewidth=0.8, alpha=0.8)

    ax3d.plot_surface(
        xx_dense,
        np.zeros_like(xx_dense),
        zz_dense,
        facecolors=cmap_field(norm_unit(obstacle_sheet)),
        linewidth=0.0,
        antialiased=False,
        shade=False,
        alpha=0.70,
    )
    yy_plane, zz_plane = np.meshgrid(y, z, indexing="ij")
    ax3d.plot_surface(
        np.full_like(yy_plane, peak_x),
        yy_plane,
        zz_plane,
        facecolors=cmap_velocity(norm_unit(u_norm)),
        linewidth=0.0,
        antialiased=False,
        shade=False,
        alpha=0.98,
    )
    ax3d.quiver(x0, 0.0, z1 + 0.12 * (z1 - z0), x1 - x0, 0.0, 0.0, color="#111827", arrow_length_ratio=0.08, linewidth=2.0)
    ax3d.text(0.5 * (x0 + x1), 0.0, z1 + 0.20 * (z1 - z0), "flow direction", ha="center", color="#111827", fontsize=10)
    ax3d.set_title("Duct, localized magnetic field, and velocity slice")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.view_init(elev=24, azim=-58)
    ax3d.set_box_aspect((x1 - x0, y1 - y0, z1 - z0))

    image = ax_diagram.pcolormesh(x_dense, z_dense, obstacle_sheet.T, shading="auto", cmap="viridis", vmin=0.0, vmax=1.0)
    ax_diagram.add_patch(Rectangle((x0, z0), x1 - x0, z1 - z0, fill=False, edgecolor="#111827", linewidth=1.4))
    ax_diagram.axvline(peak_x, color="#f8fafc", linestyle="--", linewidth=1.2)
    ax_diagram.annotate("localized B obstacle", xy=(peak_x, 0.0), xytext=(x0 + 0.12 * (x1 - x0), z1 - 0.15 * (z1 - z0)), arrowprops={"arrowstyle": "->", "color": "#111827"}, color="#111827")
    ax_diagram.annotate("u_x inlet", xy=(x0 + 0.18 * (x1 - x0), z0 + 0.18 * (z1 - z0)), xytext=(x0 + 0.02 * (x1 - x0), z0 + 0.18 * (z1 - z0)), arrowprops={"arrowstyle": "->", "color": "#111827"}, color="#111827")
    ax_diagram.set_title("Centerplane field diagram")
    ax_diagram.set_xlabel("x")
    ax_diagram.set_ylabel("z")
    fig.colorbar(image, ax=ax_diagram, fraction=0.046, pad=0.04, label=r"$B/B_{max}$")

    im_def = ax_deficit.pcolormesh(z_edges, y_edges, deficit / deficit_scale, shading="auto", cmap="magma", vmin=0.0, vmax=1.0)
    ax_deficit.contour(z, y, u_norm, levels=np.linspace(0.2, 0.95, 5), colors="white", linewidths=0.7, alpha=0.8)
    ax_deficit.set_title(f"Velocity deficit at peak field (x={peak_x:.2f})")
    ax_deficit.set_xlabel("z")
    ax_deficit.set_ylabel("y")
    ax_deficit.set_aspect("equal")
    fig.colorbar(im_def, ax=ax_deficit, fraction=0.046, pad=0.04, label="normalized deficit")

    ax_response.plot(x, field_scale / max(float(np.max(field_scale)), 1.0e-12), color="#1d4ed8", label=r"$B/B_{max}$")
    ax_response.plot(x, center_deficit, color="#dc2626", label="centerline velocity deficit")
    ax_response.axvline(peak_x, color="#64748b", linestyle=":", linewidth=1.0, label="peak field")
    ax_response.set_title("Obstacle response")
    ax_response.set_xlabel("x")
    ax_response.set_ylabel("normalized response")
    ax_response.legend(loc="upper right")

    return _save_figure_pair(fig, out_dir, "magnetic_obstacle_schematic")


def write_wham_mirror_overview_plots(
    solution,
    *,
    table_path: str | Path,
    pipe_radius: float,
    coil_separation: float,
    out_dir: str | Path,
    case_title: str,
    coil_inner_radius: float = 0.5 * 86.0e-3,
    coil_outer_radius: float = 0.5 * 730.0e-3,
    autodiff_summary: dict[str, object] | None = None,
) -> list[Path]:
    from .field_models import load_tabulated_field, sample_tabulated_field_volume

    out_dir = _prepare_plot_output(out_dir)

    table = load_tabulated_field(table_path)
    x_axis = np.asarray(table["x"], dtype=float)
    z_axis = np.asarray(table["z"], dtype=float)
    x_dense = np.linspace(float(x_axis[0]), float(x_axis[-1]), 181)
    z_dense = np.linspace(float(z_axis[0]), float(z_axis[-1]), 181)
    xx, zz = np.meshgrid(x_dense, z_dense, indexing="ij")
    centerplane = np.asarray(
        sample_tabulated_field_volume(
            table_path,
            x=xx,
            y=np.zeros_like(xx),
            z=zz,
        ),
        dtype=float,
    )
    bmag_center = np.linalg.norm(centerplane, axis=-1)
    peak_bmag = max(float(np.max(bmag_center)), 1.0e-12)

    bundle = solution.bundle
    x = np.asarray(bundle.x, dtype=float)
    field_scale = np.asarray(bundle.field_scale, dtype=float)
    mean_velocity = np.asarray(bundle.mean_velocity, dtype=float)
    current_proxy = np.asarray(bundle.current_scaled_pressure_proxy, dtype=float)
    pressure_span = np.max(np.asarray(bundle.p, dtype=float), axis=(1, 2)) - np.min(np.asarray(bundle.p, dtype=float), axis=(1, 2))
    peak_index = int(np.argmax(field_scale)) if field_scale.size else 0
    peak_station_x = float(x[peak_index]) if x.size else 0.0

    u_peak = np.asarray(bundle.u[peak_index], dtype=float)
    nr, ntheta = u_peak.shape
    radial = np.linspace(0.0, pipe_radius, nr)
    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    rr, tt = np.meshgrid(radial, theta, indexing="ij")
    disk_y = rr * np.cos(tt)
    disk_z = rr * np.sin(tt)
    disk_x = np.full_like(disk_y, peak_station_x)
    u_peak_norm = np.clip(u_peak / max(float(np.max(np.abs(u_peak))), 1.0e-12), 0.0, None)

    fig = plt.figure(figsize=(15.6, 9.2), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.35, 1.0], height_ratios=[1.0, 1.0, 1.0])
    ax3d = fig.add_subplot(gs[:, 0], projection="3d")
    ax_contour = fig.add_subplot(gs[0, 1])
    ax_response = fig.add_subplot(gs[1, 1])
    ax_autodiff = fig.add_subplot(gs[2, 1])
    fig.suptitle(case_title, fontsize=17)

    cmap_field = plt.cm.viridis
    norm_field = colors.Normalize(vmin=0.0, vmax=peak_bmag)
    cmap_velocity = plt.cm.magma
    norm_velocity = colors.Normalize(vmin=0.0, vmax=1.0)

    x_cyl = np.linspace(float(x_axis[0]), float(x_axis[-1]), 160)
    theta_cyl = np.linspace(0.0, 2.0 * np.pi, 120)
    x_cyl_grid, theta_cyl_grid = np.meshgrid(x_cyl, theta_cyl, indexing="ij")
    y_cyl = pipe_radius * np.cos(theta_cyl_grid)
    z_cyl = pipe_radius * np.sin(theta_cyl_grid)
    ax3d.plot_surface(
        x_cyl_grid,
        y_cyl,
        z_cyl,
        color="#cbd5e1",
        alpha=0.12,
        linewidth=0.0,
        antialiased=False,
        shade=False,
    )
    ax3d.plot(
        [float(x_axis[0]), float(x_axis[-1])],
        [0.0, 0.0],
        [0.0, 0.0],
        color="#111827",
        linewidth=1.8,
        alpha=0.85,
    )

    phi = np.linspace(0.0, 2.0 * np.pi, 240)
    for z_center, style in ((-0.5 * coil_separation, "#1f2937"), (0.5 * coil_separation, "#1f2937")):
        ax3d.plot(
            coil_outer_radius * np.cos(phi),
            coil_outer_radius * np.sin(phi),
            np.full_like(phi, z_center),
            color=style,
            linewidth=1.8,
            alpha=0.85,
        )
        ax3d.plot(
            coil_inner_radius * np.cos(phi),
            coil_inner_radius * np.sin(phi),
            np.full_like(phi, z_center),
            color="#94a3b8",
            linewidth=1.2,
            alpha=0.85,
        )

    y_sheet = np.zeros_like(xx)
    ax3d.plot_surface(
        xx,
        y_sheet,
        zz,
        facecolors=cmap_field(norm_field(bmag_center)),
        linewidth=0.0,
        antialiased=False,
        shade=False,
        alpha=0.74,
    )
    ax3d.plot_surface(
        disk_x,
        disk_y,
        disk_z,
        facecolors=cmap_velocity(norm_velocity(u_peak_norm)),
        linewidth=0.0,
        antialiased=True,
        shade=False,
        alpha=0.98,
    )
    ax3d.text(
        peak_station_x,
        0.0,
        1.15 * pipe_radius,
        "velocity cross-section",
        color="#111827",
        ha="center",
    )
    ax3d.text(
        float(x_axis[0]) + 0.04 * float(x_axis[-1] - x_axis[0]),
        0.0,
        0.95 * float(z_axis[-1]),
        "centerplane |B| contours",
        color="#111827",
    )
    ax3d.set_title("3D mirror field, pipe, and solved velocity slice")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    radial_extent = max(pipe_radius * 1.3, coil_outer_radius * 1.15)
    z_extent = max(float(np.max(np.abs(z_axis))), 0.5 * coil_separation + 0.15)
    ax3d.set_xlim(float(x_axis[0]), float(x_axis[-1]))
    ax3d.set_ylim(-radial_extent, radial_extent)
    ax3d.set_zlim(-z_extent, z_extent)
    ax3d.set_box_aspect((float(x_axis[-1] - x_axis[0]), 2.0 * radial_extent, 2.0 * z_extent))
    ax3d.view_init(elev=22, azim=-57)

    contour_levels = np.linspace(0.0, peak_bmag, 12)
    im = ax_contour.contourf(x_dense, z_dense, bmag_center.T, levels=contour_levels, cmap="viridis")
    ax_contour.contour(x_dense, z_dense, bmag_center.T, levels=contour_levels[2:-1:2], colors="white", linewidths=0.8, alpha=0.75)
    ax_contour.add_patch(
        Rectangle(
            (float(x_axis[0]), -pipe_radius),
            float(x_axis[-1] - x_axis[0]),
            2.0 * pipe_radius,
            fill=False,
            edgecolor="#111827",
            linewidth=1.4,
            linestyle="--",
        )
    )
    ax_contour.axhline(-0.5 * coil_separation, color="#1f2937", linestyle=":", linewidth=1.1)
    ax_contour.axhline(0.5 * coil_separation, color="#1f2937", linestyle=":", linewidth=1.1)
    ax_contour.text(float(x_axis[0]), 0.5 * coil_separation + 0.03, "coil centers", color="#1f2937", va="bottom")
    ax_contour.set_title(r"Midplane $|B(x,0,z)|$")
    ax_contour.set_xlabel("x")
    ax_contour.set_ylabel("z")
    plt.colorbar(im, ax=ax_contour, fraction=0.046, pad=0.04, label=r"$|B|$")

    mean_velocity_norm = mean_velocity / max(float(mean_velocity[0]), 1.0e-12)
    current_proxy_norm = current_proxy / max(float(np.max(np.abs(current_proxy))), 1.0e-12)
    pressure_span_norm = pressure_span / max(float(np.max(np.abs(pressure_span))), 1.0e-12)
    ax_response.plot(x, field_scale / max(float(np.max(field_scale)), 1.0e-12), color="#1d4ed8", label=r"$B/B_{max}$")
    ax_response.plot(x, mean_velocity_norm, color="#0f766e", label=r"$\bar{u}/\bar{u}_{in}$")
    ax_response.plot(x, current_proxy_norm, color="#b45309", linestyle="--", label="current proxy / peak")
    ax_response.plot(x, pressure_span_norm, color="#7c3aed", linestyle=":", label=r"$\Delta p/\Delta p_{max}$")
    ax_response.set_title("Executable pipe response")
    ax_response.set_xlabel("x")
    ax_response.set_ylabel("Normalized response")
    ax_response.legend(loc="best")

    if autodiff_summary is not None:
        separation = np.asarray(autodiff_summary["separation_sweep"], dtype=float)
        pressure_curve = np.asarray(autodiff_summary["pressure_drop_curve"], dtype=float)
        sensitivity_curve = np.asarray(autodiff_summary["sensitivity_curve"], dtype=float)
        reference_separation = float(autodiff_summary["reference_separation"])
        ax_autodiff.plot(separation, pressure_curve, color="#0f766e", marker="o", label="pressure-drop proxy")
        ax_autodiff.axvline(reference_separation, color="#111827", linestyle="--", linewidth=1.0)
        ax_autodiff.set_xlabel("coil separation")
        ax_autodiff.set_ylabel("proxy", color="#0f766e")
        ax_autodiff.tick_params(axis="y", labelcolor="#0f766e")
        ax_autodiff.set_title("Autodiff sensitivity to coil separation")
        ax_sens = ax_autodiff.twinx()
        ax_sens.plot(separation, sensitivity_curve, color="#7c3aed", marker="s", label=r"$d(\Delta p)/ds$")
        ax_sens.set_ylabel(r"$d(\Delta p)/ds$", color="#7c3aed")
        ax_sens.tick_params(axis="y", labelcolor="#7c3aed")
        handles = [
            Line2D([], [], color="#0f766e", marker="o", label="pressure-drop proxy"),
            Line2D([], [], color="#7c3aed", marker="s", label=r"$d(\Delta p)/ds$"),
        ]
        ax_autodiff.legend(handles=handles, loc="best")
    else:
        ax_autodiff.text(0.5, 0.5, "Autodiff summary not provided", ha="center", va="center", transform=ax_autodiff.transAxes)
        ax_autodiff.set_axis_off()

    return _save_figure_pair(fig, out_dir, "wham_mirror_overview")


def write_magnetic_obstacle_regime_plots(
    records: list[dict[str, float]],
    out_dir: str | Path,
    *,
    case_title: str,
) -> list[Path]:
    out_dir = _prepare_plot_output(out_dir)
    if not records:
        raise ValueError("Need at least one record to write magnetic-obstacle regime plots")

    forcings = sorted({float(row["forcing"]) for row in records})
    base_bz_values = sorted({float(row["base_bz"]) for row in records})
    forcing_index = {value: idx for idx, value in enumerate(forcings)}
    field_index = {value: idx for idx, value in enumerate(base_bz_values)}

    def build_grid(key: str) -> np.ndarray:
        grid = np.full((len(base_bz_values), len(forcings)), np.nan, dtype=float)
        for row in records:
            grid[field_index[float(row["base_bz"])] , forcing_index[float(row["forcing"])]] = float(row[key])
        return grid

    peak_velocity = build_grid("peak_velocity_deficit_ratio")
    pressure_proxy = build_grid("pressure_excess_proxy")
    current_proxy = build_grid("current_proxy_peak")
    distortion = 0.5 * (build_grid("y_l2_distortion") + build_grid("z_l2_distortion"))

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 9.2), constrained_layout=True)
    fig.suptitle(case_title, fontsize=16)

    panels = (
        (peak_velocity, "Peak velocity-deficit ratio", "viridis"),
        (pressure_proxy, "Pressure-excess proxy", "magma"),
        (current_proxy, "Current proxy peak", "plasma"),
        (distortion, "Mean cross-cut distortion", "cividis"),
    )
    for ax, (grid, title, cmap_name) in zip(axes.ravel(), panels, strict=True):
        image = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap_name)
        ax.set_title(title)
        ax.set_xticks(range(len(forcings)), [f"{value:g}" for value in forcings])
        ax.set_yticks(range(len(base_bz_values)), [f"{value:g}" for value in base_bz_values])
        ax.set_xlabel("forcing")
        ax.set_ylabel(r"$B_z$ scale")
        for iy in range(grid.shape[0]):
            for ix in range(grid.shape[1]):
                value = grid[iy, ix]
                if np.isfinite(value):
                    ax.text(ix, iy, f"{value:.2e}", ha="center", va="center", fontsize=9, color="white")
        plt.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    return _save_figure_pair(fig, out_dir, "magnetic_obstacle_regime_scan")


def write_strong_scaling_plots(
    records: list[dict[str, object]],
    out_dir: str | Path,
    *,
    case_title: str,
    resource_label: str = "Device count",
) -> list[Path]:
    """Plot warm runtime and speedup against a named execution-resource count."""

    out_dir = _prepare_plot_output(out_dir)

    groups: dict[str, list[dict[str, object]]] = {}
    for record in records:
        groups.setdefault(str(record["platform"]), []).append(record)
    for values in groups.values():
        values.sort(key=lambda item: int(item["num_devices"]))

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.2), constrained_layout=False)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.23, top=0.82, wspace=0.30)
    fig.suptitle(case_title, fontsize=15, y=0.96)
    palette = ["#0f766e", "#1d4ed8", "#b45309", "#7c3aed"]

    legend_handles: list[Line2D] = []
    legend_labels: list[str] = []
    for color, (platform_name, values) in zip(palette, groups.items(), strict=False):
        device_counts = np.asarray([int(item["num_devices"]) for item in values], dtype=float)
        runtimes = np.asarray([float(item["warm_seconds"] if "warm_seconds" in item
            else item["mean_seconds"]) for item in values], dtype=float)
        baseline = runtimes[0]
        speedup = baseline / np.maximum(runtimes, 1.0e-12)
        ny_value = values[0].get("ny")
        nz_value = values[0].get("nz")
        nx_value = values[0].get("nx")
        iteration_value = values[0].get("iterations")
        platform_label = str(platform_name)
        if ny_value is None or nz_value is None:
            label = platform_label
        else:
            ny = int(ny_value)
            nz = int(nz_value)
            benchmark_kind = str(values[0].get("benchmark_kind", ""))
            shape_text = f"{ny}×{nz}" if nx_value in (None, 0) else f"{int(nx_value)}×{ny}×{nz}"
            if iteration_value is None:
                label = f"{platform_label} ({shape_text})"
            else:
                suffix = "" if not benchmark_kind else f", {benchmark_kind}"
                label = f"{platform_label}: {shape_text}, {int(iteration_value)} iters{suffix}"

        axes[0].plot(device_counts, runtimes, marker="o", color=color, label=label)
        axes[1].plot(device_counts, speedup, marker="o", color=color, label=label)
        legend_handles.append(Line2D([0], [0], color=color, marker="o", label=label))
        legend_labels.append(label)

    ideal_device_counts = np.asarray(sorted({int(item["num_devices"]) for item in records}), dtype=float)
    axes[1].plot(
        ideal_device_counts,
        ideal_device_counts / ideal_device_counts[0],
        linestyle="--",
        color="#64748b",
        alpha=0.85,
        label="Ideal linear speedup",
    )
    legend_handles.append(Line2D([0], [0], color="#64748b", linestyle="--", label="Ideal linear speedup"))
    legend_labels.append("Ideal linear speedup")

    axes[0].set_title("Warm runtime", fontsize=13)
    axes[0].set_xlabel(resource_label)
    axes[0].set_ylabel("Runtime [s]")
    axes[0].set_yscale("log")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(sorted({int(item["num_devices"]) for item in records}))
    axes[0].get_xaxis().set_major_formatter(ScalarFormatter())
    axes[1].set_title("Strong-scaling speedup", fontsize=13)
    axes[1].set_xlabel(resource_label)
    axes[1].set_ylabel("Warm-runtime speedup")
    axes[1].set_xscale("log", base=2)
    axes[1].set_xticks(sorted({int(item["num_devices"]) for item in records}))
    axes[1].get_xaxis().set_major_formatter(ScalarFormatter())
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        frameon=True,
        fontsize=9.5,
    )

    return _save_figure_pair(fig, out_dir, "strong_scaling")


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
    out_dir = _prepare_plot_output(out_dir)

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
    axes[0].legend(loc="upper right", bbox_to_anchor=(0.98, 0.98))
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
    axes[1].legend(
        lines_left + lines_right,
        labels_left + labels_right,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.08),
        ncol=3,
    )

    return _save_figure_pair(fig, out_dir, "autodiff_summary")


def write_operator_verification_plots(
    records: list[dict[str, float]],
    out_dir: str | Path,
    *,
    case_title: str,
) -> list[Path]:
    out_dir = _prepare_plot_output(out_dir)

    resolution = np.asarray([float(item["resolution"]) for item in records], dtype=float)
    spacing = np.asarray([float(item["max_spacing"]) for item in records], dtype=float)
    gradient_y = np.asarray([float(item["gradient_y_l2_error"]) for item in records], dtype=float)
    gradient_z = np.asarray([float(item["gradient_z_l2_error"]) for item in records], dtype=float)
    laplacian = np.asarray([float(item["laplacian_l2_error"]) for item in records], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), constrained_layout=True)
    fig.suptitle(case_title, fontsize=16)

    axes[0].loglog(spacing, gradient_y, marker="o", color="#0f766e", label=r"$\partial_y$ error")
    axes[0].loglog(spacing, gradient_z, marker="s", color="#b45309", label=r"$\partial_z$ error")
    axes[0].loglog(spacing, laplacian, marker="^", color="#1d4ed8", label=r"$\nabla^2$ error")
    axes[0].set_xlabel("Max cell spacing")
    axes[0].set_ylabel(r"$L_2$ error")
    axes[0].set_title("Observed-order convergence")
    axes[0].invert_xaxis()
    axes[0].legend(loc="upper left")

    grad_order_y = np.log(gradient_y[:-1] / gradient_y[1:]) / np.log(spacing[:-1] / spacing[1:]) if len(records) > 1 else np.asarray([])
    grad_order_z = np.log(gradient_z[:-1] / gradient_z[1:]) / np.log(spacing[:-1] / spacing[1:]) if len(records) > 1 else np.asarray([])
    lap_order = np.log(laplacian[:-1] / laplacian[1:]) / np.log(spacing[:-1] / spacing[1:]) if len(records) > 1 else np.asarray([])
    axes[0].text(
        0.03,
        0.03,
        "\n".join(
            [
                f"grad-y order ≈ {float(np.mean(grad_order_y)):.2f}" if grad_order_y.size else "",
                f"grad-z order ≈ {float(np.mean(grad_order_z)):.2f}" if grad_order_z.size else "",
                f"laplacian order ≈ {float(np.mean(lap_order)):.2f}" if lap_order.size else "",
            ]
        ).strip(),
        transform=axes[0].transAxes,
        ha="left",
        va="bottom",
        fontsize=10.5,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.92},
    )

    axes[1].plot(resolution, gradient_y, marker="o", color="#0f766e", label=r"$\partial_y$")
    axes[1].plot(resolution, gradient_z, marker="s", color="#b45309", label=r"$\partial_z$")
    axes[1].plot(resolution, laplacian, marker="^", color="#1d4ed8", label=r"$\nabla^2$")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Cross-section resolution")
    axes[1].set_ylabel(r"$L_2$ error")
    axes[1].set_title("Error decay with refinement")
    axes[1].legend(loc="upper right")

    return _save_figure_pair(fig, out_dir, "operator_verification")


def write_interface_verification_plots(
    records: list[dict[str, float]],
    profile: dict[str, np.ndarray],
    out_dir: str | Path,
    *,
    case_title: str,
    interface_location: float = 0.0,
) -> list[Path]:
    out_dir = _prepare_plot_output(out_dir)

    spacing = np.asarray([float(item["max_spacing"]) for item in records], dtype=float)
    profile_error = np.asarray([float(item["profile_l2_error"]) for item in records], dtype=float)
    flux_error = np.asarray([float(item["flux_error"]) for item in records], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0), constrained_layout=True)
    fig.suptitle(case_title, fontsize=16)

    axes[0].plot(profile["y"], profile["u_exact"], color="#111827", linewidth=2.0, label="Exact")
    axes[0].plot(profile["y"], profile["u_numeric"], color="#1d4ed8", linestyle="--", linewidth=2.0, label="LMX harmonic-average FV")
    axes[0].axvline(interface_location, color="#7c3aed", linestyle=":", linewidth=1.5, label="Conductivity jump")
    axes[0].set_xlabel("Cross-stream coordinate")
    axes[0].set_ylabel("Potential / scalar state")
    axes[0].set_title("Piecewise-linear interface solution")
    axes[0].legend(loc="upper left")

    axes[1].loglog(spacing, profile_error, marker="o", color="#0f766e", label="Profile $L_2$ error")
    axes[1].loglog(spacing, flux_error, marker="s", color="#b45309", label="Flux error")
    axes[1].set_xlabel("Max cell spacing")
    axes[1].set_ylabel("Error")
    axes[1].set_title("Layered-media convergence")
    axes[1].invert_xaxis()
    axes[1].legend(loc="upper left")

    if len(records) > 1 and np.all(profile_error > 1e-12) and np.all(flux_error > 1e-12):
        profile_order = np.log(profile_error[:-1] / profile_error[1:]) / np.log(spacing[:-1] / spacing[1:])
        flux_order = np.log(flux_error[:-1] / flux_error[1:]) / np.log(spacing[:-1] / spacing[1:])
        axes[1].text(
            0.03,
            0.03,
            "\n".join(
                [
                    f"profile order ≈ {float(np.mean(profile_order)):.2f}",
                    f"flux order ≈ {float(np.mean(flux_order)):.2f}",
                ]
            ),
            transform=axes[1].transAxes,
            ha="left",
            va="bottom",
            fontsize=10.5,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.92},
        )
    else:
        axes[1].text(
            0.03,
            0.03,
            "Roundoff-limited exact reproduction\non the aligned interface case",
            transform=axes[1].transAxes,
            ha="left",
            va="bottom",
            fontsize=10.5,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.92},
        )

    return _save_figure_pair(fig, out_dir, "interface_verification")


def write_freemhd_parity_plots(
    records: list[dict[str, object]],
    out_dir: str | Path,
    *,
    case_title: str,
) -> list[Path]:
    out_dir = _prepare_plot_output(out_dir)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2), constrained_layout=True)
    fig.suptitle(case_title, fontsize=16)

    case_axes = [axes[0, 0], axes[0, 1]]
    colors_case = {"y": "#1d4ed8", "z": "#dc2626"}
    for ax, record in zip(case_axes, records[:2], strict=False):
        case_name = str(record["case_kind"]).capitalize()
        for axis, label in (("y", "Transverse cut"), ("z", "Vertical cut")):
            profile = record[f"{axis}_profile"]
            coordinate = np.asarray(profile["coordinate"], dtype=float)
            simulated = np.asarray(profile["simulated"], dtype=float)
            reference = np.asarray(profile["reference"], dtype=float)
            color = colors_case[axis]
            ax.plot(coordinate, reference, color=color, linewidth=2.0, label=f"{label} FreeMHD")
            ax.plot(coordinate, simulated, color=color, linewidth=1.9, linestyle="--", label=f"{label} LMX")
        ax.set_title(
            f"{case_name} parity\n"
            f"$L_2(y)$={float(record['y_l2_error']):.2e}, "
            f"$L_2(z)$={float(record['z_l2_error']):.2e}"
        )
        ax.set_xlabel("Normalized coordinate")
        ax.set_ylabel(r"$u/u_{max}$")
        ax.set_xlim(-1.0, 1.0)
        ax.set_ylim(-0.02, 1.08)
        ax.legend(loc="lower center", ncol=2, fontsize=10)

    names = [str(record["case_kind"]).capitalize() for record in records]
    freemhd_times = np.asarray([float(record["freemhd_execution_seconds"]) for record in records], dtype=float)
    lmx_times = np.asarray([float(record["lmx_execution_seconds"]) for record in records], dtype=float)

    ax = axes[1, 0]
    colors_runtime = {"Shercliff": "#0f766e", "Hunt": "#7c3aed"}
    for record in records:
        case_name = str(record["case_kind"]).capitalize()
        color = colors_runtime.get(case_name, "#0f766e")
        freemhd_history = record.get("freemhd_u_max_history", {})
        lmx_history = record.get("lmx_u_max_history", {})
        if freemhd_history:
            ax.plot(
                np.asarray(freemhd_history["time"], dtype=float),
                np.asarray(freemhd_history["value"], dtype=float),
                color=color,
                linewidth=2.1,
                label=f"{case_name} FreeMHD",
            )
        if lmx_history:
            ax.plot(
                np.asarray(lmx_history["time"], dtype=float),
                np.asarray(lmx_history["value"], dtype=float),
                color=color,
                linewidth=1.9,
                linestyle="--",
                label=f"{case_name} LMX",
            )
    ax.set_xlabel("Time [s]")
    ax.set_ylabel(r"$u_{max}$")
    ax.set_title("Transient peak-velocity evolution")
    ax.legend(loc="best", fontsize=10, ncol=2)

    ax = axes[1, 1]
    x_positions = np.arange(len(names), dtype=float)
    width = 0.34
    ax.bar(x_positions - 0.5 * width, freemhd_times, width=width, color="#0f766e", label="FreeMHD wall time")
    ax.bar(x_positions + 0.5 * width, lmx_times, width=width, color="#b45309", label="LMX wall time")
    ax.set_xticks(x_positions, names)
    ax.set_ylabel("Seconds")
    ax.set_title("Runtime comparison on the same host")
    for index, record in enumerate(records):
        ax.text(
            x_positions[index],
            max(freemhd_times[index], lmx_times[index]) * 1.03,
            (
                rf"$L_2(y)$={float(record['y_l2_error']):.2e}" "\n"
                rf"$L_2(z)$={float(record['z_l2_error']):.2e}" "\n"
                rf"$|u_{{max}}|$ diff={float(record['u_max_abs_diff']):.2e}"
            ),
            ha="center",
            va="bottom",
            fontsize=10,
            bbox={"boxstyle": "round,pad=0.24", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.92},
        )
    ax.legend(loc="upper right", fontsize=10)

    return _save_figure_pair(fig, out_dir, "freemhd_closed_channel_parity")


def write_freemhd_observable_parity_plots(
    records: list[dict[str, object]],
    out_dir: str | Path,
    *,
    case_title: str,
    output_stem: str = "freemhd_closed_channel_observable_parity",
) -> list[Path]:
    out_dir = _prepare_plot_output(out_dir)

    fig, axes = plt.subplots(2, 4, figsize=(18.0, 8.8), constrained_layout=True)
    fig.suptitle(case_title, fontsize=16)
    cut_colors = {"y": "#1d4ed8", "z": "#dc2626"}
    observable_titles = {
        "velocity": r"Velocity $u/u_{peak}$",
        "potential": r"Potential $\phi/|\phi|_{max}$",
        "current": r"Cut-aligned current $J/|J|_{max}$",
        "lorentz": r"Lorentz $J\times B_x/|J\times B_x|_{max}$",
    }

    for row_index, record in enumerate(records[:2]):
        case_name = str(record["case_kind"]).capitalize()
        for column_index, observable in enumerate(("velocity", "potential", "current", "lorentz")):
            ax = axes[row_index, column_index]
            observable_payload = record["observables"][observable]
            for axis in ("y", "z"):
                cut = observable_payload[axis]
                coordinate = np.asarray(cut["coordinate"], dtype=float)
                reference = np.asarray(cut["reference"], dtype=float)
                simulated = np.asarray(cut["simulated"], dtype=float)
                color = cut_colors[axis]
                label_prefix = "Transverse" if axis == "y" else "Vertical"
                ax.plot(coordinate, reference, color=color, linewidth=2.0, label=f"{label_prefix} FreeMHD")
                ax.plot(coordinate, simulated, color=color, linestyle="--", linewidth=1.8, label=f"{label_prefix} LMX")
            metric_lines = []
            for axis in ("y", "z"):
                cut = observable_payload[axis]
                metric_lines.append(rf"$L_2({axis})$={float(cut['l2_error']):.2e}")
            peak_ratio = float(observable_payload["peak_ratio"])
            metric_lines.append(rf"peak ratio={peak_ratio:.3f}")
            ax.set_title(observable_titles[observable], fontsize=13)
            ax.text(
                0.98,
                0.04,
                "\n".join(metric_lines),
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=9.5,
                bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.92},
            )
            ax.set_xlim(-1.0, 1.0)
            ax.set_xlabel("Normalized coordinate")
            if column_index == 0:
                ax.set_ylabel(case_name)
            if row_index == 0 and column_index == 3:
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.30), ncol=2, fontsize=9.5)

    return _save_figure_pair(fig, out_dir, output_stem)
