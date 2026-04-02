from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib import colors
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np

from .core import Solution
from .validation import extract_midplane_profile


def _set_publication_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "grid.alpha": 0.18,
            "grid.linewidth": 0.5,
            "grid.color": "#4f4f4f",
            "legend.frameon": False,
            "legend.fontsize": 10,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
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
    _set_publication_style()
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
) -> list[Path]:
    if not frames:
        return []
    _set_publication_style()
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

    def _movie_field(index: int) -> np.ndarray:
        field = np.asarray(display_fields[index])
        if use_normalized_positive:
            return field / frame_peaks[index]
        return field

    fig2d, ax2d = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = ax2d.pcolormesh(mesh.z_faces, mesh.y_faces, _movie_field(0), shading="auto", cmap=cmap, norm=norm)
    ax2d.set_xlabel("z")
    ax2d.set_ylabel("y")
    effective_label = f"Normalized {movie_label.lower()}" if use_normalized_positive else movie_label
    ax2d.set_title(f"{case_title}\n2D {effective_label.lower()}")
    ax2d.set_aspect("equal")
    annotation_bbox = {"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "none", "alpha": 0.85}
    time_text = ax2d.text(0.02, 0.98, "", transform=ax2d.transAxes, ha="left", va="top", bbox=annotation_bbox)
    peak_text = ax2d.text(0.98, 0.98, "", transform=ax2d.transAxes, ha="right", va="top", bbox=annotation_bbox)
    effective_colorbar_label = f"{colorbar_label} / max|u(t)|" if use_normalized_positive else colorbar_label
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
        time_text.set_text(f"t = {times[index]:.2e}")
        peak_text.set_text(f"max|u| = {frame_peaks[index]:.2e}")
        return image, time_text, peak_text

    anim2d = animation.FuncAnimation(fig2d, update_2d, frames=len(frames), interval=1000 / fps, blit=False)
    update_2d(len(frames) - 1)
    poster_2d = out_dir / f"{output_stem}_2d_poster.png"
    fig2d.savefig(poster_2d, bbox_inches="tight")
    poster_2d_pdf = out_dir / f"{output_stem}_2d_poster.pdf"
    fig2d.savefig(poster_2d_pdf, bbox_inches="tight")

    y_centers = mesh.y_centers
    z_centers = mesh.z_centers
    zz, yy = np.meshgrid(np.asarray(z_centers), np.asarray(y_centers))
    fig3d = plt.figure(figsize=(8, 6), constrained_layout=True)
    ax3d = fig3d.add_subplot(111, projection="3d")

    def update_3d(index: int):
        ax3d.cla()
        field = _movie_field(index)
        surface = ax3d.plot_surface(
            zz,
            yy,
            field,
            cmap=cmap,
            norm=norm,
            linewidth=0,
            antialiased=True,
        )
        ax3d.set_xlabel("z")
        ax3d.set_ylabel("y")
        ax3d.set_zlabel(effective_colorbar_label)
        ax3d.set_title(f"{case_title} | 3D {effective_label.lower()}\n t = {times[index]:.2e} | max|u| = {frame_peaks[index]:.2e}")
        if use_normalized_positive:
            ax3d.set_zlim(0.0, 1.05)
        else:
            ax3d.set_zlim(-stack_abs_max, stack_abs_max)
        ax3d.view_init(elev=26, azim=38 + 8 * index)
        return (surface,)

    anim3d = animation.FuncAnimation(fig3d, update_3d, frames=len(frames), interval=1000 / fps, blit=False)
    update_3d(len(frames) - 1)
    poster_3d = out_dir / f"{output_stem}_3d_poster.png"
    fig3d.savefig(poster_3d, bbox_inches="tight")
    poster_3d_pdf = out_dir / f"{output_stem}_3d_poster.pdf"
    fig3d.savefig(poster_3d_pdf, bbox_inches="tight")

    outputs: list[Path] = [poster_2d, poster_2d_pdf, poster_3d, poster_3d_pdf]
    for suffix, writer_name in _safe_writer_candidates():
        writer = animation.writers[writer_name](fps=fps)
        path2d = out_dir / f"{output_stem}_2d.{suffix}"
        anim2d.save(path2d, writer=writer, dpi=140)
        outputs.append(path2d)
        path3d = out_dir / f"{output_stem}_3d.{suffix}"
        writer3d = animation.writers[writer_name](fps=fps)
        anim3d.save(path3d, writer=writer3d, dpi=140)
        outputs.append(path3d)

    plt.close(fig2d)
    plt.close(fig3d)
    return outputs
