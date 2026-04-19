from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, Polygon, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .cases import make_hunt_case, make_shercliff_case
from .example_runner import solve_case_snapshots
from .mesh import StructuredMesh, generate_layered_duct_mesh
from .plotting import write_transient_movies
from .reference_data import load_hunt_analytical, load_shercliff_analytical
from .solvers import solve_steady
from .validation import closed_channel_validation, extract_midplane_profile


def _set_showcase_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 12,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
        }
    )


def _add_slide_title(fig: plt.Figure, title: str) -> None:
    fig.text(0.03, 0.95, title, ha="left", va="top", fontsize=26, fontweight="bold", color="#2b2b2b")
    fig.add_artist(Rectangle((0.03, 0.885), 0.035, 0.008, transform=fig.transFigure, color="#c00000", clip_on=False))


def _fluid_crop(mesh: StructuredMesh, field: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if mesh.fluid_mask is None:
        return (
            np.asarray(mesh.y_faces, dtype=float),
            np.asarray(mesh.z_faces, dtype=float),
            np.asarray(mesh.y_centers, dtype=float),
            np.asarray(mesh.z_centers, dtype=float),
            np.asarray(field, dtype=float),
        )
    mask = np.asarray(mesh.fluid_mask, dtype=bool)
    y_idx = np.where(mask.any(axis=1))[0]
    z_idx = np.where(mask.any(axis=0))[0]
    y0, y1 = int(y_idx[0]), int(y_idx[-1]) + 1
    z0, z1 = int(z_idx[0]), int(z_idx[-1]) + 1
    return (
        np.asarray(mesh.y_faces[y0 : y1 + 1], dtype=float),
        np.asarray(mesh.z_faces[z0 : z1 + 1], dtype=float),
        np.asarray(mesh.y_centers[y0:y1], dtype=float),
        np.asarray(mesh.z_centers[z0:z1], dtype=float),
        np.asarray(field[y0:y1, z0:z1], dtype=float),
    )


def _normalized_fluid_field(solution) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y_faces, z_faces, y_centers, z_centers, field = _fluid_crop(solution.mesh, np.asarray(solution.state.u, dtype=float))
    scale = max(float(np.max(np.abs(field))), 1.0e-12)
    return y_faces, z_faces, y_centers, z_centers, field / scale


def _fluid_field(solution) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return _fluid_crop(solution.mesh, np.asarray(solution.state.u, dtype=float))


def _surface_colors(field: np.ndarray, *, cmap: str = "coolwarm", vmax: float = 1.0) -> np.ndarray:
    cmap_obj = plt.get_cmap(cmap)
    norm = colors.Normalize(vmin=0.0, vmax=vmax)
    return cmap_obj(norm(field))


def solve_closed_channel_benchmark(
    case_kind: str,
    *,
    ha: float = 20.0,
    width: float = 0.2,
    height: float = 0.2,
    ny: int = 96,
    nz: int = 96,
    wall_cells: int = 10,
    wall_thickness: float = 0.02,
    fluid_conductivity: float = 1.0e6,
    density: float = 1.0e4,
    viscosity: float = 1.0e-3,
    conducting_wall_conductivity: float = 1.0e7,
    insulating_wall_conductivity: float = 1.0e-6,
    coupling_iterations: int = 16,
    potential_iterations: int = 160,
    max_steps: int = 160,
):
    if case_kind == "shercliff":
        case = make_shercliff_case(
            ha=ha,
            width=width,
            height=height,
            ny=ny,
            nz=nz,
            conductivity=fluid_conductivity,
            density=density,
            viscosity=viscosity,
        )
    elif case_kind == "hunt":
        case = make_hunt_case(
            ha=ha,
            width=width,
            height=height,
            ny=ny,
            nz=nz,
            wall_cells=wall_cells,
            wall_thickness=wall_thickness,
            fluid_conductivity=fluid_conductivity,
            wall_conductivity=conducting_wall_conductivity,
            insulator_conductivity=insulating_wall_conductivity,
            density=density,
            viscosity=viscosity,
        )
    else:
        raise ValueError(f"Unsupported case kind {case_kind!r}")

    case = replace(
        case,
        solver=replace(case.solver, coupling_iterations=coupling_iterations, coupling_tolerance=1.0e-9),
        time_stepper=replace(
            case.time_stepper,
            max_steps=max_steps,
            potential_iterations=potential_iterations,
            steady_tolerance=1.0e-9,
            potential_tolerance=1.0e-9,
            potential_relaxation=1.0,
        ),
    )
    solution = solve_steady(case)
    comparison = closed_channel_validation(solution, case_kind, int(ha))
    return case, solution, comparison


def write_lm_duct_geometry_setup_figure(
    out_dir: str | Path,
    *,
    width: float = 0.2,
    height: float = 0.2,
    fluid_conductivity: float = 1.0e6,
    conducting_wall_conductivity: float = 1.0e7,
    insulating_wall_conductivity: float = 1.0e-6,
    density: float = 1.0e4,
    viscosity: float = 1.0e-3,
) -> list[Path]:
    _set_showcase_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12.8, 6.8), constrained_layout=False)
    _add_slide_title(fig, "Geometry Setup - LM Duct Flow")
    ax = fig.add_axes([0.03, 0.08, 0.94, 0.78])
    ax.set_axis_off()

    front = np.asarray([[0.23, 0.22], [0.40, 0.12], [0.40, 0.56], [0.23, 0.66]])
    top = np.asarray([[0.23, 0.66], [0.40, 0.56], [0.68, 0.78], [0.51, 0.88]])
    side = np.asarray([[0.40, 0.12], [0.68, 0.34], [0.68, 0.78], [0.40, 0.56]])

    ax.add_patch(Polygon(front, closed=True, facecolor="#b000a8", edgecolor="#facc15", linewidth=1.5))
    ax.add_patch(Polygon(top, closed=True, facecolor="#c00000", edgecolor="none", alpha=0.95))
    ax.add_patch(Polygon(side, closed=True, facecolor="#4b2e83", edgecolor="none", alpha=0.95))

    ax.text(0.33, 0.39, "Fluid\nσ=1e+6 S/m\nρ=1e+4 kg/m³\nμ=1e-3 Pa·s", color="white", fontsize=20, ha="center", va="center", weight="bold")
    ax.text(0.095, 0.72, "Shercliff", fontsize=22, color="#2b2b2b")
    ax.text(0.80, 0.72, "Hunt", fontsize=22, color="#2b2b2b")
    ax.text(0.09, 0.59, f"σ={insulating_wall_conductivity:.0e} S/m", fontsize=18, color="#2b2b2b")
    ax.text(0.80, 0.59, f"σ={insulating_wall_conductivity:.0e} S/m", fontsize=18, color="#2b2b2b")
    ax.text(0.80, 0.47, f"σ={conducting_wall_conductivity:.0e} S/m", fontsize=18, color="#2b2b2b")
    ax.text(0.25, 0.10, f"{width:.2f} m × {height:.2f} m", fontsize=22, color="#2b2b2b", rotation=-18)

    arrow_style = dict(arrowstyle="->", mutation_scale=18, color="#111827", linewidth=2.0)
    ax.add_patch(FancyArrowPatch((0.20, 0.61), (0.37, 0.60), **arrow_style))
    ax.add_patch(FancyArrowPatch((0.20, 0.61), (0.44, 0.43), **arrow_style))
    ax.add_patch(FancyArrowPatch((0.79, 0.61), (0.58, 0.73), **arrow_style))
    ax.add_patch(FancyArrowPatch((0.79, 0.48), (0.53, 0.41), **arrow_style))

    png_path = out_dir / "lm_duct_geometry_setup.png"
    pdf_path = out_dir / "lm_duct_geometry_setup.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def write_structured_mesh_figure(
    mesh: StructuredMesh,
    out_dir: str | Path,
    *,
    title: str = "Test Case Mesh - Ha = 20",
    nx: int = 64,
    length: float = 1.0,
) -> list[Path]:
    _set_showcase_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12.8, 6.8))
    _add_slide_title(fig, title)
    ax_cross = fig.add_axes([0.39, 0.28, 0.33, 0.48])
    ax_long = fig.add_axes([0.15, 0.08, 0.58, 0.22])
    ax_inset = fig.add_axes([0.76, 0.56, 0.18, 0.22])

    for ax in (ax_cross, ax_inset):
        for y in np.asarray(mesh.y_faces):
            ax.plot([mesh.z_faces[0], mesh.z_faces[-1]], [y, y], color="#1f3b99", linewidth=0.35, alpha=0.75)
        for z in np.asarray(mesh.z_faces):
            ax.plot([z, z], [mesh.y_faces[0], mesh.y_faces[-1]], color="#1f3b99", linewidth=0.35, alpha=0.75)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_edgecolor("#111827")

    ax_inset.set_xlim(float(mesh.z_faces[-3]), float(mesh.z_faces[-1]))
    ax_inset.set_ylim(float(mesh.y_faces[-3]), float(mesh.y_faces[-1]))

    ax_long.set_axis_off()
    duct = Rectangle((0.0, 0.0), length, 1.0, facecolor="#4b2e83", alpha=0.95)
    ax_long.add_patch(duct)
    for x in np.linspace(0.0, length, nx + 1):
        ax_long.plot([x, x], [0.0, 1.0], color="#1f3b99", linewidth=0.25, alpha=0.85)
    for y_line in np.linspace(0.0, 1.0, mesh.ny + 1):
        ax_long.plot([0.0, length], [y_line, y_line], color="#1f3b99", linewidth=0.25, alpha=0.85)
    ax_long.set_xlim(-0.02 * length, 1.02 * length)
    ax_long.set_ylim(-0.05, 1.05)

    dy = np.diff(np.asarray(mesh.y_faces, dtype=float))
    dz = np.diff(np.asarray(mesh.z_faces, dtype=float))
    min_spacing = min(float(np.min(dy)), float(np.min(dz)))
    max_spacing = max(float(np.max(dy)), float(np.max(dz)))
    expansion_ratio = max_spacing / max(min_spacing, 1.0e-12)
    total_cells = int(nx * mesh.ny * mesh.nz)
    half_width = 0.5 * (float(mesh.y_faces[-1] - mesh.y_faces[0]))
    target_delta = 0.05 * half_width
    cells_in_bl = int(np.count_nonzero(np.asarray(mesh.y_centers) > (mesh.y_faces[-1] - target_delta)))

    fig.text(0.07, 0.72, f"Cells across Hartmann BL ≈ {cells_in_bl}", fontsize=22, color="#2b2b2b")
    fig.text(0.07, 0.66, f"BL thickness ≈ {target_delta / half_width * 100:.0f} %", fontsize=22, color="#2b2b2b")
    fig.text(0.07, 0.60, f"Expansion ratio ≈ {expansion_ratio:.1f}", fontsize=22, color="#2b2b2b")
    fig.text(0.07, 0.54, f"Cell total ≈ {total_cells:,}", fontsize=22, color="#2b2b2b")
    fig.text(0.07, 0.48, "Mesh type = structured hex mesh", fontsize=22, color="#2b2b2b")
    fig.text(0.77, 0.43, "Boundary-layer\nclustering at the wall", fontsize=18, ha="left", va="center", color="#2b2b2b")

    link = ConnectionPatch(xyA=(0.96, 0.96), coordsA=ax_cross.transAxes, xyB=(0.02, 0.78), coordsB=ax_inset.transAxes, arrowstyle="->", mutation_scale=20, color="#111827", linewidth=2.0)
    fig.add_artist(link)

    png_path = out_dir / "structured_mesh_ha20.png"
    pdf_path = out_dir / "structured_mesh_ha20.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def write_boundary_layer_figure(solution, out_dir: str | Path, *, title: str, cmap: str = "coolwarm") -> list[Path]:
    _set_showcase_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y_faces, z_faces, _, _, field = _fluid_field(solution)
    vmax = max(float(np.max(field)), 1.0e-12)

    fig, ax = plt.subplots(figsize=(10.5, 6.7))
    _add_slide_title(fig, title)
    image = ax.pcolormesh(z_faces, y_faces, field, shading="auto", cmap=cmap, vmin=0.0, vmax=1.05 * vmax)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(1.6)
        spine.set_edgecolor("#8c8c8c")
    cbar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.06, pad=0.08)
    cbar.set_label("Velocity magnitude")
    png_path = out_dir / "boundary_layer_development.png"
    pdf_path = out_dir / "boundary_layer_development.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def write_annotated_layer_figure(
    solution,
    out_dir: str | Path,
    *,
    title: str,
    case_kind: str,
    ha: float,
    half_width: float,
) -> list[Path]:
    _set_showcase_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y_faces, z_faces, _, _, field = _fluid_field(solution)
    vmax = max(float(np.max(field)), 1.0e-12)

    delta_ha = half_width / ha
    delta_side = half_width / np.sqrt(ha)
    fig, ax = plt.subplots(figsize=(12.5, 7.0))
    _add_slide_title(fig, title)
    image = ax.pcolormesh(z_faces, y_faces, field, shading="auto", cmap="turbo", vmin=0.0, vmax=1.05 * vmax)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])

    zmin, zmax = float(z_faces[0]), float(z_faces[-1])
    ymin, ymax = float(y_faces[0]), float(y_faces[-1])
    hartmann_left = Rectangle((zmin, ymin), delta_ha, ymax - ymin, fill=False, edgecolor="#dc2626", linewidth=2.0)
    hartmann_right = Rectangle((zmax - delta_ha, ymin), delta_ha, ymax - ymin, fill=False, edgecolor="#dc2626", linewidth=2.0)
    side_bottom = Rectangle((zmin, ymin), zmax - zmin, delta_side, fill=False, edgecolor="#111827", linewidth=2.0)
    side_top = Rectangle((zmin, ymax - delta_side), zmax - zmin, delta_side, fill=False, edgecolor="#111827", linewidth=2.0)
    for patch in (hartmann_left, hartmann_right, side_bottom, side_top):
        ax.add_patch(patch)

    side_label = "Shercliff\nLayers" if case_kind == "shercliff" else "Hunt/Side\nLayers"
    ax.annotate(
        side_label,
        xy=((zmin + zmax) * 0.25, ymax - 0.5 * delta_side),
        xytext=(zmin - 0.18 * (zmax - zmin), ymax + 0.02 * (ymax - ymin)),
        arrowprops={"arrowstyle": "-", "color": "#111827", "linewidth": 2.0},
        fontsize=18,
        ha="right",
        va="center",
    )
    ax.annotate(
        "Hartmann\nLayers",
        xy=(zmax - 0.5 * delta_ha, 0.0),
        xytext=(zmax + 0.28 * (zmax - zmin), 0.15 * (ymax - ymin)),
        arrowprops={"arrowstyle": "-", "color": "#111827", "linewidth": 2.0},
        fontsize=18,
        ha="left",
        va="center",
    )
    ax.text(
        zmax + 0.26 * (zmax - zmin),
        ymin + 0.05 * (ymax - ymin),
        f"δ_Ha = {delta_ha:.3f} m\nδ_s = {delta_side:.3f} m",
        fontsize=18,
        ha="left",
        va="bottom",
        color="#2b2b2b",
    )
    cbar = fig.colorbar(image, ax=ax, orientation="horizontal", fraction=0.06, pad=0.08)
    cbar.set_label(f"U magnitude - {case_kind.capitalize()} flow [Ha = {int(ha)}]")
    png_path = out_dir / "annotated_layers.png"
    pdf_path = out_dir / "annotated_layers.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def _draw_duct_box(ax, *, length: float, y_faces: np.ndarray, z_faces: np.ndarray) -> None:
    y0, y1 = float(y_faces[0]), float(y_faces[-1])
    z0, z1 = float(z_faces[0]), float(z_faces[-1])
    verts = [
        [(0.0, z0, y0), (length, z0, y0), (length, z1, y0), (0.0, z1, y0)],
        [(0.0, z0, y1), (length, z0, y1), (length, z1, y1), (0.0, z1, y1)],
        [(0.0, z0, y0), (length, z0, y0), (length, z0, y1), (0.0, z0, y1)],
        [(0.0, z1, y0), (length, z1, y0), (length, z1, y1), (0.0, z1, y1)],
    ]
    ax.add_collection3d(Poly3DCollection(verts, facecolors="#d9d9d9", edgecolors="none", alpha=0.18))
    for y in (y0, y1):
        for z in (z0, z1):
            ax.plot([0.0, length], [z, z], [y, y], color="#9ca3af", linewidth=1.0, alpha=0.7)


def _profile_surface_x(field: np.ndarray, *, length: float, x_plane: float, amplitude: float) -> np.ndarray:
    normalized = np.clip(np.asarray(field, dtype=float), 0.0, None)
    peak = max(float(np.max(normalized)), 1.0e-12)
    return x_plane + amplitude * normalized / peak


def write_velocity_profile_volume_figure(solution, out_dir: str | Path, *, title: str, case_kind: str, length: float = 1.0) -> list[Path]:
    _set_showcase_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y_faces, z_faces, y_centers, z_centers, field = _fluid_field(solution)
    zz, yy = np.meshgrid(z_centers, y_centers)
    vmax = max(float(np.max(field)), 1.0e-12)
    colors_rgba = _surface_colors(field, cmap="coolwarm", vmax=vmax)
    x_plane = 0.40 * length
    x_surface = _profile_surface_x(field / vmax, length=length, x_plane=x_plane, amplitude=0.20 * length)

    fig = plt.figure(figsize=(13.2, 6.8))
    _add_slide_title(fig, title)
    ax = fig.add_axes([0.06, 0.08, 0.72, 0.72], projection="3d")
    inset = fig.add_axes([0.79, 0.10, 0.19, 0.34], projection="3d")

    for target_ax in (ax, inset):
        _draw_duct_box(target_ax, length=length, y_faces=y_faces, z_faces=z_faces)
        target_ax.plot_surface(
            x_surface,
            zz,
            yy,
            facecolors=colors_rgba,
            shade=False,
            linewidth=0.25,
            edgecolor=(0.1, 0.1, 0.1, 0.12),
            antialiased=True,
        )
        target_ax.set_xlim(0.0, length)
        target_ax.set_ylim(float(z_faces[0]), float(z_faces[-1]))
        target_ax.set_zlim(float(y_faces[0]), float(y_faces[-1]))
        target_ax.set_box_aspect((4.0, 1.2, 1.2))
        target_ax.set_axis_off()

    ax.view_init(elev=12, azim=-92)
    inset.view_init(elev=20, azim=-130)
    cax = fig.add_axes([0.28, 0.14, 0.34, 0.03])
    sm = plt.cm.ScalarMappable(norm=colors.Normalize(vmin=0.0, vmax=vmax), cmap="coolwarm")
    cbar = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cbar.set_label("Velocity Magnitude")

    png_path = out_dir / "velocity_profile_volume.png"
    pdf_path = out_dir / "velocity_profile_volume.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def write_closed_channel_profile_comparison_figure(
    out_dir: str | Path,
    *,
    shercliff_solution,
    hunt_solution,
    ha: float = 20.0,
) -> list[Path]:
    _set_showcase_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shercliff_ref = load_shercliff_analytical(int(ha))
    hunt_ref = load_hunt_analytical(int(ha))
    shercliff_profile = extract_midplane_profile(shercliff_solution, axis="z", fluid_only=True)
    hunt_profile = extract_midplane_profile(hunt_solution, axis="z", fluid_only=True)
    shercliff_peak = max(float(np.max(np.asarray(shercliff_profile["u"], dtype=float))), 1.0e-12)
    hunt_peak = max(float(np.max(np.asarray(hunt_profile["u"], dtype=float))), 1.0e-12)
    shercliff_ref_peak = max(float(np.max(np.asarray(shercliff_ref.midplane_z, dtype=float))), 1.0e-12)
    hunt_ref_peak = max(float(np.max(np.asarray(hunt_ref.midplane_z, dtype=float))), 1.0e-12)
    shercliff_scaled = np.asarray(shercliff_profile["u"], dtype=float) * (shercliff_ref_peak / shercliff_peak)
    hunt_scaled = np.asarray(hunt_profile["u"], dtype=float) * (hunt_ref_peak / hunt_peak)

    fig, ax = plt.subplots(figsize=(10.2, 6.4))
    _add_slide_title(fig, f"LMX benchmarking: velocity profiles (Ha = {int(ha)})")
    ax.plot(
        np.asarray(shercliff_ref.coordinate),
        np.asarray(shercliff_ref.midplane_z),
        color="#111827",
        linestyle="--",
        linewidth=1.8,
        label="Analytical: Shercliff (insulating)",
    )
    ax.plot(
        np.asarray(shercliff_profile["z"]),
        shercliff_scaled,
        color="#1d4ed8",
        marker="x",
        markersize=5,
        linewidth=1.2,
        label="LMX: Shercliff (peak-matched)",
    )
    ax.plot(
        np.asarray(hunt_ref.coordinate),
        np.asarray(hunt_ref.midplane_z),
        color="#7f1d1d",
        linestyle="--",
        linewidth=1.8,
        label="Analytical: Hunt (conducting)",
    )
    ax.plot(
        np.asarray(hunt_profile["z"]),
        hunt_scaled,
        color="#dc2626",
        marker="o",
        markersize=3.2,
        linewidth=1.2,
        label="LMX: Hunt (peak-matched)",
    )
    ax.set_xlabel("Position (z) [m]")
    ax.set_ylabel("Streamwise Velocity u(x) [m/s]")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", frameon=True)
    png_path = out_dir / "analytic_velocity_profiles.png"
    pdf_path = out_dir / "analytic_velocity_profiles.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def write_closed_channel_startup_movies(
    case_kind: str,
    out_dir: str | Path,
    *,
    ha: float = 20.0,
    width: float = 0.2,
    height: float = 0.2,
    ny: int = 49,
    nz: int = 49,
    wall_cells: int = 6,
    fluid_conductivity: float = 1.0e6,
    density: float = 1.0e4,
    viscosity: float = 1.0e-3,
    conducting_wall_conductivity: float = 1.0e7,
    insulating_wall_conductivity: float = 1.0e-6,
    dt: float = 1.0e-5,
    t_final: float = 2.0e-3,
    coupling_iterations: int = 6,
    potential_iterations: int = 48,
    fps: int = 12,
) -> list[Path]:
    if case_kind == "shercliff":
        case = make_shercliff_case(
            ha=ha,
            width=width,
            height=height,
            ny=ny,
            nz=nz,
            conductivity=fluid_conductivity,
            density=density,
            viscosity=viscosity,
        )
    elif case_kind == "hunt":
        case = make_hunt_case(
            ha=ha,
            width=width,
            height=height,
            ny=ny,
            nz=nz,
            wall_cells=wall_cells,
            wall_thickness=0.02,
            fluid_conductivity=fluid_conductivity,
            wall_conductivity=conducting_wall_conductivity,
            insulator_conductivity=insulating_wall_conductivity,
            density=density,
            viscosity=viscosity,
        )
    else:
        raise ValueError(f"Unsupported startup movie case {case_kind!r}")

    steps = max(1, int(t_final / dt))
    case = replace(
        case,
        solver=replace(case.solver, coupling_iterations=coupling_iterations, coupling_tolerance=1.0e-8),
        time_stepper=replace(
            case.time_stepper,
            dt=dt,
            t_final=t_final,
            max_steps=steps,
            potential_iterations=potential_iterations,
            potential_tolerance=1.0e-8,
            potential_relaxation=1.0,
        ),
        initial_velocity=1.0,
    )
    frames = solve_case_snapshots(case, frame_count=steps)
    return write_transient_movies(
        frames,
        out_dir,
        case_title=f"LMX {case_kind.capitalize()} startup",
        output_stem=f"{case_kind}_startup",
        fps=fps,
        symmetry_average_axes=("y", "z"),
    )
