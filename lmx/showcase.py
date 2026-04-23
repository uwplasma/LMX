from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, Polygon, Rectangle
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .cases import make_hunt_case, make_shercliff_case
from .example_runner import solve_case_snapshots
from .mesh import StructuredMesh, generate_layered_duct_mesh
from .plotting import write_transient_movies
from .reference_data import load_hunt_analytical, load_shercliff_analytical
from .solvers import _build_mesh, solve_steady
from .specs import BoundaryCondition
from .validation import closed_channel_validation, extract_midplane_profile, hartmann_validation
from .core import MHDState


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
    potential_tolerance: float = 1.0e-9,
    max_steps: int = 160,
    velocity_update_limit: float | None = None,
    current_reconstruction: str = "face_averaged",
    drive_mode: str = "forcing",
    pressure_gradient: float | None = None,
    target_mean_velocity: float = 0.1,
    initial_profile: str = "analytic",
):
    reference = None
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

    if drive_mode == "flow_rate":
        case = replace(
            case,
            forcing=0.0,
            initial_velocity=target_mean_velocity,
            boundary_conditions=case.boundary_conditions
            + (
                BoundaryCondition(
                    "inlet",
                    "inlet_flow_rate",
                    value=target_mean_velocity * width * height,
                    axis="x",
                ),
            ),
        )
    elif drive_mode == "pressure_gradient":
        reference = load_shercliff_analytical(int(ha)) if case_kind == "shercliff" else load_hunt_analytical(int(ha))
        if pressure_gradient is None:
            pressure_gradient = reference.pressure_drop
        if pressure_gradient is None:
            raise ValueError(f"No analytical pressure-gradient reference is available for {case_kind} Ha={ha:g}")
        case = replace(case, forcing=float(pressure_gradient))
    elif drive_mode != "forcing":
        raise ValueError(f"Unsupported closed-channel benchmark drive mode {drive_mode!r}")

    initial_state = None
    if initial_profile == "analytic":
        if reference is None:
            reference = load_shercliff_analytical(int(ha)) if case_kind == "shercliff" else load_hunt_analytical(int(ha))
        mesh = _build_mesh(case)
        y_target = np.asarray(mesh.y_centers, dtype=float)
        z_target = np.asarray(mesh.z_centers, dtype=float)
        ref_coord = np.asarray(reference.coordinate, dtype=float)
        ref_y = np.asarray(reference.midplane_y, dtype=float)
        ref_z = np.asarray(reference.midplane_z, dtype=float)
        y_profile = np.interp(y_target, ref_coord, ref_y)
        z_profile = np.interp(z_target, ref_coord, ref_z)
        yz_field = np.outer(y_profile, z_profile)
        yz_scale = max(float(np.max(np.abs(yz_field))), 1.0e-12)
        yz_field = yz_field / yz_scale
        if mesh.fluid_mask is not None:
            yz_field = np.where(np.asarray(mesh.fluid_mask, dtype=bool), yz_field, 0.0)
        zeros = np.zeros_like(yz_field)
        initial_state = MHDState(
            u=jnp.asarray(yz_field, dtype=float),
            phi=jnp.asarray(zeros, dtype=float),
            jy=jnp.asarray(zeros, dtype=float),
            jz=jnp.asarray(zeros, dtype=float),
            lorentz_x=jnp.asarray(zeros, dtype=float),
            time=0.0,
            residual=0.0,
        )
    elif initial_profile != "zero":
        raise ValueError(f"Unsupported closed-channel benchmark initial profile {initial_profile!r}")

    case = replace(
        case,
        solver=replace(case.solver, coupling_iterations=coupling_iterations, coupling_tolerance=1.0e-9),
        time_stepper=replace(
            case.time_stepper,
            max_steps=max_steps,
            potential_iterations=potential_iterations,
            steady_tolerance=1.0e-9,
            potential_tolerance=potential_tolerance,
            steady_potential_tolerance=potential_tolerance,
            potential_relaxation=1.0,
            current_reconstruction=current_reconstruction,
            velocity_update_limit=case.time_stepper.velocity_update_limit if velocity_update_limit is None else velocity_update_limit,
        ),
    )
    solution = solve_steady(case, initial_state=initial_state)
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
    hartmann_solution,
    shercliff_solution,
    hunt_solution,
    ha: float = 20.0,
) -> list[Path]:
    _set_showcase_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hartmann_comparison = hartmann_validation(hartmann_solution, ha)
    shercliff_validation = closed_channel_validation(shercliff_solution, "shercliff", int(ha))
    hunt_validation = closed_channel_validation(hunt_solution, "hunt", int(ha))
    fig = plt.figure(figsize=(12.0, 8.8))
    _add_slide_title(fig, f"LMX benchmarking: straight-duct analytical validation (Ha = {int(ha)})")
    grid = fig.add_gridspec(2, 2, left=0.08, right=0.97, bottom=0.10, top=0.88, wspace=0.24, hspace=0.28)
    ax_hartmann = fig.add_subplot(grid[0, 0])
    ax_shercliff = fig.add_subplot(grid[0, 1])
    ax_hunt = fig.add_subplot(grid[1, 0])
    ax_summary = fig.add_subplot(grid[1, 1])

    hartmann_coord = np.asarray(hartmann_comparison.coordinate, dtype=float)
    hartmann_sim = np.asarray(hartmann_comparison.simulated, dtype=float)
    hartmann_ref = np.asarray(hartmann_comparison.reference, dtype=float)
    positive_mask = hartmann_coord >= 0.0

    ax_hartmann.plot(hartmann_coord, hartmann_ref, color="#111827", linestyle="--", linewidth=1.8, label="Analytical")
    ax_hartmann.plot(
        hartmann_coord,
        hartmann_sim,
        color="#b91c1c",
        marker="o",
        markersize=3.0,
        markerfacecolor="white",
        linewidth=1.4,
        label="LMX",
    )
    ax_hartmann.set_title("Hartmann centerline")
    ax_hartmann.set_xlabel("Normalized wall-normal coordinate")
    ax_hartmann.set_ylabel("u / max(u)")
    ax_hartmann.set_xlim(-1.02, 1.02)
    ax_hartmann.set_ylim(-0.02, 1.05)
    ax_hartmann.grid(True, alpha=0.25)
    ax_hartmann.legend(loc="lower right", frameon=True)
    ax_hartmann.text(
        0.03,
        0.97,
        (
            f"L2 = {hartmann_comparison.l2_error:.2e}\n"
            f"L∞ = {hartmann_comparison.linf_error:.2e}"
        ),
        transform=ax_hartmann.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.92},
    )
    inset = ax_hartmann.inset_axes([0.55, 0.18, 0.38, 0.40])
    inset.plot(hartmann_coord[positive_mask], hartmann_ref[positive_mask], color="#111827", linestyle="--", linewidth=1.5)
    inset.plot(
        hartmann_coord[positive_mask],
        hartmann_sim[positive_mask],
        color="#b91c1c",
        marker="o",
        markersize=2.2,
        markerfacecolor="white",
        linewidth=1.1,
    )
    inset.set_xlim(0.72, 1.02)
    inset.set_ylim(-0.02, min(0.42, 1.05 * max(np.max(hartmann_ref[positive_mask]), np.max(hartmann_sim[positive_mask]))))
    inset.set_title("Hartmann layer", fontsize=9)
    inset.grid(True, alpha=0.18)
    inset.tick_params(labelsize=8)

    def _plot_closed_channel_case(ax, title: str, validation, z_color: str, y_color: str) -> None:
        z_comp = validation.z_profile
        y_comp = validation.y_profile
        ax.plot(
            np.asarray(z_comp.coordinate, dtype=float),
            np.asarray(z_comp.reference, dtype=float),
            color=z_color,
            linestyle="--",
            linewidth=1.8,
            label="Analytical z-cut",
        )
        ax.plot(
            np.asarray(z_comp.coordinate, dtype=float),
            np.asarray(z_comp.simulated, dtype=float),
            color=z_color,
            marker="o",
            markersize=2.8,
            markerfacecolor="white",
            linewidth=1.3,
            label="LMX z-cut",
        )
        ax.plot(
            np.asarray(y_comp.coordinate, dtype=float),
            np.asarray(y_comp.reference, dtype=float),
            color=y_color,
            linestyle="--",
            linewidth=1.8,
            label="Analytical y-cut",
        )
        ax.plot(
            np.asarray(y_comp.coordinate, dtype=float),
            np.asarray(y_comp.simulated, dtype=float),
            color=y_color,
            marker="x",
            markersize=4.2,
            linewidth=1.3,
            label="LMX y-cut",
        )
        ax.set_title(title)
        ax.set_xlabel("Normalized coordinate")
        ax.set_ylabel("u / max(u)")
        ax.set_xlim(-1.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="lower right", frameon=True, ncol=2)

    _plot_closed_channel_case(ax_shercliff, "Shercliff duct", shercliff_validation, "#1d4ed8", "#0f766e")
    ax_shercliff.text(
        0.03,
        0.97,
        (
            f"y: L2={shercliff_validation.y_profile.l2_error:.2e}, L∞={shercliff_validation.y_profile.linf_error:.2e}\n"
            f"z: L2={shercliff_validation.z_profile.l2_error:.2e}, L∞={shercliff_validation.z_profile.linf_error:.2e}"
        ),
        transform=ax_shercliff.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.92},
    )

    _plot_closed_channel_case(ax_hunt, "Hunt duct", hunt_validation, "#dc2626", "#7c3aed")
    ax_hunt.text(
        0.03,
        0.97,
        (
            f"y: L2={hunt_validation.y_profile.l2_error:.2e}, L∞={hunt_validation.y_profile.linf_error:.2e}\n"
            f"z: L2={hunt_validation.z_profile.l2_error:.2e}, L∞={hunt_validation.z_profile.linf_error:.2e}"
        ),
        transform=ax_hunt.transAxes,
        va="top",
        ha="left",
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.92},
    )

    ax_summary.axis("off")
    summary_lines = [
        "Validation summary",
        "",
        "Literature anchors",
        "• Hartmann analytical profile",
        "• Shercliff insulating-duct analytical solution",
        "• Hunt conducting-wall analytical solution",
        "",
        "Current bounded errors",
        f"• Hartmann: L2={hartmann_comparison.l2_error:.2e}, L∞={hartmann_comparison.linf_error:.2e}",
        f"• Shercliff y/z: {shercliff_validation.y_profile.l2_error:.2e} / {shercliff_validation.z_profile.l2_error:.2e}",
        f"• Hunt y/z: {hunt_validation.y_profile.l2_error:.2e} / {hunt_validation.z_profile.l2_error:.2e}",
        "",
        "Reference data",
        f"• Shercliff: {Path(shercliff_validation.reference_path).name}",
        f"• Hunt: {Path(hunt_validation.reference_path).name}",
    ]
    ax_summary.text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        fontsize=12,
        bbox={"facecolor": "#f8fafc", "edgecolor": "#cbd5e1", "alpha": 0.98, "boxstyle": "round,pad=0.45"},
    )

    png_path = out_dir / "analytic_velocity_profiles.png"
    pdf_path = out_dir / "analytic_velocity_profiles.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def write_closed_channel_validation_ladder_figure(
    out_dir: str | Path,
    *,
    shercliff_records: list[dict[str, object]],
    hunt_records: list[dict[str, object]],
) -> list[Path]:
    _set_showcase_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.8))
    _add_slide_title(fig, "LMX benchmarking: straight-duct validation ladder")
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.88, wspace=0.22, hspace=0.28)

    ha_colors = {20: "#fde047", 100: "#84cc16", 1000: "#2563eb"}

    def _draw_case(ax, records, *, axis_key: str, title: str, xlabel: str) -> None:
        line_handles = []
        marker_handles = []
        for index, record in enumerate(records):
            ha = int(record["ha"])
            color = ha_colors.get(ha, plt.cm.viridis(index / max(len(records) - 1, 1)))
            comparison = record[axis_key]
            coordinate = np.asarray(comparison.coordinate, dtype=float)
            reference = np.asarray(comparison.reference, dtype=float)
            simulated = np.asarray(comparison.simulated, dtype=float)
            marker = "o" if axis_key == "z_profile" else "s"
            ax.plot(coordinate, reference, color=color, linewidth=1.8)
            ax.plot(
                coordinate,
                simulated,
                color=color,
                marker=marker,
                markersize=3.2,
                markerfacecolor="white",
                linewidth=1.0,
            )
            line_handles.append(Line2D([0], [0], color=color, linewidth=1.8, label=f"Analytical, Ha={ha}"))
            marker_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=color,
                    marker=marker,
                    markersize=4.0,
                    markerfacecolor="white",
                    linewidth=1.0,
                    label=f"LMX, Ha={ha}",
                )
            )
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("u / max(u)")
        ax.set_xlim(-1.02, 1.02)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.25)
        handles = line_handles + marker_handles
        ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=True)

    _draw_case(axes[0, 0], shercliff_records, axis_key="y_profile", title="Shercliff Hartmann cut", xlabel="y / a")
    _draw_case(axes[0, 1], shercliff_records, axis_key="z_profile", title="Shercliff side-layer cut", xlabel="z / a")
    _draw_case(axes[1, 0], hunt_records, axis_key="y_profile", title="Hunt Hartmann cut", xlabel="y / a")
    _draw_case(axes[1, 1], hunt_records, axis_key="z_profile", title="Hunt side-layer cut", xlabel="z / a")

    def _error_block(records: list[dict[str, object]]) -> str:
        lines = []
        for record in records:
            ha = int(record["ha"])
            y_comp = record["y_profile"]
            z_comp = record["z_profile"]
            lines.append(f"Ha={ha}: y L2={y_comp.l2_error:.2e}, z L2={z_comp.l2_error:.2e}")
        return "\n".join(lines)

    axes[0, 0].text(
        0.03,
        0.97,
        _error_block(shercliff_records),
        transform=axes[0, 0].transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.92},
    )
    axes[1, 0].text(
        0.03,
        0.97,
        _error_block(hunt_records),
        transform=axes[1, 0].transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "edgecolor": "#d1d5db", "alpha": 0.92},
    )

    png_path = out_dir / "closed_channel_validation_ladder.png"
    pdf_path = out_dir / "closed_channel_validation_ladder.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def write_hartmann_validation_ladder_figure(
    out_dir: str | Path,
    *,
    hartmann_records: list[dict[str, object]],
) -> list[Path]:
    _set_showcase_style()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12.0, 8.4))
    _add_slide_title(fig, "LMX benchmarking: Hartmann validation ladder")
    grid = fig.add_gridspec(2, 2, left=0.08, right=0.97, bottom=0.10, top=0.88, wspace=0.24, hspace=0.30)
    ax_profile = fig.add_subplot(grid[0, 0])
    ax_zoom = fig.add_subplot(grid[0, 1])
    ax_error = fig.add_subplot(grid[1, 0])
    ax_summary = fig.add_subplot(grid[1, 1])

    ha_values = [int(record["ha"]) for record in hartmann_records]
    palette = plt.cm.viridis(np.linspace(0.15, 0.85, max(len(hartmann_records), 2)))

    full_handles: list[Line2D] = []
    for idx, record in enumerate(hartmann_records):
        comparison = record["comparison"]
        color = palette[min(idx, len(palette) - 1)]
        coordinate = np.asarray(comparison.coordinate, dtype=float)
        reference = np.asarray(comparison.reference, dtype=float)
        simulated = np.asarray(comparison.simulated, dtype=float)
        positive_mask = coordinate >= 0.0
        ha = int(record["ha"])

        ax_profile.plot(coordinate, reference, color=color, linestyle="--", linewidth=1.8)
        ax_profile.plot(
            coordinate,
            simulated,
            color=color,
            marker="o",
            markersize=2.8,
            markerfacecolor="white",
            linewidth=1.2,
        )
        ax_zoom.plot(coordinate[positive_mask], reference[positive_mask], color=color, linestyle="--", linewidth=1.8)
        ax_zoom.plot(
            coordinate[positive_mask],
            simulated[positive_mask],
            color=color,
            marker="o",
            markersize=2.6,
            markerfacecolor="white",
            linewidth=1.1,
        )
        full_handles.extend(
            [
                Line2D([0], [0], color=color, linestyle="--", linewidth=1.8, label=f"Analytical, Ha={ha}"),
                Line2D(
                    [0],
                    [0],
                    color=color,
                    marker="o",
                    markersize=4.0,
                    markerfacecolor="white",
                    linewidth=1.0,
                    label=f"LMX, Ha={ha}",
                ),
            ]
        )

    ax_profile.set_title("Hartmann centerline")
    ax_profile.set_xlabel("Normalized wall-normal coordinate")
    ax_profile.set_ylabel("u / max(u)")
    ax_profile.set_xlim(-1.02, 1.02)
    ax_profile.set_ylim(-0.02, 1.05)
    ax_profile.grid(True, alpha=0.25)
    ax_profile.legend(handles=full_handles, loc="lower right", frameon=True, ncol=2)

    ax_zoom.set_title("Hartmann wall-layer zoom")
    ax_zoom.set_xlabel("Normalized wall-normal coordinate")
    ax_zoom.set_ylabel("u / max(u)")
    ax_zoom.set_xlim(0.72, 1.02)
    ax_zoom.set_ylim(-0.02, 0.42)
    ax_zoom.grid(True, alpha=0.25)

    l2_errors = [float(record["comparison"].l2_error) for record in hartmann_records]
    linf_errors = [float(record["comparison"].linf_error) for record in hartmann_records]
    ax_error.plot(ha_values, l2_errors, color="#1d4ed8", marker="o", linewidth=1.8, label="L2 error")
    ax_error.plot(ha_values, linf_errors, color="#b91c1c", marker="s", linewidth=1.8, label="L∞ error")
    ax_error.axhline(1.2e-2, color="#0f766e", linestyle="--", linewidth=1.5, label="Release L2 target")
    ax_error.set_title("Validation error by Hartmann number")
    ax_error.set_xlabel("Hartmann number")
    ax_error.set_ylabel("Normalized profile error")
    ax_error.set_yscale("log")
    ax_error.grid(True, alpha=0.25)
    ax_error.legend(loc="best", frameon=True)

    summary_lines = [
        "Validation summary",
        "",
        "Literature anchor",
        "• Classical Hartmann analytical profile",
        "",
        "Current bounded errors",
    ]
    for record in hartmann_records:
        comparison = record["comparison"]
        summary_lines.append(
            f"• Ha={int(record['ha'])}: L2={comparison.l2_error:.2e}, L∞={comparison.linf_error:.2e}"
        )
    summary_lines.extend(
        [
            "",
            "Acceptance",
            "• Release target: L2 <= 1.2e-2",
            "• Hartmann remains an explicit open quality lane",
            "  if any retained cut stays above that threshold",
        ]
    )
    ax_summary.axis("off")
    ax_summary.text(
        0.02,
        0.98,
        "\n".join(summary_lines),
        va="top",
        ha="left",
        fontsize=12,
        bbox={"facecolor": "#f8fafc", "edgecolor": "#cbd5e1", "alpha": 0.98, "boxstyle": "round,pad=0.45"},
    )

    png_path = out_dir / "hartmann_validation_ladder.png"
    pdf_path = out_dir / "hartmann_validation_ladder.pdf"
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
        solver=replace(case.solver, mode="transient", coupling_iterations=coupling_iterations, coupling_tolerance=1.0e-8),
        time_stepper=replace(
            case.time_stepper,
            dt=dt,
            t_final=t_final,
            max_steps=steps,
            potential_iterations=potential_iterations,
            potential_tolerance=1.0e-8,
            potential_relaxation=1.0,
            current_reconstruction="face_averaged",
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
