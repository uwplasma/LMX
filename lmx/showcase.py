from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from . import solvers
from .cases import make_hunt_case, make_shercliff_case
from .mesh import StructuredMesh
from .physics import build_material_fields
from .plotting import _save_figure_pair, write_transient_movies
from .reference_data import load_hunt_analytical, load_shercliff_analytical
from .solvers import _build_mesh, solve_steady
from .specs import BoundaryCondition
from .validation import closed_channel_validation, hartmann_validation
from .core import MHDState


def _set_showcase_style() -> None:
    global plt, Line2D, Rectangle
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Rectangle

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


def solve_case_snapshots(
    case,
    *,
    frame_count: int = 12,
) -> list[dict[str, object]]:
    """Advance a fully developed case and retain evenly spaced movie frames."""

    mesh = solvers._build_mesh(case)
    materials = build_material_fields(case, mesh)
    target_mean_velocity = solvers._target_mean_velocity(case)
    potential_solver = solvers._resolve_potential_solver(
        case.time_stepper.potential_solver, materials.fluid_mask
    )
    interpolate_direct_fluid_walls = not bool(materials.fluid_mask.all())
    (
        initial_u,
        initial_phi,
        initial_jy,
        initial_jz,
        initial_lorentz,
        start_time,
    ) = solvers._initial_solver_state(
        case=case,
        mesh=mesh,
        fluid_mask=materials.fluid_mask,
        interpolate_direct_fluid_walls=interpolate_direct_fluid_walls,
        initial_state=None,
    )
    dt = case.time_stepper.dt
    steps = solvers._bounded_time_step_count(
        start_time=start_time,
        dt=dt,
        t_final=case.time_stepper.t_final,
        max_steps=case.time_stepper.max_steps,
    )
    stride = max(1, steps // max(frame_count, 1))

    frames: list[dict[str, object]] = []
    initial_mean_velocity = float(
        jnp.mean(jnp.where(materials.fluid_mask, initial_u, 0.0))
    )
    frames.append(
        {
            "time": float(start_time),
            "case": case,
            "u": initial_u,
            "phi": initial_phi,
            "jy": initial_jy,
            "jz": initial_jz,
            "lorentz_x": initial_lorentz,
            "fluid_mask": materials.fluid_mask,
            "residual": 0.0,
            "potential_residual": 0.0,
            "potential_iterations": 0.0,
            "face_current_max": 0.0,
            "emf_max": 0.0,
            "face_lorentz_max": 0.0,
            "mean_velocity": initial_mean_velocity,
            "applied_forcing": float(case.forcing),
            "pressure_proxy": float(case.forcing),
            "mesh": mesh,
        }
    )
    u = initial_u
    for step_index in range(steps):
        step_time = float(start_time + (step_index + 1) * dt)
        if case.solver.kind != "fully_developed_inductionless":
            raise NotImplementedError(
                "solve_case_snapshots only supports fully_developed_inductionless cases"
            )
        linear_solver = (
            "solvax_pcg"
            if case.solver.linear_solver == "auto"
            else case.solver.linear_solver
        )
        (
            u,
            phi,
            jy,
            jz,
            lorentz,
            residual,
            potential_residual,
            potential_iteration_count,
            _linear_residual,
            _linear_iteration_count,
            face_current_max,
            emf_max,
            face_lorentz_max,
            mean_velocity,
            applied_forcing,
            _potential_initial_residual,
            _linear_initial_residual,
        ) = solvers._fully_developed_case_step(
            case=case,
            mesh=mesh,
            materials=materials,
            u_previous=u,
            step_time=step_time,
            potential_solver=potential_solver,
            target_mean_velocity=target_mean_velocity,
            linear_solver=linear_solver,
            preconditioner=case.solver.preconditioner,
            coupling_iterations=case.solver.coupling_iterations,
            coupling_tolerance=case.solver.coupling_tolerance,
        )
        if step_index % stride == 0 or step_index == steps - 1:
            frames.append(
                {
                    "time": step_time,
                    "case": case,
                    "u": u,
                    "phi": phi,
                    "jy": jy,
                    "jz": jz,
                    "lorentz_x": lorentz,
                    "fluid_mask": materials.fluid_mask,
                    "residual": float(residual),
                    "potential_residual": float(potential_residual),
                    "potential_iterations": float(potential_iteration_count),
                    "face_current_max": float(face_current_max),
                    "emf_max": float(emf_max),
                    "face_lorentz_max": float(face_lorentz_max),
                    "mean_velocity": float(mean_velocity),
                    "applied_forcing": float(applied_forcing),
                    "pressure_proxy": float(applied_forcing),
                    "mesh": mesh,
                }
            )
    return frames


def solve_closed_channel_benchmark(
    case_kind: str,
    *,
    mesh: StructuredMesh | None = None,
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
    linear_solver: str = "auto",
    drive_mode: str = "forcing",
    pressure_gradient: float | None = None,
    target_mean_velocity: float = 0.1,
    initial_profile: str = "analytic",
    reference_root: str | Path | None = None,
):
    reference = None
    reference_kwargs = {} if reference_root is None else {"reference_root": reference_root}
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
        reference = (
            load_shercliff_analytical(int(ha), **reference_kwargs)
            if case_kind == "shercliff"
            else load_hunt_analytical(int(ha), **reference_kwargs)
        )
        if pressure_gradient is None:
            pressure_gradient = reference.pressure_drop
        if pressure_gradient is None:
            raise ValueError(f"No analytical pressure-gradient reference is available for {case_kind} Ha={ha:g}")
        case = replace(case, forcing=float(pressure_gradient))
    elif drive_mode != "forcing":
        raise ValueError(f"Unsupported closed-channel benchmark drive mode {drive_mode!r}")

    if mesh is not None:
        case = replace(case, reference_phi_cell=(mesh.ny // 2, mesh.nz // 2))

    initial_state = None
    if initial_profile == "analytic":
        if reference is None:
            reference = (
                load_shercliff_analytical(int(ha), **reference_kwargs)
                if case_kind == "shercliff"
                else load_hunt_analytical(int(ha), **reference_kwargs)
            )
        solve_mesh = _build_mesh(case) if mesh is None else mesh
        y_target = np.asarray(solve_mesh.y_centers, dtype=float)
        z_target = np.asarray(solve_mesh.z_centers, dtype=float)
        ref_coord = np.asarray(reference.coordinate, dtype=float)
        ref_y = np.asarray(reference.midplane_y, dtype=float)
        ref_z = np.asarray(reference.midplane_z, dtype=float)
        y_profile = np.interp(y_target, ref_coord, ref_y)
        z_profile = np.interp(z_target, ref_coord, ref_z)
        yz_field = np.outer(y_profile, z_profile)
        yz_scale = max(float(np.max(np.abs(yz_field))), 1.0e-12)
        yz_field = yz_field / yz_scale
        if solve_mesh.fluid_mask is not None:
            yz_field = np.where(np.asarray(solve_mesh.fluid_mask, dtype=bool), yz_field, 0.0)
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
        solver=replace(
            case.solver,
            coupling_iterations=coupling_iterations,
            coupling_tolerance=1.0e-9,
            linear_solver=linear_solver,
        ),
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
    solution = solve_steady(case, mesh=mesh, initial_state=initial_state)
    comparison = closed_channel_validation(solution, case_kind, int(ha), **reference_kwargs)
    return case, solution, comparison


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

    return _save_figure_pair(fig, out_dir, "analytic_velocity_profiles")


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

    return _save_figure_pair(fig, out_dir, "closed_channel_validation_ladder")


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

    return _save_figure_pair(fig, out_dir, "hartmann_validation_ladder")


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
    frame_count: int = 42,
    fps: int = 6,
    include_3d: bool = True,
) -> list[Path]:
    """Solve every startup step and render a bounded sample of physical states."""

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
    frames = solve_case_snapshots(
        case, frame_count=min(steps, max(int(frame_count), 1))
    )
    return write_transient_movies(
        frames,
        out_dir,
        case_title=f"LMX {case_kind.capitalize()} startup",
        output_stem=f"{case_kind}_startup",
        fps=fps,
        include_3d=include_3d,
        symmetry_average_axes=("y", "z"),
    )
