"""Plots for LMX solutions and FreeMHD comparisons."""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from .specs import Solution
from .validation import extract_midplane_profile


def _load_matplotlib() -> None:
    global plt, colors
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import colors


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

    from matplotlib import pyplot

    save_options = {"bbox_inches": "tight"} if tight else {}
    if dpi is not None:
        save_options["dpi"] = dpi
    paths = [out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"]
    for path in paths:
        fig.savefig(path, **save_options)
    pyplot.close(fig)
    return paths


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
    _plot_field(
        axes[0, 1],
        solution,
        solution.state.phi,
        title="Electric potential φ",
        cmap="PuOr_r",
    )
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
        axes[0].plot(
            time_history,
            solution.diagnostics.u_max_history,
            color="#1d4ed8",
            label="max |u|",
        )
        if solution.diagnostics.current_max_history.size:
            axes[0].plot(
                time_history,
                solution.diagnostics.current_max_history,
                color="#b91c1c",
                label="max |J|",
            )
        if solution.diagnostics.lorentz_max_history.size:
            axes[0].plot(
                time_history,
                solution.diagnostics.lorentz_max_history,
                color="#6d28d9",
                label="max |J×B|",
            )
        axes[0].set_title("Trace magnitudes")
        axes[0].set_xlabel("time")
        axes[0].set_ylabel("magnitude")
        axes[0].legend(loc="upper left", bbox_to_anchor=(0.02, 0.98))

        axes[1].plot(
            time_history,
            solution.diagnostics.residual_history,
            color="#0f766e",
            label="velocity residual",
        )
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


def _centers_to_edges(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=float)
    if data.size <= 1:
        center = float(data[0]) if data.size else 0.0
        return np.asarray([center - 0.5, center + 0.5], dtype=float)
    midpoints = 0.5 * (data[1:] + data[:-1])
    first = data[0] - 0.5 * (data[1] - data[0])
    last = data[-1] + 0.5 * (data[-1] - data[-2])
    return np.concatenate([[first], midpoints, [last]])


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
    axes[0, 1].plot(
        x,
        np.maximum(np.abs(axial_current), 1.0e-16),
        color="#111827",
        alpha=0.6,
        label="|Axial current|",
    )
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
                ax.plot(
                    coordinate,
                    reference,
                    color=color,
                    linewidth=2.0,
                    label=f"{label_prefix} FreeMHD",
                )
                ax.plot(
                    coordinate,
                    simulated,
                    color=color,
                    linestyle="--",
                    linewidth=1.8,
                    label=f"{label_prefix} LMX",
                )
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
                bbox={
                    "boxstyle": "round,pad=0.22",
                    "facecolor": "white",
                    "edgecolor": "#cbd5e1",
                    "alpha": 0.92,
                },
            )
            ax.set_xlim(-1.0, 1.0)
            ax.set_xlabel("Normalized coordinate")
            if column_index == 0:
                ax.set_ylabel(case_name)
            if row_index == 0 and column_index == 3:
                ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.30), ncol=2, fontsize=9.5)

    return _save_figure_pair(fig, out_dir, output_stem)
