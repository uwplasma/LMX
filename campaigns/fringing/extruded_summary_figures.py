from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lmx.fringing import (
    build_layered_duct_extruded_problem,
    build_square_duct_extruded_problem,
    solve_extruded_inductionless,
)


def _set_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (14.0, 8.5),
            "axes.grid": True,
            "grid.alpha": 0.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
        }
    )


def _build_problem(
    geometry_kind: str, *, ha_peak: float, ny: int, nz: int, nx_stations: int
):
    if geometry_kind == "rect_duct":
        return build_square_duct_extruded_problem(
            ha_peak=ha_peak,
            ny=ny,
            nz=nz,
            nx_stations=nx_stations,
        )
    if geometry_kind == "layered_duct":
        return build_layered_duct_extruded_problem(
            ha_peak=ha_peak,
            ny=ny,
            nz=nz,
            nx_stations=nx_stations,
            wall_cells=max(1, min(2, ny // 6)),
            insulator_cells=max(1, min(2, ny // 8)),
        )
    raise ValueError(f"Unsupported geometry_kind {geometry_kind!r}")


def _write_geometry_3d_plot(
    solution, out_dir: Path, *, title: str, stem: str
) -> list[str]:
    bundle = solution.bundle
    peak_index = int(np.argmax(np.asarray(bundle.field_scale)))
    y = np.asarray(bundle.y, dtype=float)
    z = np.asarray(bundle.z, dtype=float)
    x = np.asarray(bundle.x, dtype=float)
    y_grid, z_grid = np.meshgrid(y, z, indexing="ij")
    x_grid, y_grid_surface = np.meshgrid(x, y, indexing="xy")
    x_grid_z, z_grid_surface = np.meshgrid(x, z, indexing="xy")

    _set_style()
    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(14.0, 9.0)
    grid = fig.add_gridspec(2, 2)
    ax3d = fig.add_subplot(grid[:, 0], projection="3d")
    ax_xy = fig.add_subplot(grid[0, 1])
    ax_xz = fig.add_subplot(grid[1, 1])
    fig.suptitle(title, fontsize=16)

    u_station = np.asarray(bundle.u[peak_index], dtype=float)
    surface = ax3d.plot_surface(
        np.full_like(y_grid, x[peak_index]),
        y_grid,
        z_grid,
        facecolors=plt.cm.viridis(
            (u_station - u_station.min()) / max(np.ptp(u_station), 1.0e-12)
        ),
        rstride=1,
        cstride=1,
        shade=False,
        antialiased=True,
        alpha=0.95,
    )
    surface.set_edgecolor("none")
    ax3d.set_title(f"Peak-field cross-section at x={x[peak_index]:.2f}")
    ax3d.set_xlabel("x")
    ax3d.set_ylabel("y")
    ax3d.set_zlabel("z")
    ax3d.view_init(elev=22, azim=-58)

    u_xymid = np.asarray(bundle.u[:, :, bundle.u.shape[2] // 2], dtype=float).T
    u_xzmid = np.asarray(bundle.u[:, bundle.u.shape[1] // 2, :], dtype=float).T
    im_xy = ax_xy.pcolormesh(
        x_grid, y_grid_surface, u_xymid, shading="auto", cmap="viridis"
    )
    plt.colorbar(im_xy, ax=ax_xy, fraction=0.046, pad=0.04)
    ax_xy.set_title("Midplane velocity u(x, y, zmid)")
    ax_xy.set_xlabel("x")
    ax_xy.set_ylabel("y")

    im_xz = ax_xz.pcolormesh(
        x_grid_z, z_grid_surface, u_xzmid, shading="auto", cmap="viridis"
    )
    plt.colorbar(im_xz, ax=ax_xz, fraction=0.046, pad=0.04)
    ax_xz.set_title("Centerline-normal velocity u(x, ymid, z)")
    ax_xz.set_xlabel("x")
    ax_xz.set_ylabel("z")

    png = out_dir / f"{stem}.png"
    pdf = out_dir / f"{stem}.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return [png.name, pdf.name]


def run_extruded_summary_figures(
    *,
    out_dir: Path,
    ha_peak: float = 20.0,
    ny: int = 10,
    nz: int = 10,
    nx_stations: int = 7,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rect = solve_extruded_inductionless(
        _build_problem(
            "rect_duct", ha_peak=ha_peak, ny=ny, nz=nz, nx_stations=nx_stations
        )
    )
    layered = solve_extruded_inductionless(
        _build_problem(
            "layered_duct", ha_peak=ha_peak, ny=ny, nz=nz, nx_stations=nx_stations
        )
    )

    rect_plots = _write_geometry_3d_plot(
        rect, out_dir, title="LMX rectangular fringing slice", stem="fringing_rect_3d"
    )
    layered_plots = _write_geometry_3d_plot(
        layered, out_dir, title="LMX layered fringing slice", stem="fringing_layered_3d"
    )

    _set_style()
    fig, axes = plt.subplots(2, 2, constrained_layout=True)
    fig.suptitle("LMX fringing summary", fontsize=16)
    for solution, label, color in (
        (rect, "Rectangular", "#0f766e"),
        (layered, "Layered", "#b45309"),
    ):
        x = np.asarray(solution.bundle.x, dtype=float)
        axes[0, 0].plot(
            x, np.asarray(solution.bundle.mean_velocity), label=label, color=color
        )
        axes[0, 1].semilogy(
            x,
            np.maximum(np.asarray(solution.bundle.charge_balance_residual), 1.0e-16),
            label=label,
            color=color,
        )
        axes[1, 0].plot(
            x, np.asarray(solution.bundle.axial_current), label=label, color=color
        )
        pressure_span = np.max(np.asarray(solution.bundle.p), axis=(1, 2)) - np.min(
            np.asarray(solution.bundle.p), axis=(1, 2)
        )
        axes[1, 1].plot(x, pressure_span, label=label, color=color)
    axes[0, 0].set_title("Mean velocity history")
    axes[0, 1].set_title("Charge-balance residual")
    axes[1, 0].set_title("Axial current history")
    axes[1, 1].set_title("Pressure-span history")
    for ax in axes.ravel():
        ax.set_xlabel("x")
        ax.legend(frameon=False)
    axes[0, 1].set_ylabel(r"$\max |\nabla \cdot J|$")
    png = out_dir / "fringing_summary_panel.png"
    pdf = out_dir / "fringing_summary_panel.pdf"
    fig.savefig(png, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "case": "extruded_summary_figures",
        "rectangular": {
            "max_charge_balance_residual": rect.validation.max_charge_balance_residual,
            "net_boundary_current_residual": rect.validation.net_boundary_current_residual,
            "peak_velocity_span": rect.validation.peak_velocity_span,
        },
        "layered": {
            "max_charge_balance_residual": layered.validation.max_charge_balance_residual,
            "net_boundary_current_residual": layered.validation.net_boundary_current_residual,
            "peak_velocity_span": layered.validation.peak_velocity_span,
        },
        "plots": rect_plots + layered_plots + [png.name, pdf.name],
    }
    (out_dir / "extruded_summary_figures_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate additional fringing figures for extruded cases."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/examples/extruded_summary_figures"),
    )
    parser.add_argument("--ha-peak", type=float, default=20.0)
    parser.add_argument("--ny", type=int, default=10)
    parser.add_argument("--nz", type=int, default=10)
    parser.add_argument("--nx-stations", type=int, default=7)
    args = parser.parse_args(argv)
    run_extruded_summary_figures(
        out_dir=args.output,
        ha_peak=args.ha_peak,
        ny=args.ny,
        nz=args.nz,
        nx_stations=args.nx_stations,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
