from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lmx.fringing import (
    build_layered_duct_extruded_problem,
    build_pipe_ogrid_extruded_problem,
    build_square_duct_extruded_problem,
    solve_extruded_inductionless,
)


def _set_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (12, 5.0),
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
        }
    )


def run_fringing_benchmark_demo(
    *,
    out_dir: Path,
    geometry_kind: str = "rect_duct",
    ha_peak: float = 20.0,
    ny: int = 12,
    nz: int = 12,
    nx_stations: int = 7,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if geometry_kind == "rect_duct":
        problem = build_square_duct_extruded_problem(
            ha_peak=ha_peak,
            ny=ny,
            nz=nz,
            nx_stations=nx_stations,
        )
    elif geometry_kind == "layered_duct":
        problem = build_layered_duct_extruded_problem(
            ha_peak=ha_peak,
            ny=ny,
            nz=nz,
            nx_stations=nx_stations,
            wall_cells=max(1, min(4, ny // 6)),
            insulator_cells=max(1, min(3, ny // 8)),
        )
    elif geometry_kind == "pipe_ogrid":
        problem = build_pipe_ogrid_extruded_problem(
            ha_peak=ha_peak,
            nr=ny,
            ntheta=max(8, nz * 4),
            nx_stations=nx_stations,
        )
    else:
        raise ValueError(f"Unsupported geometry_kind {geometry_kind!r}")
    solution = solve_extruded_inductionless(problem)
    history = solution.station_history
    extruded = solution.bundle

    _set_style()
    fig, axes = plt.subplots(2, 3, constrained_layout=True)
    fig.suptitle("LMX extruded inductionless fringing slice", fontsize=16)
    x = np.asarray([item["x"] for item in history])
    field_scale = np.asarray([item["field_scale"] for item in history])
    mean_velocity = np.asarray([item["mean_velocity"] for item in history])
    pressure_proxy = np.asarray([item["current_scaled_pressure_proxy"] for item in history])
    charge_balance = np.asarray(extruded.charge_balance_residual)
    z_mid = extruded.u.shape[2] // 2
    y_mid = extruded.u.shape[1] // 2
    x_grid, y_grid = np.meshgrid(np.asarray(extruded.x), np.asarray(extruded.y), indexing="xy")
    x_grid_z, z_grid = np.meshgrid(np.asarray(extruded.x), np.asarray(extruded.z), indexing="xy")
    y_label = "r" if geometry_kind == "pipe_ogrid" else "y"
    z_label = r"$\theta$" if geometry_kind == "pipe_ogrid" else "z"

    axes[0, 0].plot(x, field_scale, color="#1d4ed8")
    axes[0, 0].set_title("Axial field profile")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel(r"$B/B_{max}$")

    axes[0, 1].plot(x, mean_velocity, color="#0f766e")
    axes[0, 1].set_title("Cross-sectional mean velocity")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel(r"$\bar{u}$")

    axes[0, 2].plot(x, pressure_proxy, color="#b45309")
    axes[0, 2].set_title("Pressure surrogate")
    axes[0, 2].set_xlabel("x")
    axes[0, 2].set_ylabel("Current-scaled proxy")

    contour_y = axes[1, 0].contourf(x_grid, y_grid, np.asarray(extruded.u[:, :, z_mid]).T, levels=18, cmap="viridis")
    axes[1, 0].set_title("Midplane velocity u(x, y, zmid)")
    axes[1, 0].set_xlabel("x")
    axes[1, 0].set_ylabel(y_label)
    fig.colorbar(contour_y, ax=axes[1, 0], shrink=0.9, label="u")

    contour_z = axes[1, 1].contourf(x_grid_z, z_grid, np.asarray(extruded.u[:, y_mid, :]).T, levels=18, cmap="viridis")
    axes[1, 1].set_title("Centerline-normal velocity u(x, ymid, z)")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel(z_label)
    fig.colorbar(contour_z, ax=axes[1, 1], shrink=0.9, label="u")

    axes[1, 2].semilogy(x, np.maximum(charge_balance, 1.0e-16), color="#7c3aed", label="Charge balance")
    wall_leak = np.asarray(extruded.wall_current_leakage)
    axes[1, 2].semilogy(x, np.maximum(wall_leak, 1.0e-16), color="#dc2626", linestyle="--", label="Wall leakage")
    axes[1, 2].set_title("Current conservation")
    axes[1, 2].set_xlabel("x")
    axes[1, 2].set_ylabel("Residual")
    axes[1, 2].legend(frameon=False)

    png_path = out_dir / "fringing_benchmark.png"
    pdf_path = out_dir / "fringing_benchmark.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "case": problem.case.name,
        "geometry_kind": geometry_kind,
        "solver_kind": problem.case.solver.kind,
        "history": list(history),
        "extruded_bundle": {
            "x": np.asarray(extruded.x).tolist(),
            "y": np.asarray(extruded.y).tolist(),
            "z": np.asarray(extruded.z).tolist(),
            "field_scale": np.asarray(extruded.field_scale).tolist(),
            "u_shape": list(np.asarray(extruded.u).shape),
            "charge_balance_residual": np.asarray(extruded.charge_balance_residual).tolist(),
            "axial_current": np.asarray(extruded.axial_current).tolist(),
            "wall_current_leakage": np.asarray(extruded.wall_current_leakage).tolist(),
        },
        "validation": {
            "station_count": solution.validation.station_count,
            "max_residual": solution.validation.max_residual,
            "max_charge_balance_residual": solution.validation.max_charge_balance_residual,
            "mean_velocity_span": solution.validation.mean_velocity_span,
            "volumetric_flow_rate_span": solution.validation.volumetric_flow_rate_span,
            "axial_current_span": solution.validation.axial_current_span,
            "max_wall_current_leakage": solution.validation.max_wall_current_leakage,
            "net_boundary_current_residual": solution.validation.net_boundary_current_residual,
            "field_mean_velocity_correlation": solution.validation.field_mean_velocity_correlation,
        },
        "plots": [png_path.name, pdf_path.name],
        "notes": (
            "This example now runs through the explicit extruded_inductionless "
            "slice entry point, writing both station history and stacked axial "
            "field bundles for u, v, w, p, phi, current, and Lorentz force. "
            f"{geometry_kind} now runs through the explicit extruded slice "
            "entry point. Rectangular and layered ducts both use the low-Re "
            "3D projection path here; broader production hardening remains "
            "future work."
        ),
    }
    (out_dir / "fringing_benchmark_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the LMX fringing-field benchmark scaffold.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/examples/fringing_benchmark"))
    parser.add_argument("--geometry-kind", choices=("rect_duct", "layered_duct", "pipe_ogrid"), default="rect_duct")
    parser.add_argument("--ha-peak", type=float, default=20.0)
    parser.add_argument("--ny", type=int, default=12)
    parser.add_argument("--nz", type=int, default=12)
    parser.add_argument("--nx-stations", type=int, default=7)
    args = parser.parse_args(argv)
    run_fringing_benchmark_demo(
        out_dir=args.output,
        geometry_kind=args.geometry_kind,
        ha_peak=args.ha_peak,
        ny=args.ny,
        nz=args.nz,
        nx_stations=args.nx_stations,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
