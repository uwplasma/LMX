from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lmx.fringing import (
    build_square_duct_fringing_benchmark,
    run_extruded_inductionless_slice,
    run_fringing_station_sweep,
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
    ha_peak: float = 20.0,
    ny: int = 12,
    nz: int = 12,
    nx_stations: int = 7,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    base_case, profile = build_square_duct_fringing_benchmark(
        ha_peak=ha_peak,
        ny=ny,
        nz=nz,
        nx_stations=nx_stations,
    )
    history = run_fringing_station_sweep(base_case, profile)
    extruded = run_extruded_inductionless_slice(base_case, profile)

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
    axes[1, 0].set_ylabel("y")
    fig.colorbar(contour_y, ax=axes[1, 0], shrink=0.9, label="u")

    contour_z = axes[1, 1].contourf(x_grid_z, z_grid, np.asarray(extruded.u[:, y_mid, :]).T, levels=18, cmap="viridis")
    axes[1, 1].set_title("Centerline-normal velocity u(x, ymid, z)")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].set_ylabel("z")
    fig.colorbar(contour_z, ax=axes[1, 1], shrink=0.9, label="u")

    axes[1, 2].semilogy(x, np.maximum(charge_balance, 1.0e-16), color="#7c3aed")
    axes[1, 2].set_title("Charge-balance residual")
    axes[1, 2].set_xlabel("x")
    axes[1, 2].set_ylabel("Residual")

    png_path = out_dir / "fringing_benchmark.png"
    pdf_path = out_dir / "fringing_benchmark.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "case": base_case.name,
        "solver_kind": base_case.solver.kind,
        "history": history,
        "extruded_bundle": {
            "x": np.asarray(extruded.x).tolist(),
            "y": np.asarray(extruded.y).tolist(),
            "z": np.asarray(extruded.z).tolist(),
            "field_scale": np.asarray(extruded.field_scale).tolist(),
            "u_shape": list(np.asarray(extruded.u).shape),
            "charge_balance_residual": np.asarray(extruded.charge_balance_residual).tolist(),
        },
        "plots": [png_path.name, pdf_path.name],
        "notes": (
            "This example now writes a stacked axial field bundle from stationwise "
            "fully developed solves. It is the first retained extruded_inductionless "
            "research slice, but it is still not a full 3D pressure-velocity solve."
        ),
    }
    (out_dir / "fringing_benchmark_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the LMX fringing-field benchmark scaffold.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/examples/fringing_benchmark"))
    parser.add_argument("--ha-peak", type=float, default=20.0)
    parser.add_argument("--ny", type=int, default=12)
    parser.add_argument("--nz", type=int, default=12)
    parser.add_argument("--nx-stations", type=int, default=7)
    args = parser.parse_args(argv)
    run_fringing_benchmark_demo(
        out_dir=args.output,
        ha_peak=args.ha_peak,
        ny=args.ny,
        nz=args.nz,
        nx_stations=args.nx_stations,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
