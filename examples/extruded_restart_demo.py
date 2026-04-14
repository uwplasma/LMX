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
from lmx.io import (
    load_extruded_restart_bundle,
    validate_extruded_restart_bundle,
    write_extruded_restart_npz,
    write_extruded_solution_outputs,
)


def _set_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (12.5, 6.5),
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
        }
    )


def _build_problem(geometry_kind: str, *, ha_peak: float, ny: int, nz: int, nx_stations: int):
    if geometry_kind == "rect_duct":
        return build_square_duct_extruded_problem(ha_peak=ha_peak, ny=ny, nz=nz, nx_stations=nx_stations)
    if geometry_kind == "layered_duct":
        return build_layered_duct_extruded_problem(
            ha_peak=ha_peak,
            ny=ny,
            nz=nz,
            nx_stations=nx_stations,
            wall_cells=1,
            insulator_cells=1,
        )
    if geometry_kind == "pipe_ogrid":
        return build_pipe_ogrid_extruded_problem(
            ha_peak=ha_peak,
            nr=ny,
            ntheta=max(12, nz * 4),
            nx_stations=nx_stations,
        )
    raise ValueError(f"Unsupported geometry_kind {geometry_kind!r}")


def run_extruded_restart_demo(
    *,
    out_dir: Path,
    geometry_kind: str = "layered_duct",
    ha_peak: float = 10.0,
    ny: int = 6,
    nz: int = 6,
    nx_stations: int = 5,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    problem = _build_problem(geometry_kind, ha_peak=ha_peak, ny=ny, nz=nz, nx_stations=nx_stations)

    base_dir = out_dir / "base"
    resumed_dir = out_dir / "resumed"
    base_solution = solve_extruded_inductionless(problem)
    base_outputs = write_extruded_solution_outputs(base_solution, problem.case, base_dir, write_plots=True)
    restart_path = write_extruded_restart_npz(base_solution, problem.case, base_dir / "restart" / f"{problem.case.name}_restart.npz")
    restart_bundle = load_extruded_restart_bundle(restart_path)
    validate_extruded_restart_bundle(restart_bundle, case=problem.case)

    resumed_solution = solve_extruded_inductionless(problem, initial_bundle=restart_bundle.bundle)
    resumed_outputs = write_extruded_solution_outputs(resumed_solution, problem.case, resumed_dir, write_plots=True)
    resumed_restart_path = write_extruded_restart_npz(
        resumed_solution,
        problem.case,
        resumed_dir / "restart" / f"{problem.case.name}_restart.npz",
    )

    x = np.asarray(base_solution.bundle.x)
    base_mean = np.asarray(base_solution.bundle.mean_velocity)
    resumed_mean = np.asarray(resumed_solution.bundle.mean_velocity)
    base_charge = np.asarray(base_solution.bundle.charge_balance_residual)
    resumed_charge = np.asarray(resumed_solution.bundle.charge_balance_residual)
    mean_difference = np.abs(base_mean - resumed_mean)
    charge_difference = np.abs(base_charge - resumed_charge)

    _set_style()
    fig, axes = plt.subplots(1, 3, constrained_layout=True)
    fig.suptitle("LMX extruded restart / resume reproducibility", fontsize=16)

    axes[0].plot(x, base_mean, color="#0f766e", label="Base", linewidth=2.0)
    axes[0].plot(x, resumed_mean, color="#b45309", linestyle="--", label="Restarted", linewidth=2.0)
    axes[0].set_title("Mean velocity history")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel(r"$\bar{u}$")
    axes[0].legend(frameon=False)

    axes[1].semilogy(x, np.maximum(base_charge, 1.0e-16), color="#7c3aed", label="Base")
    axes[1].semilogy(x, np.maximum(resumed_charge, 1.0e-16), color="#dc2626", linestyle="--", label="Restarted")
    axes[1].set_title("Charge-balance residual")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel(r"$\max |\nabla \cdot J|$")
    axes[1].legend(frameon=False)

    axes[2].semilogy(x, np.maximum(mean_difference, 1.0e-16), color="#0891b2", label=r"$|\Delta \bar{u}|$")
    axes[2].semilogy(x, np.maximum(charge_difference, 1.0e-16), color="#7c3aed", linestyle="--", label=r"$|\Delta \nabla \cdot J|$")
    axes[2].set_title("Restart-to-direct difference")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("Absolute difference")
    axes[2].legend(frameon=False)

    png_path = out_dir / "extruded_restart_demo.png"
    pdf_path = out_dir / "extruded_restart_demo.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "geometry_kind": geometry_kind,
        "case": problem.case.name,
        "restart_input": str(restart_path),
        "restart_output": str(resumed_restart_path),
        "base_outputs": {key: [str(path) for path in value] for key, value in base_outputs.items()},
        "resumed_outputs": {key: [str(path) for path in value] for key, value in resumed_outputs.items()},
        "max_mean_velocity_difference": float(np.max(mean_difference)),
        "max_charge_balance_difference": float(np.max(charge_difference)),
        "plots": [png_path.name, pdf_path.name],
    }
    (out_dir / "extruded_restart_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the LMX extruded restart / resume reproducibility demo.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/examples/extruded_restart_demo"))
    parser.add_argument("--geometry-kind", choices=("rect_duct", "layered_duct", "pipe_ogrid"), default="layered_duct")
    parser.add_argument("--ha-peak", type=float, default=10.0)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument("--nz", type=int, default=6)
    parser.add_argument("--nx-stations", type=int, default=5)
    args = parser.parse_args(argv)
    run_extruded_restart_demo(
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
