from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lmx.autodiff import run_extruded_target_inverse_design
from lmx.fringing import (
    build_pipe_ogrid_extruded_problem,
    build_square_duct_extruded_problem,
    solve_extruded_inductionless,
)


def _set_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (12.5, 7.0),
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
        }
    )


def run_autodiff_extruded_target_demo(
    *,
    out_dir: Path,
    geometry_kind: str = "rect_duct",
    ha_peak: float = 8.0,
    ny: int = 6,
    nz: int = 6,
    nx_stations: int = 7,
    steps: int = 10,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    if geometry_kind == "pipe_ogrid":
        problem = build_pipe_ogrid_extruded_problem(
            ha_peak=ha_peak,
            nr=ny,
            ntheta=max(8, nz * 4),
            nx_stations=nx_stations,
        )
    else:
        problem = build_square_duct_extruded_problem(
            ha_peak=ha_peak,
            ny=ny,
            nz=nz,
            nx_stations=nx_stations,
        )
    solution = solve_extruded_inductionless(problem)
    inverse = run_extruded_target_inverse_design(
        solution,
        ny=max(6, ny),
        nz=max(6, nz),
        steps=steps,
    )

    target = inverse["target"]
    recovered = inverse["recovered"]
    x = np.asarray(target["x"])

    _set_style()
    fig, axes = plt.subplots(2, 2, constrained_layout=True)
    fig.suptitle("LMX extruded-target autodiff inverse design", fontsize=16)

    axes[0, 0].plot(x, np.asarray(target["mean_velocity"]), label="Target", color="#0f766e", linewidth=2.0)
    axes[0, 0].plot(x, np.asarray(recovered["recovered_mean_velocity"]), label="Recovered", color="#b45309", linestyle="--", linewidth=2.0)
    axes[0, 0].set_title("Axial mean velocity")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel(r"$\bar{u}$")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(x, np.asarray(target["current_proxy"]), label="Target", color="#1d4ed8", linewidth=2.0)
    axes[0, 1].plot(x, np.asarray(recovered["recovered_current_proxy"]), label="Recovered", color="#dc2626", linestyle="--", linewidth=2.0)
    axes[0, 1].set_title("Current-weighted pressure proxy")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("Proxy")
    axes[0, 1].legend(frameon=False)

    history = recovered["history"]
    axes[1, 0].semilogy([item["iteration"] for item in history], [item["loss"] for item in history], color="#7c3aed")
    axes[1, 0].set_title("Inverse-design loss history")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Loss")

    axes[1, 1].semilogy(x, np.maximum(np.asarray(target["charge_balance_residual"]), 1.0e-16), label="Charge balance", color="#7c3aed")
    axes[1, 1].semilogy(x, np.maximum(np.asarray(target["wall_current_leakage"]), 1.0e-16), label="Wall leakage", color="#dc2626", linestyle="--")
    axes[1, 1].plot(x, np.asarray(target["axial_current"]), label="Axial current", color="#0891b2")
    axes[1, 1].set_title("Extruded conservation observables")
    axes[1, 1].set_xlabel("x")
    axes[1, 1].legend(frameon=False)

    png_path = out_dir / "autodiff_extruded_target.png"
    pdf_path = out_dir / "autodiff_extruded_target.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "geometry_kind": geometry_kind,
        "case": problem.case.name,
        "solver_kind": problem.case.solver.kind,
        "forcing": inverse["forcing"],
        "target": {
            "x": np.asarray(target["x"]).tolist(),
            "field_scale": np.asarray(target["field_scale"]).tolist(),
            "mean_velocity": np.asarray(target["mean_velocity"]).tolist(),
            "current_proxy": np.asarray(target["current_proxy"]).tolist(),
            "charge_balance_residual": np.asarray(target["charge_balance_residual"]).tolist(),
            "wall_current_leakage": np.asarray(target["wall_current_leakage"]).tolist(),
            "axial_current": np.asarray(target["axial_current"]).tolist(),
        },
        "recovered": {
            "model": recovered.get("model", "unknown"),
            "peak_hartmann_number": recovered["peak_hartmann_number"],
            "entry_center": recovered["entry_center"],
            "exit_center": recovered["exit_center"],
            "transition_width": recovered["transition_width"],
            "history": recovered["history"],
        },
        "plots": [png_path.name, pdf_path.name],
    }
    (out_dir / "autodiff_extruded_target_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the LMX extruded-target autodiff inverse-design demo.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/examples/autodiff_extruded_target"))
    parser.add_argument("--geometry-kind", choices=("rect_duct", "pipe_ogrid"), default="rect_duct")
    parser.add_argument("--ha-peak", type=float, default=8.0)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument("--nz", type=int, default=6)
    parser.add_argument("--nx-stations", type=int, default=7)
    parser.add_argument("--steps", type=int, default=10)
    args = parser.parse_args(argv)
    run_autodiff_extruded_target_demo(
        out_dir=args.output,
        geometry_kind=args.geometry_kind,
        ha_peak=args.ha_peak,
        ny=args.ny,
        nz=args.nz,
        nx_stations=args.nx_stations,
        steps=args.steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
