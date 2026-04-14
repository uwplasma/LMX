from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lmx.autodiff import (
    build_fringing_autodiff_problem,
    extruded_rect_projection_iteration_history,
    run_extruded_rect_projection_trajectory_inverse_design,
)


def _set_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (13.5, 7.5),
            "axes.grid": True,
            "grid.alpha": 0.2,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
        }
    )


def run_autodiff_extruded_trajectory_demo(
    *,
    out_dir: Path,
    nx_stations: int = 7,
    ny: int = 6,
    nz: int = 6,
    steps: int = 8,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    problem = build_fringing_autodiff_problem(
        nx_stations=nx_stations,
        ny=ny,
        nz=nz,
        potential_iterations=10,
        velocity_iterations=12,
        macro_iterations=4,
    )
    station_indices = np.asarray([1, nx_stations - 2], dtype=int)
    target = extruded_rect_projection_iteration_history(
        problem,
        forcing=1.0,
        peak_hartmann_number=9.0,
        entry_center=1.2,
        exit_center=4.1,
        transition_width=0.4,
        station_indices=station_indices,
    )
    result = run_extruded_rect_projection_trajectory_inverse_design(
        problem,
        target_u_history=target["u_field"],
        target_phi_history=target["phi_field"],
        target_jy_history=target["jy_field"],
        target_pressure_history=target["pressure_field"],
        target_charge_balance_history=target["charge_balance_residual"],
        target_boundary_current_history=target["boundary_current_residual"],
        station_indices=station_indices,
        forcing=1.0,
        peak_hartmann_init=6.0,
        entry_center_init=0.8,
        exit_center_init=4.8,
        transition_width_init=0.7,
        steps=steps,
    )

    _set_style()
    fig, axes = plt.subplots(2, 2, constrained_layout=True)
    fig.suptitle("LMX extruded projection-trajectory autodiff", fontsize=16)
    iterations = np.arange(target["u_field"].shape[0], dtype=float)
    target_u = np.asarray(target["u_field"][:, 0]).reshape(target["u_field"].shape[0], -1)
    recovered_u = np.asarray(result["recovered_u_history"][:, 0]).reshape(target["u_field"].shape[0], -1)
    target_phi = np.asarray(target["phi_field"][:, -1]).reshape(target["phi_field"].shape[0], -1)
    recovered_phi = np.asarray(result["recovered_phi_history"][:, -1]).reshape(target["phi_field"].shape[0], -1)

    axes[0, 0].semilogy(
        [item["iteration"] for item in result["history"]],
        [item["loss"] for item in result["history"]],
        color="#7c3aed",
        linewidth=2.0,
    )
    axes[0, 0].set_title("Trajectory-objective loss history")
    axes[0, 0].set_xlabel("Optimization iteration")
    axes[0, 0].set_ylabel("Loss")

    axes[0, 1].plot(iterations, np.linalg.norm(target_u - recovered_u, axis=1), marker="o", color="#0f766e")
    axes[0, 1].set_title("Upstream station velocity mismatch")
    axes[0, 1].set_xlabel("Projection iteration")
    axes[0, 1].set_ylabel(r"$||u_{target} - u_{rec}||_2$")

    axes[1, 0].plot(iterations, np.linalg.norm(target_phi - recovered_phi, axis=1), marker="o", color="#b45309")
    axes[1, 0].set_title("Downstream station potential mismatch")
    axes[1, 0].set_xlabel("Projection iteration")
    axes[1, 0].set_ylabel(r"$||\phi_{target} - \phi_{rec}||_2$")

    axes[1, 1].plot(iterations, np.asarray(target["charge_balance_residual"][:, 0]), label="Target", color="#0891b2")
    axes[1, 1].plot(iterations, np.asarray(result["recovered_charge_balance_history"][:, 0]), label="Recovered", color="#dc2626", linestyle="--")
    axes[1, 1].set_title("Charge-balance trajectory")
    axes[1, 1].set_xlabel("Projection iteration")
    axes[1, 1].set_ylabel(r"$\max |\nabla \cdot J|$")
    axes[1, 1].legend(frameon=False)

    png_path = out_dir / "autodiff_extruded_trajectory.png"
    pdf_path = out_dir / "autodiff_extruded_trajectory.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "case": "extruded_rect_projection_trajectory_design",
        "station_indices": station_indices.tolist(),
        "target_parameters": {
            "peak_hartmann_number": 9.0,
            "entry_center": 1.2,
            "exit_center": 4.1,
            "transition_width": 0.4,
        },
        "recovered": {
            "peak_hartmann_number": result["peak_hartmann_number"],
            "entry_center": result["entry_center"],
            "exit_center": result["exit_center"],
            "transition_width": result["transition_width"],
            "history": result["history"],
            "model": result["model"],
        },
        "plots": [png_path.name, pdf_path.name],
    }
    (out_dir / "autodiff_extruded_trajectory_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the LMX projection-trajectory extruded autodiff demo.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/examples/autodiff_extruded_trajectory"))
    parser.add_argument("--nx-stations", type=int, default=7)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument("--nz", type=int, default=6)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args(argv)
    run_autodiff_extruded_trajectory_demo(
        out_dir=args.output,
        nx_stations=args.nx_stations,
        ny=args.ny,
        nz=args.nz,
        steps=args.steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
