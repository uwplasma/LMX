from __future__ import annotations

import argparse
import json
from dataclasses import replace
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


def _build_problem(
    geometry_kind: str, *, ha_peak: float, ny: int, nz: int, nx_stations: int
):
    if geometry_kind == "rect_duct":
        return build_square_duct_extruded_problem(
            ha_peak=ha_peak, ny=ny, nz=nz, nx_stations=nx_stations
        )
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
    split_steps: int = 3,
    resume_steps: int = 3,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    problem = _build_problem(
        geometry_kind, ha_peak=ha_peak, ny=ny, nz=nz, nx_stations=nx_stations
    )
    # A nonzero initial velocity selects the same fixed-flow constraint before
    # and after the checkpoint, making this a strict restart-equivalence demo.
    problem = replace(problem, case=replace(problem.case, initial_velocity=1.0))
    base_problem = replace(
        problem,
        case=replace(
            problem.case,
            time_stepper=replace(problem.case.time_stepper, max_steps=split_steps),
        ),
    )
    resumed_problem = replace(
        problem,
        case=replace(
            problem.case,
            time_stepper=replace(problem.case.time_stepper, max_steps=resume_steps),
        ),
    )
    direct_problem = replace(
        problem,
        case=replace(
            problem.case,
            time_stepper=replace(
                problem.case.time_stepper, max_steps=split_steps + resume_steps
            ),
        ),
    )

    base_dir = out_dir / "base"
    resumed_dir = out_dir / "resumed"
    direct_dir = out_dir / "direct"
    base_solution = solve_extruded_inductionless(base_problem)
    base_outputs = write_extruded_solution_outputs(
        base_solution, base_problem.case, base_dir, write_plots=True
    )
    restart_path = write_extruded_restart_npz(
        base_solution,
        base_problem.case,
        base_dir / "restart" / f"{base_problem.case.name}_restart.npz",
    )
    restart_bundle = load_extruded_restart_bundle(restart_path)
    validate_extruded_restart_bundle(restart_bundle, case=resumed_problem.case)

    resumed_solution = solve_extruded_inductionless(
        resumed_problem, initial_bundle=restart_bundle.bundle
    )
    resumed_outputs = write_extruded_solution_outputs(
        resumed_solution, resumed_problem.case, resumed_dir, write_plots=True
    )
    resumed_restart_path = write_extruded_restart_npz(
        resumed_solution,
        resumed_problem.case,
        resumed_dir / "restart" / f"{resumed_problem.case.name}_restart.npz",
    )
    direct_solution = solve_extruded_inductionless(direct_problem)
    direct_outputs = write_extruded_solution_outputs(
        direct_solution, direct_problem.case, direct_dir, write_plots=True
    )

    x = np.asarray(direct_solution.bundle.x)
    direct_mean = np.asarray(direct_solution.bundle.mean_velocity)
    resumed_mean = np.asarray(resumed_solution.bundle.mean_velocity)
    direct_charge = np.asarray(direct_solution.bundle.charge_balance_residual)
    resumed_charge = np.asarray(resumed_solution.bundle.charge_balance_residual)
    mean_difference = np.abs(direct_mean - resumed_mean)
    charge_difference = np.abs(direct_charge - resumed_charge)
    max_state_difference = max(
        float(
            np.max(
                np.abs(
                    np.asarray(getattr(direct_solution.bundle, name))
                    - np.asarray(getattr(resumed_solution.bundle, name))
                )
            )
        )
        for name in ("u", "v", "w", "p", "phi")
    )

    _set_style()
    fig, axes = plt.subplots(1, 3, constrained_layout=True)
    fig.suptitle("LMX extruded restart / resume reproducibility", fontsize=16)

    axes[0].plot(x, direct_mean, color="#0f766e", label="Direct", linewidth=2.0)
    axes[0].plot(
        x,
        resumed_mean,
        color="#b45309",
        linestyle="--",
        label="Restarted",
        linewidth=2.0,
    )
    axes[0].set_title("Mean velocity history")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel(r"$\bar{u}$")
    axes[0].legend(frameon=False)

    axes[1].semilogy(
        x, np.maximum(direct_charge, 1.0e-16), color="#7c3aed", label="Direct"
    )
    axes[1].semilogy(
        x,
        np.maximum(resumed_charge, 1.0e-16),
        color="#dc2626",
        linestyle="--",
        label="Restarted",
    )
    axes[1].set_title("Charge-balance residual")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel(r"$\max |\nabla \cdot J|$")
    axes[1].legend(frameon=False)

    axes[2].semilogy(
        x,
        np.maximum(mean_difference, 1.0e-16),
        color="#0891b2",
        label=r"$|\Delta \bar{u}|$",
    )
    axes[2].semilogy(
        x,
        np.maximum(charge_difference, 1.0e-16),
        color="#7c3aed",
        linestyle="--",
        label=r"$|\Delta \nabla \cdot J|$",
    )
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
        "split_steps": split_steps,
        "resume_steps": resume_steps,
        "restart_input": str(restart_path),
        "restart_output": str(resumed_restart_path),
        "base_outputs": {
            key: [str(path) for path in value] for key, value in base_outputs.items()
        },
        "resumed_outputs": {
            key: [str(path) for path in value] for key, value in resumed_outputs.items()
        },
        "direct_outputs": {
            key: [str(path) for path in value] for key, value in direct_outputs.items()
        },
        "max_mean_velocity_difference": float(np.max(mean_difference)),
        "max_charge_balance_difference": float(np.max(charge_difference)),
        "max_state_difference": max_state_difference,
        "plots": [png_path.name, pdf_path.name],
    }
    (out_dir / "extruded_restart_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the LMX extruded restart / resume reproducibility demo."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/examples/extruded_restart_demo")
    )
    parser.add_argument(
        "--geometry-kind",
        choices=("rect_duct", "layered_duct", "pipe_ogrid"),
        default="layered_duct",
    )
    parser.add_argument("--ha-peak", type=float, default=10.0)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument("--nz", type=int, default=6)
    parser.add_argument("--nx-stations", type=int, default=5)
    parser.add_argument("--split-steps", type=int, default=3)
    parser.add_argument("--resume-steps", type=int, default=3)
    args = parser.parse_args(argv)
    run_extruded_restart_demo(
        out_dir=args.output,
        geometry_kind=args.geometry_kind,
        ha_peak=args.ha_peak,
        ny=args.ny,
        nz=args.nz,
        nx_stations=args.nx_stations,
        split_steps=args.split_steps,
        resume_steps=args.resume_steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
