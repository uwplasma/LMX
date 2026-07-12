from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lmx.autodiff import (
    build_fringing_autodiff_problem,
    fringing_mean_velocity_history,
    run_fringing_history_inverse_design,
)


def _set_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (12, 6.5),
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
        }
    )


def run_autodiff_fringing_design_demo(
    *,
    out_dir: Path,
    forcing: float = 1.0,
    peak_hartmann_target: float = 18.0,
    entry_center_target: float = 1.6,
    exit_center_target: float = 4.3,
    transition_width_target: float = 0.35,
    peak_hartmann_init: float = 10.0,
    entry_center_init: float = 1.0,
    exit_center_init: float = 5.0,
    transition_width_init: float = 0.7,
    steps: int = 16,
    nx_stations: int = 15,
    ny: int = 12,
    nz: int = 12,
    macro_iterations: int = 3,
    potential_iterations: int = 12,
    velocity_iterations: int = 16,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    problem = build_fringing_autodiff_problem(
        nx_stations=nx_stations,
        ny=ny,
        nz=nz,
        macro_iterations=macro_iterations,
        potential_iterations=potential_iterations,
        velocity_iterations=velocity_iterations,
    )
    target = fringing_mean_velocity_history(
        problem,
        forcing=forcing,
        peak_hartmann_number=peak_hartmann_target,
        entry_center=entry_center_target,
        exit_center=exit_center_target,
        transition_width=transition_width_target,
    )
    result = run_fringing_history_inverse_design(
        problem,
        target_mean_velocity=target["mean_velocity"],
        forcing=forcing,
        peak_hartmann_init=peak_hartmann_init,
        entry_center_init=entry_center_init,
        exit_center_init=exit_center_init,
        transition_width_init=transition_width_init,
        steps=steps,
    )

    _set_style()
    fig, axes = plt.subplots(2, 2, constrained_layout=True)
    fig.suptitle("LMX autodiff fringing-profile inverse design", fontsize=16)
    x = np.asarray(result["x"])
    axes[0, 0].plot(
        x,
        np.asarray(target["field_scale"]),
        label="Target",
        color="#1d4ed8",
        linewidth=2.0,
    )
    axes[0, 0].plot(
        x,
        np.asarray(result["recovered_field_scale"]),
        label="Recovered",
        color="#dc2626",
        linestyle="--",
        linewidth=2.0,
    )
    axes[0, 0].set_title("Fringing field scale")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel(r"$B/B_{peak}$")
    axes[0, 0].legend(frameon=False)

    axes[0, 1].plot(
        x,
        np.asarray(target["mean_velocity"]),
        label="Target",
        color="#0f766e",
        linewidth=2.0,
    )
    axes[0, 1].plot(
        x,
        np.asarray(result["recovered_mean_velocity"]),
        label="Recovered",
        color="#b45309",
        linestyle="--",
        linewidth=2.0,
    )
    axes[0, 1].set_title("Mean velocity history")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel(r"$\bar{u}$")
    axes[0, 1].legend(frameon=False)

    history = result["history"]
    axes[1, 0].semilogy(
        [item["iteration"] for item in history],
        [item["loss"] for item in history],
        color="#7c3aed",
    )
    axes[1, 0].set_title("Loss history")
    axes[1, 0].set_xlabel("Iteration")
    axes[1, 0].set_ylabel("Loss")

    axes[1, 1].plot(
        [item["iteration"] for item in history],
        [item["peak_hartmann_number"] for item in history],
        label="Peak Ha",
        color="#1d4ed8",
    )
    axes[1, 1].plot(
        [item["iteration"] for item in history],
        [item["entry_center"] for item in history],
        label="Entry",
        color="#0f766e",
    )
    axes[1, 1].plot(
        [item["iteration"] for item in history],
        [item["exit_center"] for item in history],
        label="Exit",
        color="#b45309",
    )
    axes[1, 1].plot(
        [item["iteration"] for item in history],
        [item["transition_width"] for item in history],
        label="Width",
        color="#7c3aed",
    )
    axes[1, 1].set_title("Recovered parameters")
    axes[1, 1].set_xlabel("Iteration")
    axes[1, 1].legend(frameon=False, ncol=2)

    png_path = out_dir / "autodiff_fringing_design.png"
    pdf_path = out_dir / "autodiff_fringing_design.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "target": {
            "peak_hartmann_number": peak_hartmann_target,
            "entry_center": entry_center_target,
            "exit_center": exit_center_target,
            "transition_width": transition_width_target,
        },
        "recovered": {
            "peak_hartmann_number": result["peak_hartmann_number"],
            "entry_center": result["entry_center"],
            "exit_center": result["exit_center"],
            "transition_width": result["transition_width"],
        },
        "history": result["history"],
        "plots": [png_path.name, pdf_path.name],
    }
    (out_dir / "autodiff_fringing_design_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the LMX fringing-profile autodiff inverse-design demo."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/examples/autodiff_fringing_design"),
    )
    parser.add_argument("--steps", type=int, default=16)
    args = parser.parse_args(argv)
    run_autodiff_fringing_design_demo(out_dir=args.output, steps=args.steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
