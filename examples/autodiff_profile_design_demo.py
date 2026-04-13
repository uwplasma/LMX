from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lmx.autodiff import (
    build_hartmann_autodiff_problem,
    run_hartmann_profile_inverse_design,
    solve_differentiable_hartmann,
)


def _set_style() -> None:
    plt.style.use("default")
    plt.rcParams.update(
        {
            "figure.figsize": (12, 4.8),
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
        }
    )


def run_autodiff_profile_design_demo(
    *,
    out_dir: Path,
    target_forcing: float = 1.0,
    target_hartmann_number: float = 12.0,
    forcing_init: float = 0.25,
    hartmann_init: float = 4.0,
    learning_rate_forcing: float = 25.0,
    learning_rate_ha: float = 4.0,
    steps: int = 20,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    problem = build_hartmann_autodiff_problem(
        ny=48,
        nz=48,
        macro_iterations=8,
        potential_iterations=80,
        velocity_iterations=120,
    )
    target_u, _ = solve_differentiable_hartmann(
        problem,
        forcing=target_forcing,
        hartmann_number=target_hartmann_number,
    )
    target_profile = target_u[:, target_u.shape[1] // 2]
    design = run_hartmann_profile_inverse_design(
        problem,
        target_profile=target_profile,
        forcing_init=forcing_init,
        hartmann_init=hartmann_init,
        learning_rate_forcing=learning_rate_forcing,
        learning_rate_ha=learning_rate_ha,
        steps=steps,
    )

    _set_style()
    fig, axes = plt.subplots(1, 3, constrained_layout=True)
    fig.suptitle("LMX autodiff profile-matching inverse design", fontsize=16)

    history = design["history"]
    iterations = np.asarray([record["iteration"] for record in history])
    losses = np.asarray([record["loss"] for record in history])
    forcing_values = np.asarray([record["forcing"] for record in history])
    ha_values = np.asarray([record["hartmann_number"] for record in history])

    axes[0].semilogy(iterations, np.maximum(losses, 1.0e-16), color="#1d4ed8")
    axes[0].set_title("Profile loss")
    axes[0].set_xlabel("Iteration")
    axes[0].set_ylabel("Loss")

    axes[1].plot(iterations, forcing_values, color="#0f766e", label="forcing")
    axes[1].axhline(target_forcing, color="#0f766e", linestyle="--", alpha=0.7, label="target forcing")
    axes[1].plot(iterations, ha_values, color="#b45309", label="Hartmann number")
    axes[1].axhline(target_hartmann_number, color="#b45309", linestyle="--", alpha=0.7, label="target Ha")
    axes[1].set_title("Recovered parameters")
    axes[1].set_xlabel("Iteration")
    axes[1].legend()

    y = np.asarray(problem.mesh.y_centers)
    axes[2].plot(y, np.asarray(target_profile), color="#111827", label="target profile")
    axes[2].plot(y, np.asarray(design["recovered_profile"]), color="#7c3aed", linestyle="--", label="recovered profile")
    axes[2].set_title("Centerline profile match")
    axes[2].set_xlabel("y")
    axes[2].set_ylabel("u")
    axes[2].legend()

    png_path = out_dir / "autodiff_profile_design.png"
    pdf_path = out_dir / "autodiff_profile_design.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "target_forcing": target_forcing,
        "target_hartmann_number": target_hartmann_number,
        "forcing_init": forcing_init,
        "hartmann_init": hartmann_init,
        "learning_rate_forcing": learning_rate_forcing,
        "learning_rate_ha": learning_rate_ha,
        "steps": steps,
        "history": history,
        "recovered_forcing": design["forcing"],
        "recovered_hartmann_number": design["hartmann_number"],
        "recovered_phi_max": design["recovered_phi_max"],
        "plots": [png_path.name, pdf_path.name],
    }
    (out_dir / "autodiff_profile_design_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the LMX autodiff full-profile inverse-design demo.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/examples/autodiff_profile_design"))
    parser.add_argument("--target-forcing", type=float, default=1.0)
    parser.add_argument("--target-ha", type=float, default=12.0)
    parser.add_argument("--forcing-init", type=float, default=0.25)
    parser.add_argument("--hartmann-init", type=float, default=4.0)
    parser.add_argument("--learning-rate-forcing", type=float, default=25.0)
    parser.add_argument("--learning-rate-ha", type=float, default=4.0)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args(argv)
    run_autodiff_profile_design_demo(
        out_dir=args.output,
        target_forcing=args.target_forcing,
        target_hartmann_number=args.target_ha,
        forcing_init=args.forcing_init,
        hartmann_init=args.hartmann_init,
        learning_rate_forcing=args.learning_rate_forcing,
        learning_rate_ha=args.learning_rate_ha,
        steps=args.steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
