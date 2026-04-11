from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from lmx.autodiff import (
    build_hartmann_autodiff_problem,
    hartmann_mean_velocity_gradients,
    hartmann_mean_velocity_finite_difference_gradients,
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


def run_autodiff_sensitivity_demo(
    *,
    out_dir: Path,
    forcing: float = 1.0,
    hartmann_values: jnp.ndarray | None = None,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    values = hartmann_values if hartmann_values is not None else jnp.linspace(2.0, 28.0, 20)
    problem = build_hartmann_autodiff_problem(ny=48, nz=48, macro_iterations=8, potential_iterations=80, velocity_iterations=120)

    records = []
    for ha in values:
        autodiff = hartmann_mean_velocity_gradients(problem, forcing=forcing, hartmann_number=ha)
        finite_diff = hartmann_mean_velocity_finite_difference_gradients(problem, forcing=forcing, hartmann_number=ha)
        records.append(
            {
                "hartmann_number": float(ha),
                "mean_velocity": float(autodiff["mean_velocity"]),
                "d_mean_velocity_d_ha": float(autodiff["d_mean_velocity_d_ha"]),
                "d_mean_velocity_d_forcing": float(autodiff["d_mean_velocity_d_forcing"]),
                "fd_d_mean_velocity_d_ha": float(finite_diff["d_mean_velocity_d_ha"]),
                "fd_d_mean_velocity_d_forcing": float(finite_diff["d_mean_velocity_d_forcing"]),
            }
        )

    _set_style()
    fig, axes = plt.subplots(1, 2, constrained_layout=True)
    fig.suptitle("LMX autodiff sensitivity validation", fontsize=16)
    ha = np.asarray([item["hartmann_number"] for item in records])
    grad_ha = np.asarray([item["d_mean_velocity_d_ha"] for item in records])
    fd_grad_ha = np.asarray([item["fd_d_mean_velocity_d_ha"] for item in records])
    grad_forcing = np.asarray([item["d_mean_velocity_d_forcing"] for item in records])
    fd_grad_forcing = np.asarray([item["fd_d_mean_velocity_d_forcing"] for item in records])

    axes[0].plot(ha, grad_ha, color="#0f766e", label="autodiff")
    axes[0].plot(ha, fd_grad_ha, color="#b45309", linestyle="--", label="finite difference")
    axes[0].set_title(r"$d\bar{u}/dHa$")
    axes[0].set_xlabel("Hartmann number")
    axes[0].set_ylabel("Sensitivity")
    axes[0].legend()

    axes[1].plot(ha, grad_forcing, color="#1d4ed8", label="autodiff")
    axes[1].plot(ha, fd_grad_forcing, color="#7c3aed", linestyle="--", label="finite difference")
    axes[1].set_title(r"$d\bar{u}/dF$")
    axes[1].set_xlabel("Hartmann number")
    axes[1].set_ylabel("Sensitivity")
    axes[1].legend()

    png_path = out_dir / "autodiff_sensitivity_validation.png"
    pdf_path = out_dir / "autodiff_sensitivity_validation.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "forcing": forcing,
        "records": records,
        "plots": [png_path.name, pdf_path.name],
    }
    (out_dir / "autodiff_sensitivity_summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate LMX autodiff sensitivities against finite differences.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/examples/autodiff_sensitivity"))
    parser.add_argument("--forcing", type=float, default=1.0)
    args = parser.parse_args(argv)
    run_autodiff_sensitivity_demo(out_dir=args.output, forcing=args.forcing)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
