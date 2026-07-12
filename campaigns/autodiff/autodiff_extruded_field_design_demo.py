from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from lmx.autodiff import (
    build_fringing_autodiff_problem,
    extruded_rect_projection_history,
    run_extruded_rect_projection_field_inverse_design,
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


def run_autodiff_extruded_field_design_demo(
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
        macro_iterations=3,
    )
    target = extruded_rect_projection_history(
        problem,
        forcing=1.0,
        peak_hartmann_number=9.0,
        entry_center=1.2,
        exit_center=4.1,
        transition_width=0.4,
    )
    station_indices = np.asarray([1, nx_stations - 2], dtype=int)
    result = run_extruded_rect_projection_field_inverse_design(
        problem,
        target_u_field=target["u_field"][station_indices],
        target_phi_field=target["phi_field"][station_indices],
        target_jy_field=target["jy_field"][station_indices],
        target_pressure_field=target["pressure_field"][station_indices],
        station_indices=station_indices,
        forcing=1.0,
        peak_hartmann_init=6.0,
        entry_center_init=0.8,
        exit_center_init=4.8,
        transition_width_init=0.7,
        steps=steps,
    )

    _set_style()
    fig, axes = plt.subplots(2, 3, constrained_layout=True)
    fig.suptitle("LMX extruded field-level autodiff inverse design", fontsize=16)

    first_station = 0
    second_station = 1
    extent = [-1.0, 1.0, -1.0, 1.0]

    def _panel(ax, target_field, recovered_field, title: str, cmap: str) -> None:
        im = ax.imshow(
            np.asarray(target_field - recovered_field),
            origin="lower",
            cmap=cmap,
            extent=extent,
        )
        ax.set_title(title)
        ax.set_xlabel("z")
        ax.set_ylabel("y")
        fig.colorbar(im, ax=ax, shrink=0.82)

    _panel(
        axes[0, 0],
        np.asarray(target["u_field"][station_indices[first_station]]),
        np.asarray(result["recovered_u_field"][first_station]),
        "u mismatch at upstream interior station",
        "viridis",
    )
    _panel(
        axes[0, 1],
        np.asarray(target["phi_field"][station_indices[first_station]]),
        np.asarray(result["recovered_phi_field"][first_station]),
        "phi mismatch at upstream interior station",
        "coolwarm",
    )
    _panel(
        axes[0, 2],
        np.asarray(target["jy_field"][station_indices[first_station]]),
        np.asarray(result["recovered_jy_field"][first_station]),
        "jy mismatch at upstream interior station",
        "magma",
    )
    _panel(
        axes[1, 0],
        np.asarray(target["u_field"][station_indices[second_station]]),
        np.asarray(result["recovered_u_field"][second_station]),
        "u mismatch at downstream interior station",
        "viridis",
    )
    _panel(
        axes[1, 1],
        np.asarray(target["pressure_field"][station_indices[second_station]]),
        np.asarray(result["recovered_pressure_field"][second_station]),
        "p mismatch at downstream interior station",
        "cividis",
    )
    axes[1, 2].semilogy(
        [item["iteration"] for item in result["history"]],
        [item["loss"] for item in result["history"]],
        color="#7c3aed",
        linewidth=2.0,
    )
    axes[1, 2].set_title("Field-objective loss history")
    axes[1, 2].set_xlabel("Iteration")
    axes[1, 2].set_ylabel("Loss")

    png_path = out_dir / "autodiff_extruded_field_design.png"
    pdf_path = out_dir / "autodiff_extruded_field_design.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "case": "extruded_rect_projection_field_design",
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
        },
        "plots": [png_path.name, pdf_path.name],
    }
    (out_dir / "autodiff_extruded_field_design_summary.json").write_text(
        json.dumps(summary, indent=2)
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the LMX field-level extruded autodiff inverse-design demo."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/examples/autodiff_extruded_field_design"),
    )
    parser.add_argument("--nx-stations", type=int, default=7)
    parser.add_argument("--ny", type=int, default=6)
    parser.add_argument("--nz", type=int, default=6)
    parser.add_argument("--steps", type=int, default=8)
    args = parser.parse_args(argv)
    run_autodiff_extruded_field_design_demo(
        out_dir=args.output,
        nx_stations=args.nx_stations,
        ny=args.ny,
        nz=args.nz,
        steps=args.steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
