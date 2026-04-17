from __future__ import annotations

import json
from pathlib import Path

from lmx import solve_closed_channel_benchmark, write_closed_channel_profile_comparison_figure


OUTPUT_DIR = Path("artifacts/examples/straight_duct_profile_comparison")
HA = 20.0
WIDTH = 0.2
HEIGHT = 0.2
NY = 72
NZ = 72
WALL_CELLS = 10
WALL_THICKNESS = 0.02

FLUID_CONDUCTIVITY = 1.0e6
CONDUCTING_WALL_CONDUCTIVITY = 1.0e7
INSULATING_WALL_CONDUCTIVITY = 1.0e-6
DENSITY = 1.0e4
VISCOSITY = 1.0e-3


def run_straight_duct_profile_comparison(
    *,
    out_dir: Path = OUTPUT_DIR,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)

    _, shercliff_solution, shercliff_comparison = solve_closed_channel_benchmark(
        "shercliff",
        ha=HA,
        width=WIDTH,
        height=HEIGHT,
        ny=NY,
        nz=NZ,
        fluid_conductivity=FLUID_CONDUCTIVITY,
        density=DENSITY,
        viscosity=VISCOSITY,
    )
    _, hunt_solution, hunt_comparison = solve_closed_channel_benchmark(
        "hunt",
        ha=HA,
        width=WIDTH,
        height=HEIGHT,
        ny=NY,
        nz=NZ,
        wall_cells=WALL_CELLS,
        wall_thickness=WALL_THICKNESS,
        fluid_conductivity=FLUID_CONDUCTIVITY,
        density=DENSITY,
        viscosity=VISCOSITY,
        conducting_wall_conductivity=CONDUCTING_WALL_CONDUCTIVITY,
        insulating_wall_conductivity=INSULATING_WALL_CONDUCTIVITY,
    )

    comparison_outputs = write_closed_channel_profile_comparison_figure(
        out_dir,
        shercliff_solution=shercliff_solution,
        hunt_solution=hunt_solution,
        ha=HA,
    )

    summary = {
        "case": "straight_duct_profile_comparison",
        "ha": HA,
        "shercliff": {
            "y_l2_error": shercliff_comparison.y_profile.l2_error,
            "z_l2_error": shercliff_comparison.z_profile.l2_error,
        },
        "hunt": {
            "y_l2_error": hunt_comparison.y_profile.l2_error,
            "z_l2_error": hunt_comparison.z_profile.l2_error,
        },
        "outputs": [path.name for path in comparison_outputs],
    }
    (out_dir / "straight_duct_profile_comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    run_straight_duct_profile_comparison()
