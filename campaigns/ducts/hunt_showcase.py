from __future__ import annotations

import json
from pathlib import Path

from lmx.showcase import (
    solve_closed_channel_benchmark,
    write_annotated_layer_figure,
    write_boundary_layer_figure,
    write_closed_channel_startup_movies,
    write_velocity_profile_volume_figure,
)


OUTPUT_DIR = Path("artifacts/examples/hunt_showcase")
HA = 20.0
WIDTH = 0.2
HEIGHT = 0.2
NY = 48
NZ = 48
WALL_CELLS = 10
WALL_THICKNESS = 0.02

FLUID_CONDUCTIVITY = 1.0e6
CONDUCTING_WALL_CONDUCTIVITY = 1.0e7
INSULATING_WALL_CONDUCTIVITY = 1.0e-6
DENSITY = 1.0e4
VISCOSITY = 1.0e-3

COUPLING_ITERATIONS = 16
POTENTIAL_ITERATIONS = 160
MAX_STEPS = 160

STARTUP_NY = 49
STARTUP_NZ = 49
STARTUP_DT = 1.0e-5
STARTUP_T_FINAL = 2.0e-3
STARTUP_COUPLING_ITERATIONS = 6
STARTUP_POTENTIAL_ITERATIONS = 48


def run_hunt_showcase(
    *,
    out_dir: Path = OUTPUT_DIR,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)

    case, solution, comparison = solve_closed_channel_benchmark(
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
        coupling_iterations=COUPLING_ITERATIONS,
        potential_iterations=POTENTIAL_ITERATIONS,
        max_steps=MAX_STEPS,
    )

    boundary_outputs = write_boundary_layer_figure(
        solution, out_dir, title="Boundary Layer Development - Hunt"
    )
    volume_outputs = write_velocity_profile_volume_figure(
        solution,
        out_dir,
        title="Liquid Metal Velocity Profile - Hunt",
        case_kind="hunt",
    )
    annotated_outputs = write_annotated_layer_figure(
        solution,
        out_dir,
        title="U Magnitude - Hunt flow",
        case_kind="hunt",
        ha=HA,
        half_width=0.5 * WIDTH,
    )
    movie_outputs = write_closed_channel_startup_movies(
        "hunt",
        out_dir,
        ha=HA,
        width=WIDTH,
        height=HEIGHT,
        ny=STARTUP_NY,
        nz=STARTUP_NZ,
        wall_cells=6,
        fluid_conductivity=FLUID_CONDUCTIVITY,
        density=DENSITY,
        viscosity=VISCOSITY,
        conducting_wall_conductivity=CONDUCTING_WALL_CONDUCTIVITY,
        insulating_wall_conductivity=INSULATING_WALL_CONDUCTIVITY,
        dt=STARTUP_DT,
        t_final=STARTUP_T_FINAL,
        coupling_iterations=STARTUP_COUPLING_ITERATIONS,
        potential_iterations=STARTUP_POTENTIAL_ITERATIONS,
    )

    summary = {
        "case": case.name,
        "comparison": {
            "y_l2_error": comparison.y_profile.l2_error,
            "z_l2_error": comparison.z_profile.l2_error,
        },
        "boundary_outputs": [path.name for path in boundary_outputs],
        "volume_outputs": [path.name for path in volume_outputs],
        "annotated_outputs": [path.name for path in annotated_outputs],
        "movie_outputs": [path.name for path in movie_outputs],
    }
    (out_dir / "hunt_showcase_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


if __name__ == "__main__":
    run_hunt_showcase()
