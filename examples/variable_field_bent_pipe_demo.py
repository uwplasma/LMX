from __future__ import annotations

import json
from pathlib import Path

from lmx import (
    build_variable_field_bent_pipe_extruded_problem,
    build_variable_field_pipe_ogrid_extruded_problem,
    solve_extruded_inductionless,
    validate_bent_pipe_low_de_baseline,
    validate_variable_field_pipe_solution,
    write_bent_pipe_overview_plots,
)


OUTPUT_DIR = Path("artifacts/examples/variable_field_bent_pipe")
RADIUS = 0.45
BEND_RADIUS = 3.6
BEND_ANGLE = 1.15
BASE_BZ = 12.0
NR = 18
NTHETA = 40
NX_STATIONS = 15


def run_variable_field_bent_pipe_demo() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bent_problem = build_variable_field_bent_pipe_extruded_problem(
        radius=RADIUS,
        bend_radius=BEND_RADIUS,
        bend_angle=BEND_ANGLE,
        base_bz=BASE_BZ,
        nr=NR,
        ntheta=NTHETA,
        nx_stations=NX_STATIONS,
    )
    straight_problem = build_variable_field_pipe_ogrid_extruded_problem(
        radius=RADIUS,
        base_bz=BASE_BZ,
        nr=NR,
        ntheta=NTHETA,
        nx_stations=NX_STATIONS,
    )
    bent_solution = solve_extruded_inductionless(bent_problem)
    straight_solution = solve_extruded_inductionless(straight_problem)
    panel_paths = write_bent_pipe_overview_plots(
        bent_solution,
        OUTPUT_DIR,
        straight_solution=straight_solution,
        title="Variable-field bent-pipe inductionless baseline",
    )
    summary = {
        "case": "variable_field_bent_pipe",
        "geometry_kind": bent_solution.bundle.geometry_kind,
        "plots": [path.name for path in panel_paths],
        "bent_validation": validate_bent_pipe_low_de_baseline(bent_solution, straight_solution),
        "field_validation": validate_variable_field_pipe_solution(bent_solution),
    }
    (OUTPUT_DIR / "variable_field_bent_pipe_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    run_variable_field_bent_pipe_demo()
