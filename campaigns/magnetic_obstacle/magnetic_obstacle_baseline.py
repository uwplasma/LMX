from __future__ import annotations

import json
from pathlib import Path

from lmx.fringing import (
    build_magnetic_obstacle_rect_extruded_problem,
    solve_extruded_inductionless,
    validate_magnetic_obstacle_baseline,
)
from lmx.plotting import write_extruded_overview_plots


OUTPUT_DIR = Path("artifacts/examples/magnetic_obstacle_baseline")
WIDTH = 2.0
HEIGHT = 2.0
BASE_BZ = 12.0
NY = 28
NZ = 28
NX_STATIONS = 17


def run_magnetic_obstacle_baseline() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    problem = build_magnetic_obstacle_rect_extruded_problem(
        width=WIDTH,
        height=HEIGHT,
        base_bz=BASE_BZ,
        ny=NY,
        nz=NZ,
        nx_stations=NX_STATIONS,
    )
    solution = solve_extruded_inductionless(problem)
    plots = write_extruded_overview_plots(
        solution, OUTPUT_DIR, case_title="Magnetic-obstacle baseline"
    )
    validation = validate_magnetic_obstacle_baseline(solution)
    summary = {
        "case": "magnetic_obstacle_baseline",
        "geometry_kind": solution.bundle.geometry_kind,
        "plots": [path.name for path in plots],
        "validation": validation,
    }
    (OUTPUT_DIR / "magnetic_obstacle_baseline_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


if __name__ == "__main__":
    run_magnetic_obstacle_baseline()
