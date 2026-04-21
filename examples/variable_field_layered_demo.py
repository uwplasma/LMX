from __future__ import annotations

import json
from pathlib import Path

from lmx import build_variable_field_layered_extruded_problem, solve_extruded_inductionless, validate_variable_field_extruded_solution, write_extruded_overview_plots


OUTPUT_DIR = Path("artifacts/examples/variable_field_layered")
WIDTH = 2.0
HEIGHT = 2.0
BASE_BZ = 12.0
PERTURBATION = 0.12
NY = 24
NZ = 24
NX_STATIONS = 15


def run_variable_field_layered_demo() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    problem = build_variable_field_layered_extruded_problem(
        width=WIDTH,
        height=HEIGHT,
        base_bz=BASE_BZ,
        perturbation=PERTURBATION,
        ny=NY,
        nz=NZ,
        nx_stations=NX_STATIONS,
    )
    solution = solve_extruded_inductionless(problem)
    plots = write_extruded_overview_plots(solution, OUTPUT_DIR, case_title="Variable-field layered duct")
    validation = validate_variable_field_extruded_solution(solution)
    summary = {
        "case": "variable_field_layered",
        "geometry_kind": solution.bundle.geometry_kind,
        "plots": [path.name for path in plots],
        "validation": validation,
    }
    (OUTPUT_DIR / "variable_field_layered_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    run_variable_field_layered_demo()
