from __future__ import annotations

import json
from pathlib import Path

from lmx.fringing import (
    build_variable_field_duct_extruded_problem,
    solve_extruded_inductionless,
    validate_variable_field_extruded_solution,
)
from lmx.plotting import (
    write_cross_section_field_plots,
    write_extruded_overview_plots,
)
from lmx.field_models import sample_cross_section_field


OUTPUT_DIR = Path("artifacts/examples/variable_field_extruded")
WIDTH = 2.4
HEIGHT = 1.6
BASE_BZ = 12.0
PERTURBATION = 0.12
NY = 32
NZ = 32
NX_STATIONS = 15


def run_variable_field_extruded_demo() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    problem = build_variable_field_duct_extruded_problem(
        width=WIDTH,
        height=HEIGHT,
        base_bz=BASE_BZ,
        perturbation=PERTURBATION,
        ny=NY,
        nz=NZ,
        nx_stations=NX_STATIONS,
    )
    solution = solve_extruded_inductionless(problem)
    field_fn = problem.case.magnetic_field.fn
    assert field_fn is not None
    y, z, field = sample_cross_section_field(
        field_fn, width=WIDTH, height=HEIGHT, ny=81, nz=81
    )
    field_plots = write_cross_section_field_plots(
        y=y,
        z=z,
        field=field,
        out_dir=OUTPUT_DIR,
        title="Analytic cross-sectional magnetic field used in the extruded solve",
    )
    extruded_plots = write_extruded_overview_plots(
        solution,
        OUTPUT_DIR,
        case_title="Variable-field extruded inductionless duct",
    )
    validation = validate_variable_field_extruded_solution(solution)
    summary = {
        "case": "variable_field_extruded",
        "geometry_kind": solution.bundle.geometry_kind,
        "field_plots": [path.name for path in field_plots],
        "extruded_plots": [path.name for path in extruded_plots],
        "validation": validation,
    }
    (OUTPUT_DIR / "variable_field_extruded_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


if __name__ == "__main__":
    run_variable_field_extruded_demo()
