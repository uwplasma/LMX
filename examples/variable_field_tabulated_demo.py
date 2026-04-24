from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from lmx import (
    build_square_duct_extruded_problem,
    solve_extruded_inductionless,
    validate_variable_field_extruded_solution,
    write_cross_section_field_plots,
    write_extruded_overview_plots,
)
from lmx.field_models import (
    make_divergence_free_cross_section_field,
    sample_cross_section_field,
    tabulated_field_quality_metrics,
    write_tabulated_field_npz,
)
from lmx.specs import MagneticFieldSpec


OUTPUT_DIR = Path("artifacts/examples/variable_field_tabulated")
TABLE_PATH = OUTPUT_DIR / "tabulated_rect_field.npz"
WIDTH = 2.4
HEIGHT = 1.6
BASE_BZ = 12.0
PERTURBATION = 0.12
NY = 28
NZ = 28
NX_STATIONS = 13


def run_variable_field_tabulated_demo() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    field_fn = make_divergence_free_cross_section_field(
        width=WIDTH,
        height=HEIGHT,
        base_bz=BASE_BZ,
        perturbation=PERTURBATION,
    )
    y, z, field = sample_cross_section_field(field_fn, width=WIDTH, height=HEIGHT, ny=81, nz=81)
    write_tabulated_field_npz(
        TABLE_PATH,
        y=y,
        z=z,
        bx=field[..., 0],
        by=field[..., 1],
        bz=field[..., 2],
    )

    problem = build_square_duct_extruded_problem(
        ha_peak=BASE_BZ,
        width=WIDTH,
        height=HEIGHT,
        ny=NY,
        nz=NZ,
        nx_stations=NX_STATIONS,
    )
    problem = replace(
        problem,
        case=replace(
            problem.case,
            name=f"{problem.case.name}_tabulated",
            magnetic_field=MagneticFieldSpec(kind="tabulated", table_path=str(TABLE_PATH)),
        ),
    )
    solution = solve_extruded_inductionless(problem)

    field_plots = write_cross_section_field_plots(
        y=y,
        z=z,
        field=field,
        out_dir=OUTPUT_DIR,
        title="Tabulated cross-sectional magnetic field used in the extruded solve",
    )
    extruded_plots = write_extruded_overview_plots(
        solution,
        OUTPUT_DIR,
        case_title="Tabulated-field extruded inductionless duct",
    )
    validation = validate_variable_field_extruded_solution(solution)
    field_quality = tabulated_field_quality_metrics(TABLE_PATH)
    summary = {
        "case": "variable_field_tabulated",
        "table_path": TABLE_PATH.name,
        "field_plots": [path.name for path in field_plots],
        "extruded_plots": [path.name for path in extruded_plots],
        "field_quality": field_quality,
        "validation": validation,
    }
    (OUTPUT_DIR / "variable_field_tabulated_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    run_variable_field_tabulated_demo()
