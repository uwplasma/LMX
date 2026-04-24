from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp

from lmx import (
    build_magnetic_obstacle_rect_extruded_problem,
    compare_magnetic_obstacle_reference_observables,
    load_magnetic_obstacle_reference_observables,
    solve_extruded_inductionless,
    validate_magnetic_obstacle_external_readiness,
    validate_magnetic_obstacle_benchmark,
    validate_magnetic_obstacle_literature_slice,
    write_extruded_overview_plots,
    write_magnetic_obstacle_benchmark_plots,
    write_magnetic_obstacle_schematic_plots,
    write_magnetic_obstacle_reference_comparison_plots,
    write_magnetic_obstacle_reference_comparison_table,
    write_magnetic_obstacle_reference_template,
)


OUTPUT_DIR = Path("artifacts/examples/magnetic_obstacle_benchmark")
WIDTH = 2.0
HEIGHT = 2.0
BASE_BZ = 60.0
NY = 40
NZ = 40
NX_STATIONS = 25
FORCING = 2.0
MAX_STEPS = 32
COUPLING_ITERATIONS = 12
POTENTIAL_ITERATIONS = 80
EXTERNAL_REFERENCE_FILENAME = "magnetic_obstacle_reference_observables.csv"
EXTERNAL_REFERENCE_TEMPLATE_FILENAME = "magnetic_obstacle_reference_observables_template.csv"


def run_magnetic_obstacle_benchmark() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    problem = build_magnetic_obstacle_rect_extruded_problem(
        width=WIDTH,
        height=HEIGHT,
        base_bz=BASE_BZ,
        ny=NY,
        nz=NZ,
        nx_stations=NX_STATIONS,
        forcing=FORCING,
    )
    problem = replace(
        problem,
        case=replace(
            problem.case,
            time_stepper=replace(
                problem.case.time_stepper,
                max_steps=MAX_STEPS,
                potential_iterations=POTENTIAL_ITERATIONS,
            ),
            solver=replace(
                problem.case.solver,
                coupling_iterations=COUPLING_ITERATIONS,
            ),
        ),
    )
    reference_problem = replace(
        problem,
        profile=replace(problem.profile, field_scale=jnp.zeros_like(problem.profile.field_scale)),
    )

    solution = solve_extruded_inductionless(problem)
    reference_solution = solve_extruded_inductionless(reference_problem)
    validation = validate_magnetic_obstacle_benchmark(solution, reference_solution)
    literature_validation = validate_magnetic_obstacle_literature_slice(solution, reference_solution)
    external_readiness = validate_magnetic_obstacle_external_readiness(solution)
    plots = [
        *write_magnetic_obstacle_schematic_plots(
            solution,
            reference_solution,
            OUTPUT_DIR,
            case_title="Magnetic-obstacle localized-field setup and response",
        ),
        *write_extruded_overview_plots(solution, OUTPUT_DIR, case_title="Magnetic-obstacle benchmark"),
        *write_magnetic_obstacle_benchmark_plots(
            solution,
            reference_solution,
            OUTPUT_DIR,
            case_title="Magnetic-obstacle internal response",
        ),
    ]
    external_reference_comparison = _write_external_reference_artifacts(
        external_readiness["observables"],
        OUTPUT_DIR,
    )
    plots.extend(Path(path) for path in external_reference_comparison.get("plots", []))
    summary = {
        "case": "magnetic_obstacle_benchmark",
        "geometry_kind": solution.bundle.geometry_kind,
        "status": "internal_lmx_response_gate",
        "validation_note": (
            "The matched reference is the same LMX case with the localized field removed. "
            "This checks response strength and conservation, but it is not an external "
            "magnetic-obstacle validation until a literature or experimental reference "
            "case is matched on geometry, Re, Ha, interaction parameter, wall model, "
            "field profile, and observables."
        ),
        "plots": [path.name for path in plots],
        "validation": validation,
        "literature_validation": literature_validation,
        "external_readiness": external_readiness,
        "external_reference_comparison": external_reference_comparison,
    }
    (OUTPUT_DIR / "magnetic_obstacle_benchmark_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _write_external_reference_artifacts(lmx_observables: dict[str, float], out_dir: Path) -> dict[str, object]:
    reference_path = out_dir / EXTERNAL_REFERENCE_FILENAME
    if not reference_path.exists():
        template_path = write_magnetic_obstacle_reference_template(out_dir / EXTERNAL_REFERENCE_TEMPLATE_FILENAME)
        return {
            "status": "external_reference_csv_missing",
            "validation_pass": False,
            "reference_path": reference_path.name,
            "template_path": template_path.name,
            "note": (
                "Fill the template with digitized literature or experimental observables "
                "to turn this internal response gate into an external-reference comparison."
            ),
        }

    reference_observables = load_magnetic_obstacle_reference_observables(reference_path)
    comparison = compare_magnetic_obstacle_reference_observables(lmx_observables, reference_observables)
    table_path = write_magnetic_obstacle_reference_comparison_table(
        comparison,
        out_dir / "magnetic_obstacle_reference_comparison.csv",
    )
    plot_paths = write_magnetic_obstacle_reference_comparison_plots(comparison, out_dir)
    return {
        "status": "external_reference_compared",
        "validation_pass": bool(comparison["validation_pass"]),
        "reference_path": reference_path.name,
        "comparison_table": table_path.name,
        "plots": [path.name for path in plot_paths],
        "comparison": comparison,
    }


if __name__ == "__main__":
    run_magnetic_obstacle_benchmark()
