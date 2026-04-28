from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from lmx import (
    build_bent_pipe_extruded_problem,
    build_pipe_ogrid_extruded_problem,
    compare_scalar_reference_observables,
    generate_bent_pipe_mesh,
    load_scalar_reference_observables,
    solve_extruded_inductionless,
    validate_bent_pipe_low_de_baseline,
    write_bent_pipe_overview_plots,
    write_dean_vortex_reference_template,
    write_geometry_preview_plots,
    write_scalar_reference_comparison_plots,
    write_scalar_reference_comparison_table,
)


OUTPUT_DIR = Path("artifacts/examples/bent_pipe_inductionless")
HA_PEAK = 20.0
PIPE_RADIUS = 0.45
BEND_RADIUS = 3.6
BEND_ANGLE = 1.15
NR = 18
NTHETA = 40
NX_STATIONS = 15
MAX_STEPS = 12
COUPLING_ITERATIONS = 8
POTENTIAL_ITERATIONS = 40
EXTERNAL_REFERENCE_FILENAME = "dean_vortex_reference_observables.csv"
EXTERNAL_REFERENCE_TEMPLATE_FILENAME = "dean_vortex_reference_observables_template.csv"


def run_bent_pipe_inductionless_demo() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    bent_problem = build_bent_pipe_extruded_problem(
        ha_peak=HA_PEAK,
        radius=PIPE_RADIUS,
        bend_radius=BEND_RADIUS,
        bend_angle=BEND_ANGLE,
        nr=NR,
        ntheta=NTHETA,
        nx_stations=NX_STATIONS,
    )
    bent_problem = replace(
        bent_problem,
        case=replace(
            bent_problem.case,
            time_stepper=replace(
                bent_problem.case.time_stepper,
                max_steps=MAX_STEPS,
                potential_iterations=POTENTIAL_ITERATIONS,
            ),
            solver=replace(
                bent_problem.case.solver,
                coupling_iterations=COUPLING_ITERATIONS,
            ),
        ),
    )
    straight_problem = build_pipe_ogrid_extruded_problem(
        ha_peak=HA_PEAK,
        radius=PIPE_RADIUS,
        nr=NR,
        ntheta=NTHETA,
        length=bent_problem.case.geometry.length,
        nx_stations=NX_STATIONS,
        entry_center=0.25 * bent_problem.case.geometry.length,
        exit_center=0.75 * bent_problem.case.geometry.length,
        transition_width=0.08 * bent_problem.case.geometry.length,
    )
    straight_problem = replace(
        straight_problem,
        profile=bent_problem.profile,
        case=replace(
            straight_problem.case,
            time_stepper=replace(
                straight_problem.case.time_stepper,
                max_steps=MAX_STEPS,
                potential_iterations=POTENTIAL_ITERATIONS,
            ),
            solver=replace(
                straight_problem.case.solver,
                coupling_iterations=COUPLING_ITERATIONS,
            ),
        ),
    )

    bent_solution = solve_extruded_inductionless(bent_problem)
    straight_solution = solve_extruded_inductionless(straight_problem)
    validation = validate_bent_pipe_low_de_baseline(bent_solution, straight_solution)

    preview_paths = write_geometry_preview_plots(
        generate_bent_pipe_mesh(
            tube_radius=PIPE_RADIUS,
            bend_radius=BEND_RADIUS,
            bend_angle=BEND_ANGLE,
            nx=NX_STATIONS,
            nr=NR,
            ntheta=NTHETA,
        ),
        OUTPUT_DIR,
        case_title="Bent pipe geometry",
    )
    panel_paths = write_bent_pipe_overview_plots(
        bent_solution,
        OUTPUT_DIR,
        straight_solution=straight_solution,
        title="LMX bent-pipe inductionless low-De baseline",
    )
    external_reference_comparison = _write_external_reference_artifacts(
        validation,
        bent_solution,
        OUTPUT_DIR,
    )

    summary = {
        "case": "bent_pipe_inductionless",
        "geometry_kind": bent_problem.case.geometry.kind,
        "solver_kind": bent_problem.case.solver.kind,
        "plots": [path.name for path in [*preview_paths, *panel_paths]],
        "validation": validation,
        "bent_validation": {
            "station_count": bent_solution.validation.station_count,
            "max_charge_balance_residual": bent_solution.validation.max_charge_balance_residual,
            "volumetric_flow_rate_span": bent_solution.validation.volumetric_flow_rate_span,
            "max_wall_current_leakage": bent_solution.validation.max_wall_current_leakage,
            "net_boundary_current_residual": bent_solution.validation.net_boundary_current_residual,
        },
        "external_reference_comparison": external_reference_comparison,
        "notes": (
            "This example exercises the current curved-centerline inductionless "
            "baseline. It is benchmarked against the straight-pipe low-De limit, "
            "not against a full Dean-vortex solver."
        ),
    }
    (OUTPUT_DIR / "bent_pipe_inductionless_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _write_external_reference_artifacts(validation: dict[str, object], bent_solution, out_dir: Path) -> dict[str, object]:
    reference_path = out_dir / EXTERNAL_REFERENCE_FILENAME
    if not reference_path.exists():
        template_path = write_dean_vortex_reference_template(out_dir / EXTERNAL_REFERENCE_TEMPLATE_FILENAME)
        return {
            "status": "external_reference_csv_missing",
            "validation_pass": False,
            "reference_path": reference_path.name,
            "template_path": template_path.name,
            "note": (
                "Fill the template with matched curved-pipe or curved-duct "
                "Dean-vortex observables to promote this low-De baseline into "
                "an external higher-inertia validation."
            ),
        }

    pressure_span = np.max(np.asarray(bent_solution.bundle.p, dtype=float), axis=(1, 2)) - np.min(
        np.asarray(bent_solution.bundle.p, dtype=float),
        axis=(1, 2),
    )
    x = np.asarray(bent_solution.bundle.x, dtype=float)
    pressure_loss_proxy = (
        float(np.trapezoid(pressure_span, x) / max(float(x[-1] - x[0]), 1.0e-12))
        if pressure_span.size > 1 and x.size > 1
        else 0.0
    )
    lmx_observables = {
        key: float(value)
        for key, value in validation.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    lmx_observables["pressure_loss_proxy"] = pressure_loss_proxy
    reference_observables = load_scalar_reference_observables(
        reference_path,
        context="Dean-vortex reference CSV",
    )
    comparison = compare_scalar_reference_observables(lmx_observables, reference_observables)
    table_path = write_scalar_reference_comparison_table(
        comparison,
        out_dir / "dean_vortex_reference_comparison.csv",
    )
    plot_paths = write_scalar_reference_comparison_plots(
        comparison,
        out_dir,
        output_stem="dean_vortex_reference_comparison",
        title="Dean-vortex external-reference observables",
        no_data_label="No compared Dean-vortex observables",
    )
    return {
        "status": "external_reference_compared",
        "validation_pass": bool(comparison["validation_pass"]),
        "reference_path": reference_path.name,
        "comparison_table": table_path.name,
        "plots": [path.name for path in plot_paths],
        "comparison": comparison,
    }


if __name__ == "__main__":
    run_bent_pipe_inductionless_demo()
