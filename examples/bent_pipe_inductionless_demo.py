from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from lmx import (
    build_bent_pipe_extruded_problem,
    build_pipe_ogrid_extruded_problem,
    generate_bent_pipe_mesh,
    solve_extruded_inductionless,
    validate_bent_pipe_low_de_baseline,
    write_bent_pipe_overview_plots,
    write_geometry_preview_plots,
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
        "notes": (
            "This example exercises the current curved-centerline inductionless "
            "baseline. It is benchmarked against the straight-pipe low-De limit, "
            "not against a full Dean-vortex solver."
        ),
    }
    (OUTPUT_DIR / "bent_pipe_inductionless_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    run_bent_pipe_inductionless_demo()
