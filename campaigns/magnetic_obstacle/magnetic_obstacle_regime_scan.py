from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp

from lmx.fringing import (
    build_magnetic_obstacle_rect_extruded_problem,
    solve_extruded_inductionless,
    validate_magnetic_obstacle_benchmark,
)
from lmx.plotting import write_magnetic_obstacle_regime_plots


OUTPUT_DIR = Path("artifacts/examples/magnetic_obstacle_regime_scan")
WIDTH = 2.0
HEIGHT = 2.0
BASE_BZ_VALUES = [20.0, 40.0, 60.0]
FORCING_VALUES = [0.5, 1.0, 2.0]
NY = 18
NZ = 18
NX_STATIONS = 13
MAX_STEPS = 18
COUPLING_ITERATIONS = 8
POTENTIAL_ITERATIONS = 40


def run_magnetic_obstacle_regime_scan() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, float]] = []
    for base_bz in BASE_BZ_VALUES:
        for forcing in FORCING_VALUES:
            problem = build_magnetic_obstacle_rect_extruded_problem(
                width=WIDTH,
                height=HEIGHT,
                base_bz=base_bz,
                ny=NY,
                nz=NZ,
                nx_stations=NX_STATIONS,
                forcing=forcing,
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
                profile=replace(
                    problem.profile,
                    field_scale=jnp.zeros_like(problem.profile.field_scale),
                ),
            )
            solution = solve_extruded_inductionless(problem)
            reference_solution = solve_extruded_inductionless(reference_problem)
            validation = validate_magnetic_obstacle_benchmark(
                solution, reference_solution
            )
            records.append(
                {
                    "base_bz": float(base_bz),
                    "forcing": float(forcing),
                    "peak_velocity_deficit_ratio": float(
                        validation["peak_velocity_deficit_ratio"]
                    ),
                    "peak_pressure_excess": float(validation["peak_pressure_excess"]),
                    "pressure_excess_proxy": float(validation["pressure_excess_proxy"]),
                    "current_proxy_peak": float(validation["current_proxy_peak"]),
                    "y_l2_distortion": float(validation["y_l2_distortion"]),
                    "z_l2_distortion": float(validation["z_l2_distortion"]),
                    "benchmark_pass": bool(validation["benchmark_pass"]),
                }
            )

    plots = write_magnetic_obstacle_regime_plots(
        records, OUTPUT_DIR, case_title="Magnetic-obstacle regime scan"
    )
    summary = {
        "case": "magnetic_obstacle_regime_scan",
        "plots": [path.name for path in plots],
        "records": records,
    }
    (OUTPUT_DIR / "magnetic_obstacle_regime_scan_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


if __name__ == "__main__":
    run_magnetic_obstacle_regime_scan()
