from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from lmx import (
    build_wham_mirror_pipe_extruded_problem,
    sample_tabulated_field_volume,
    solve_extruded_inductionless,
    validate_wham_mirror_pipe_baseline,
    write_cross_section_field_plots,
    write_extruded_overview_plots,
    write_wham_mirror_field_npz,
)


OUTPUT_DIR = Path("artifacts/examples/wham_mirror_pipe")
PIPE_RADIUS = 0.22
PIPE_LENGTH = 1.40
NR = 18
NTHETA = 48
NX_STATIONS = 25
FIELD_NX = 41
FIELD_NY = 25
FIELD_NZ = 25
COIL_SEPARATION = 1.96
RADIAL_LOOPS = 24
AXIAL_LOOPS = 6
CURRENT_SCALE = (2000.0 * 17.0 / 17.51) * 800.0
FORCING = 6.0
MAX_STEPS = 96
COUPLING_ITERATIONS = 12
POTENTIAL_ITERATIONS = 64


def run_wham_mirror_pipe_demo() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    x_field = np.linspace(-0.5 * PIPE_LENGTH, 0.5 * PIPE_LENGTH, FIELD_NX)
    y_field = np.linspace(-PIPE_RADIUS, PIPE_RADIUS, FIELD_NY)
    z_field = np.linspace(-PIPE_RADIUS, PIPE_RADIUS, FIELD_NZ)
    table_path = write_wham_mirror_field_npz(
        OUTPUT_DIR / "wham_mirror_field.npz",
        x=x_field,
        y=y_field,
        z=z_field,
        coil_separation=COIL_SEPARATION,
        current_scale=CURRENT_SCALE,
        radial_loops=RADIAL_LOOPS,
        axial_loops=AXIAL_LOOPS,
    )

    problem = build_wham_mirror_pipe_extruded_problem(
        table_path=str(table_path),
        radius=PIPE_RADIUS,
        nr=NR,
        ntheta=NTHETA,
        length=PIPE_LENGTH,
        nx_stations=NX_STATIONS,
        forcing=FORCING,
    )
    centered_x = jnp.linspace(-0.5 * PIPE_LENGTH, 0.5 * PIPE_LENGTH, NX_STATIONS)
    centerline_field = np.asarray(
        sample_tabulated_field_volume(
            table_path,
            x=np.asarray(centered_x, dtype=float),
            y=np.zeros(NX_STATIONS, dtype=float),
            z=np.zeros(NX_STATIONS, dtype=float),
        ),
        dtype=float,
    )
    centerline_bmag = np.linalg.norm(centerline_field, axis=-1)
    centerline_scale = centerline_bmag / max(float(np.max(centerline_bmag)), 1.0e-12)
    problem = replace(
        problem,
        profile=replace(problem.profile, x=centered_x, field_scale=jnp.asarray(centerline_scale, dtype=float)),
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
    solution = solve_extruded_inductionless(problem)
    validation = validate_wham_mirror_pipe_baseline(solution)

    yy, zz = np.meshgrid(y_field, z_field, indexing="ij")
    mid_field = np.asarray(
        sample_tabulated_field_volume(
            table_path,
            x=np.zeros_like(yy),
            y=yy,
            z=zz,
        ),
        dtype=float,
    )
    field_plots = write_cross_section_field_plots(
        y=y_field,
        z=z_field,
        field=mid_field,
        out_dir=OUTPUT_DIR,
        title="WHAM mirror field at pipe midplane",
    )
    overview_plots = write_extruded_overview_plots(
        solution,
        OUTPUT_DIR,
        case_title="WHAM mirror pipe baseline",
    )
    summary = {
        "case": "wham_mirror_pipe",
        "field_table": table_path.name,
        "plots": [path.name for path in [*field_plots, *overview_plots]],
        "validation": validation,
    }
    (OUTPUT_DIR / "wham_mirror_pipe_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    run_wham_mirror_pipe_demo()
