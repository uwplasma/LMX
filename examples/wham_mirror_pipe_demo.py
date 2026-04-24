from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from lmx import (
    build_fringing_autodiff_problem,
    build_wham_mirror_pipe_extruded_problem,
    sample_tabulated_field_volume,
    tabulated_field_quality_metrics,
    solve_extruded_inductionless,
    validate_wham_mirror_pipe_baseline,
    wham_mirror_pressure_drop_sensitivity,
    write_cross_section_field_plots,
    write_extruded_overview_plots,
    write_wham_mirror_overview_plots,
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
COIL_INNER_RADIUS = 0.5 * 86.0e-3
COIL_OUTER_RADIUS = 0.5 * 730.0e-3
FORCING = 6.0
MAX_STEPS = 96
COUPLING_ITERATIONS = 12
POTENTIAL_ITERATIONS = 64
AUTODIFF_NX_STATIONS = 25
AUTODIFF_NY = 12
AUTODIFF_NZ = 12
AUTODIFF_FORCING = 1.0
AUTODIFF_PEAK_HARTMANN_NUMBER = 20.0
AUTODIFF_SEPARATION_SWEEP = np.linspace(1.50, 2.30, 9)
AUTODIFF_RADIAL_LOOPS = 16
AUTODIFF_AXIAL_LOOPS = 4


def run_wham_mirror_pipe_demo() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    x_field = np.linspace(0.0, PIPE_LENGTH, FIELD_NX)
    y_field = np.linspace(-PIPE_RADIUS, PIPE_RADIUS, FIELD_NY)
    z_field = np.linspace(-PIPE_RADIUS, PIPE_RADIUS, FIELD_NZ)
    coil_frame_x_offset = -0.5 * PIPE_LENGTH
    table_path = write_wham_mirror_field_npz(
        OUTPUT_DIR / "wham_mirror_field.npz",
        x=x_field,
        y=y_field,
        z=z_field,
        coil_separation=COIL_SEPARATION,
        current_scale=CURRENT_SCALE,
        inner_radius=COIL_INNER_RADIUS,
        outer_radius=COIL_OUTER_RADIUS,
        radial_loops=RADIAL_LOOPS,
        axial_loops=AXIAL_LOOPS,
        x_offset=coil_frame_x_offset,
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
    solver_x = jnp.linspace(0.0, PIPE_LENGTH, NX_STATIONS)
    centerline_field = np.asarray(
        sample_tabulated_field_volume(
            table_path,
            x=np.asarray(solver_x, dtype=float),
            y=np.zeros(NX_STATIONS, dtype=float),
            z=np.zeros(NX_STATIONS, dtype=float),
        ),
        dtype=float,
    )
    centerline_bmag = np.linalg.norm(centerline_field, axis=-1)
    centerline_scale = centerline_bmag / max(float(np.max(centerline_bmag)), 1.0e-12)
    problem = replace(
        problem,
        profile=replace(problem.profile, x=solver_x, field_scale=jnp.asarray(centerline_scale, dtype=float)),
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
    field_quality = tabulated_field_quality_metrics(table_path)

    autodiff_problem = build_fringing_autodiff_problem(
        nx_stations=AUTODIFF_NX_STATIONS,
        length=PIPE_LENGTH,
        ny=AUTODIFF_NY,
        nz=AUTODIFF_NZ,
        macro_iterations=3,
        potential_iterations=12,
        velocity_iterations=16,
    )
    autodiff_reference = wham_mirror_pressure_drop_sensitivity(
        autodiff_problem,
        forcing=AUTODIFF_FORCING,
        peak_hartmann_number=AUTODIFF_PEAK_HARTMANN_NUMBER,
        coil_separation=COIL_SEPARATION,
        radial_loops=AUTODIFF_RADIAL_LOOPS,
        axial_loops=AUTODIFF_AXIAL_LOOPS,
    )
    autodiff_sweep = [
        wham_mirror_pressure_drop_sensitivity(
            autodiff_problem,
            forcing=AUTODIFF_FORCING,
            peak_hartmann_number=AUTODIFF_PEAK_HARTMANN_NUMBER,
            coil_separation=float(separation),
            radial_loops=AUTODIFF_RADIAL_LOOPS,
            axial_loops=AUTODIFF_AXIAL_LOOPS,
        )
        for separation in AUTODIFF_SEPARATION_SWEEP
    ]
    autodiff_summary = {
        "reference_separation": COIL_SEPARATION,
        "pressure_drop_proxy": float(autodiff_reference["pressure_drop_proxy"]),
        "d_pressure_drop_d_separation": float(autodiff_reference["d_pressure_drop_d_separation"]),
        "separation_sweep": AUTODIFF_SEPARATION_SWEEP.tolist(),
        "pressure_drop_curve": [float(item["pressure_drop_proxy"]) for item in autodiff_sweep],
        "sensitivity_curve": [float(item["d_pressure_drop_d_separation"]) for item in autodiff_sweep],
    }

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
    wham_overview = write_wham_mirror_overview_plots(
        solution,
        table_path=table_path,
        pipe_radius=PIPE_RADIUS,
        coil_separation=COIL_SEPARATION,
        coil_inner_radius=COIL_INNER_RADIUS,
        coil_outer_radius=COIL_OUTER_RADIUS,
        out_dir=OUTPUT_DIR,
        case_title="WHAM mirror pipe",
        autodiff_summary=autodiff_summary,
    )
    summary = {
        "case": "wham_mirror_pipe",
        "field_table": table_path.name,
        "field_coordinate_frame": {
            "solver_x_min": float(x_field[0]),
            "solver_x_max": float(x_field[-1]),
            "coil_frame_x_offset": float(coil_frame_x_offset),
            "coil_frame_x_min": float(x_field[0] + coil_frame_x_offset),
            "coil_frame_x_max": float(x_field[-1] + coil_frame_x_offset),
        },
        "plots": [path.name for path in [*field_plots, *overview_plots, *wham_overview]],
        "field_quality": field_quality,
        "validation": validation,
        "autodiff": autodiff_summary,
    }
    (OUTPUT_DIR / "wham_mirror_pipe_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    run_wham_mirror_pipe_demo()
