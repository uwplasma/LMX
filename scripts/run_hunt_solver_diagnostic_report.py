#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lmx.cases import make_hunt_case
from lmx.solvers import solve_steady
from lmx.validation import (
    combined_profile_error,
    extract_midplane_profile,
    inspect_freemhd_case,
    latest_field_minmax_record,
    latest_sampled_profiles,
    normalize_sample_distance,
    compare_normalized_profiles,
    validation_summary,
)
from scripts.run_freemhd_parity_report import infer_initial_velocity_x, infer_magnetic_ramp


def _build_case(
    ha: float,
    ny: int,
    nz: int,
    initial_velocity: float,
    dt: float,
    t_final: float,
    max_steps: int,
    outer_iterations: int | None,
    relaxation: float | None,
    potential_iterations: int | None,
    potential_relaxation: float | None,
    potential_solver: str | None,
    velocity_update_limit: float | None,
    ramp_start: float,
    ramp_duration: float,
) -> object:
    case = make_hunt_case(ha=ha, ny=ny, nz=nz)
    time_stepper = case.time_stepper
    updates = {
        "dt": dt,
        "t_final": t_final,
        "max_steps": max_steps,
    }
    if outer_iterations is not None:
        updates["outer_iterations"] = outer_iterations
    if relaxation is not None:
        updates["relaxation"] = relaxation
    if potential_iterations is not None:
        updates["potential_iterations"] = potential_iterations
    if potential_relaxation is not None:
        updates["potential_relaxation"] = potential_relaxation
    if potential_solver is not None:
        updates["potential_solver"] = potential_solver
    if velocity_update_limit is not None:
        updates["velocity_update_limit"] = velocity_update_limit
    return replace(
        case,
        magnetic_field=replace(case.magnetic_field, ramp_start=ramp_start, ramp_duration=ramp_duration),
        initial_velocity=initial_velocity,
        forcing=0.0,
        time_stepper=replace(time_stepper, **updates),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare a Hunt LMX run against a recovered FreeMHD case using solver diagnostics first."
    )
    parser.add_argument("--freemhd-run-dir", type=Path, required=True)
    parser.add_argument("--ha", type=float, required=True)
    parser.add_argument("--ny", type=int, default=32)
    parser.add_argument("--nz", type=int, default=32)
    parser.add_argument("--dt", type=float, default=0.002)
    parser.add_argument("--t-final", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--outer-iterations", type=int, default=None)
    parser.add_argument("--relaxation", type=float, default=None)
    parser.add_argument("--potential-iterations", type=int, default=None)
    parser.add_argument("--potential-relaxation", type=float, default=None)
    parser.add_argument("--potential-solver", choices=["auto", "jacobi", "cg", "cg_volume", "lineax_cg"], default=None)
    parser.add_argument("--velocity-update-limit", type=float, default=None)
    parser.add_argument("--initial-velocity", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    run_dir = args.freemhd_run_dir
    initial_velocity = args.initial_velocity
    if initial_velocity is None:
        initial_velocity = infer_initial_velocity_x(run_dir) or 0.0
    ramp_start, ramp_duration = infer_magnetic_ramp(run_dir)

    case = _build_case(
        ha=args.ha,
        ny=args.ny,
        nz=args.nz,
        initial_velocity=initial_velocity,
        dt=args.dt,
        t_final=args.t_final,
        max_steps=args.max_steps,
        outer_iterations=args.outer_iterations,
        relaxation=args.relaxation,
        potential_iterations=args.potential_iterations,
        potential_relaxation=args.potential_relaxation,
        potential_solver=args.potential_solver,
        velocity_update_limit=args.velocity_update_limit,
        ramp_start=ramp_start,
        ramp_duration=ramp_duration,
    )
    solution = solve_steady(case)

    inspection = inspect_freemhd_case(run_dir)
    latest_u_record = latest_field_minmax_record(run_dir, field="mag(U)")
    sampled_profiles = latest_sampled_profiles(run_dir)

    lmx_solver = {
        "case_name": solution.case_name,
        "time_stepper": asdict(case.time_stepper),
        "magnetic_field": asdict(case.magnetic_field),
        "diagnostics": validation_summary(solution, solution.case_name, ha=args.ha),
        "trace": {
            "time_history": solution.diagnostics.time_history.tolist(),
            "u_max_history": solution.diagnostics.u_max_history.tolist(),
            "current_max_history": solution.diagnostics.current_max_history.tolist(),
            "lorentz_max_history": solution.diagnostics.lorentz_max_history.tolist(),
            "residual_history": solution.diagnostics.residual_history.tolist(),
            "potential_residual_history": solution.diagnostics.potential_residual_history.tolist(),
            "potential_iterations_history": solution.diagnostics.potential_iterations_history.tolist(),
        },
    }

    freemhd_run = {
        "case_dir": str(run_dir.resolve()),
        "inspection": {
            "control_dicts": len(inspection.control_dicts),
            "region_properties": len(inspection.region_properties),
            "latest_time_dirs": len(inspection.latest_time_dirs),
            "region_zero_dirs": len(inspection.region_zero_dirs),
            "zero_field_files": len(inspection.zero_field_files),
            "processor_layout_dirs": len(inspection.processor_layout_dirs),
            "parallel_time_dirs": len(inspection.parallel_time_dirs),
        },
        "latest_u_max": None if latest_u_record is None else latest_u_record.max_value,
        "latest_time": None if latest_u_record is None else latest_u_record.time,
        "sampled_profile_pair_available": sampled_profiles is not None,
    }

    comparison: dict[str, float | str | None] = {
        "u_max_abs_diff": None,
        "sample_time": None,
        "sample_y_l2_error": None,
        "sample_z_l2_error": None,
        "sample_combined_l2_error": None,
    }
    if latest_u_record is not None:
        comparison["u_max_abs_diff"] = abs(float(solution.state.u.max()) - latest_u_record.max_value)
    if sampled_profiles is not None:
        y_sample, z_sample = sampled_profiles
        y_profile = extract_midplane_profile(solution, axis="y", fluid_only=True)
        z_profile = extract_midplane_profile(solution, axis="z", fluid_only=True)
        y_comparison = compare_normalized_profiles(
            y_profile["y"],
            y_profile["u"],
            normalize_sample_distance(y_sample.distance),
            y_sample.u_x,
        )
        z_comparison = compare_normalized_profiles(
            z_profile["z"],
            z_profile["u"],
            normalize_sample_distance(z_sample.distance),
            z_sample.u_x,
        )
        comparison.update(
            {
                "sample_time": float(Path(y_sample.path).parent.name),
                "sample_y_l2_error": y_comparison.l2_error,
                "sample_z_l2_error": z_comparison.l2_error,
                "sample_combined_l2_error": combined_profile_error(
                    y_comparison.l2_error,
                    z_comparison.l2_error,
                ),
            }
        )

    payload = {
        "case_kind": "hunt",
        "ha": args.ha,
        "initial_velocity": initial_velocity,
        "freemhd_run": freemhd_run,
        "lmx_solver": lmx_solver,
        "comparison": comparison,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
