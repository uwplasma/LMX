#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lmx.cases import make_hunt_case
from lmx.io import load_restart_bundle, validate_restart_bundle, write_restart_npz
from lmx.specs import BoundaryCondition
from lmx.solvers import _build_mesh, solve_steady
from lmx.validation import (
    combined_profile_error,
    extract_midplane_profile,
    inspect_reference_case,
    latest_field_minmax_record,
    latest_reference_sampled_profiles,
    normalize_sample_distance,
    compare_normalized_profiles,
    validation_summary,
)
from scripts.run_reference_parity_report import (
    infer_initial_velocity_x,
    infer_inlet_drive_mode,
    infer_inlet_flow_rate,
    infer_reduced_inlet_flow_rate,
    infer_magnetic_ramp,
)


def _portable_path(path: str | Path, *, relative_to: str | Path | None = None) -> str:
    candidate = Path(path)
    base = Path(relative_to) if relative_to is not None else Path.cwd()
    try:
        return str(candidate.relative_to(base))
    except ValueError:
        try:
            return str(candidate.resolve().relative_to(base.resolve()))
        except ValueError:
            return candidate.name if candidate.name else str(candidate)


def _derive_current_scaled_pressure_proxy_history(
    pressure_proxy_history: list[float],
    face_current_max_history: list[float],
    current_max_history: list[float],
) -> list[float]:
    if not pressure_proxy_history:
        return []
    current_history = face_current_max_history or current_max_history
    if not current_history or len(current_history) != len(pressure_proxy_history):
        return []
    reference_current = max(abs(float(current_history[0])), 1e-20)
    return [
        float(pressure_proxy) * float(current_value) / reference_current
        for pressure_proxy, current_value in zip(pressure_proxy_history, current_history)
    ]


def _build_case(
    ha: float,
    ny: int,
    nz: int,
    initial_velocity: float,
    drive_mode: str,
    inlet_flow_rate: float | None,
    dt: float,
    t_final: float,
    max_steps: int,
    outer_iterations: int | None,
    relaxation: float | None,
    potential_iterations: int | None,
    potential_relaxation: float | None,
    potential_solver: str | None,
    current_reconstruction: str | None,
    velocity_update_limit: float | None,
    velocity_update_limiter: str | None,
    post_update_potential_refresh: bool,
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
    if current_reconstruction is not None:
        updates["current_reconstruction"] = current_reconstruction
    if velocity_update_limit is not None:
        updates["velocity_update_limit"] = velocity_update_limit
    if velocity_update_limiter is not None:
        updates["velocity_update_limiter"] = velocity_update_limiter
    updates["post_update_potential_refresh"] = post_update_potential_refresh
    inlet_boundary: tuple[BoundaryCondition, ...] = ()
    if initial_velocity != 0.0:
        if drive_mode == "inlet_flow_rate":
            flow_rate = inlet_flow_rate
            if flow_rate is None:
                flow_rate = initial_velocity * case.geometry.width * case.geometry.height
            inlet_boundary = (BoundaryCondition("inlet", "inlet_flow_rate", value=flow_rate, axis="x"),)
        else:
            inlet_boundary = (BoundaryCondition("inlet", "inlet_velocity", value=(initial_velocity, 0.0, 0.0), axis="x"),)
    return replace(
        case,
        boundary_conditions=case.boundary_conditions + inlet_boundary,
        magnetic_field=replace(case.magnetic_field, ramp_start=ramp_start, ramp_duration=ramp_duration),
        initial_velocity=initial_velocity,
        forcing=0.0,
        time_stepper=replace(time_stepper, **updates),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare a Hunt LMX run against a recovered reference case using solver diagnostics first."
    )
    parser.add_argument("--reference-run-dir", type=Path, required=True)
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
    parser.add_argument(
        "--current-reconstruction",
        choices=["cell_centered", "face_averaged", "hybrid_face_lorentz"],
        default=None,
    )
    parser.add_argument("--drive-mode", choices=["auto", "inlet_velocity", "inlet_flow_rate"], default="auto")
    parser.add_argument("--velocity-update-limit", type=float, default=None)
    parser.add_argument("--velocity-update-limiter", choices=["global_scale", "local_clip"], default=None)
    parser.add_argument("--post-update-potential-refresh", action="store_true")
    parser.add_argument("--initial-velocity", type=float, default=None)
    parser.add_argument("--restart-npz", type=Path, default=None)
    parser.add_argument("--append-histories", action="store_true")
    parser.add_argument("--write-restart-npz", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    run_dir = args.reference_run_dir
    initial_velocity = args.initial_velocity
    if initial_velocity is None:
        initial_velocity = infer_initial_velocity_x(run_dir) or 0.0
    ramp_start, ramp_duration = infer_magnetic_ramp(run_dir)
    inferred_drive_mode = infer_inlet_drive_mode(run_dir)
    recovered_inlet_flow_rate = infer_inlet_flow_rate(run_dir)
    reduced_case_geometry = make_hunt_case(ha=args.ha, ny=args.ny, nz=args.nz).geometry
    reduced_area = reduced_case_geometry.width * reduced_case_geometry.height
    reduced_inlet_flow_rate = infer_reduced_inlet_flow_rate(
        run_dir,
        reduced_area=reduced_area,
        initial_velocity=initial_velocity,
    )
    drive_mode = (inferred_drive_mode or "inlet_velocity") if args.drive_mode == "auto" else args.drive_mode

    case = _build_case(
        ha=args.ha,
        ny=args.ny,
        nz=args.nz,
        initial_velocity=initial_velocity,
        drive_mode=drive_mode,
        inlet_flow_rate=reduced_inlet_flow_rate,
        dt=args.dt,
        t_final=args.t_final,
        max_steps=args.max_steps,
        outer_iterations=args.outer_iterations,
        relaxation=args.relaxation,
        potential_iterations=args.potential_iterations,
        potential_relaxation=args.potential_relaxation,
        potential_solver=args.potential_solver,
        current_reconstruction=args.current_reconstruction,
        velocity_update_limit=args.velocity_update_limit,
        velocity_update_limiter=args.velocity_update_limiter,
        post_update_potential_refresh=bool(args.post_update_potential_refresh),
        ramp_start=ramp_start,
        ramp_duration=ramp_duration,
    )
    initial_state = None
    initial_diagnostics = None
    restart_summary: dict[str, object] | None = None
    if args.restart_npz is not None:
        restart_bundle = load_restart_bundle(args.restart_npz)
        validate_restart_bundle(
            restart_bundle,
            mesh=_build_mesh(case),
            geometry_kind=case.geometry.kind,
            case_name=case.name,
        )
        initial_state = restart_bundle.state
        initial_diagnostics = restart_bundle.diagnostics
        restart_summary = {
            "input": str(restart_bundle.path),
            "start_time": float(restart_bundle.state.time),
            "append_histories": bool(args.append_histories),
        }
    try:
        solution = solve_steady(
            case,
            initial_state=initial_state,
            initial_diagnostics=initial_diagnostics,
            append_diagnostics=bool(args.append_histories),
        )
    except TypeError:
        solution = solve_steady(case)

    inspection = inspect_reference_case(run_dir)
    latest_u_record = latest_field_minmax_record(run_dir, field="mag(U)")
    sampled_profiles = latest_reference_sampled_profiles(run_dir)

    lmx_solver = {
        "case_name": solution.case_name,
        "time_stepper": asdict(case.time_stepper),
        "magnetic_field": asdict(case.magnetic_field),
        "diagnostics": validation_summary(solution, solution.case_name, ha=args.ha),
        "trace": {
            "time_history": solution.diagnostics.time_history.tolist(),
            "u_max_history": solution.diagnostics.u_max_history.tolist(),
            "mean_velocity_history": solution.diagnostics.mean_velocity_history.tolist(),
            "applied_forcing_history": solution.diagnostics.applied_forcing_history.tolist(),
            "pressure_proxy_history": solution.diagnostics.pressure_proxy_history.tolist(),
            "current_scaled_pressure_proxy_history": solution.diagnostics.current_scaled_pressure_proxy_history.tolist(),
            "raw_update_max_history": solution.diagnostics.raw_update_max_history.tolist(),
            "limiter_scale_history": solution.diagnostics.limiter_scale_history.tolist(),
            "limited_fraction_history": solution.diagnostics.limited_fraction_history.tolist(),
            "current_max_history": solution.diagnostics.current_max_history.tolist(),
            "face_current_max_history": solution.diagnostics.face_current_max_history.tolist(),
            "emf_max_history": solution.diagnostics.emf_max_history.tolist(),
            "lorentz_max_history": solution.diagnostics.lorentz_max_history.tolist(),
            "face_lorentz_max_history": solution.diagnostics.face_lorentz_max_history.tolist(),
            "residual_history": solution.diagnostics.residual_history.tolist(),
            "potential_residual_history": solution.diagnostics.potential_residual_history.tolist(),
            "potential_iterations_history": solution.diagnostics.potential_iterations_history.tolist(),
        },
    }
    if not lmx_solver["trace"]["current_scaled_pressure_proxy_history"]:
        lmx_solver["trace"]["current_scaled_pressure_proxy_history"] = _derive_current_scaled_pressure_proxy_history(
            lmx_solver["trace"]["pressure_proxy_history"],
            lmx_solver["trace"]["face_current_max_history"],
            lmx_solver["trace"]["current_max_history"],
        )

    reference_run = {
        "case_dir": _portable_path(run_dir),
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
        "drive_mode": drive_mode,
        "recovered_inlet_flow_rate": recovered_inlet_flow_rate,
        "reduced_inlet_flow_rate": reduced_inlet_flow_rate,
        "reference_run": reference_run,
        "lmx_solver": lmx_solver,
        "comparison": comparison,
    }
    if args.write_restart_npz is not None:
        restart_output = write_restart_npz(solution, case, args.write_restart_npz)
        restart_payload = {"output": _portable_path(restart_output)}
        if restart_summary is None:
            restart_summary = restart_payload
        else:
            restart_summary.update(restart_payload)
    if restart_summary is not None:
        payload["restart"] = restart_summary
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
