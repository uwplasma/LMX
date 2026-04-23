#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lmx.freemhd import (
    _extract_inlet_block,
    build_case_from_freemhd_reference,
    infer_initial_velocity_x,
    infer_inlet_drive_mode,
    infer_inlet_flow_rate,
    infer_magnetic_ramp,
    infer_reduced_inlet_flow_rate,
)
from lmx.validation import ValidationReport, compare_with_reference_outputs, write_validation_report


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


def build_case(
    case_kind: str,
    *,
    ha: float,
    ny: int,
    nz: int,
    initial_velocity: float,
    dt: float,
    t_final: float,
    max_steps: int,
    forcing: float,
    drive_mode: str | None = None,
    inlet_flow_rate: float | None = None,
    ramp_start: float = 0.0,
    ramp_duration: float = 0.0,
):
    from dataclasses import replace

    from lmx.cases import make_hartmann_case, make_hunt_case, make_shercliff_case
    from lmx.specs import BoundaryCondition

    factories = {
        "hartmann": make_hartmann_case,
        "shercliff": make_shercliff_case,
        "hunt": make_hunt_case,
    }
    case = factories[case_kind](ha=ha, ny=ny, nz=nz)
    boundary_conditions = case.boundary_conditions
    if case_kind == "hunt" and forcing == 0.0 and initial_velocity != 0.0 and drive_mode is not None:
        if drive_mode == "inlet_flow_rate":
            flow_rate = inlet_flow_rate
            if flow_rate is None:
                flow_rate = initial_velocity * case.geometry.width * case.geometry.height
            inlet_bc = BoundaryCondition("inlet", "inlet_flow_rate", value=flow_rate, axis="x")
        else:
            inlet_bc = BoundaryCondition("inlet", "inlet_velocity", value=(initial_velocity, 0.0, 0.0), axis="x")
        boundary_conditions = boundary_conditions + (inlet_bc,)
    return replace(
        case,
        boundary_conditions=boundary_conditions,
        magnetic_field=replace(case.magnetic_field, ramp_start=ramp_start, ramp_duration=ramp_duration),
        initial_velocity=initial_velocity,
        forcing=forcing,
        time_stepper=replace(case.time_stepper, dt=dt, t_final=t_final, max_steps=max_steps),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build an LMX case, compare it with a reference case output directory, and write a parity JSON report.")
    parser.add_argument("--case-kind", choices=("hartmann", "shercliff", "hunt"), required=True)
    parser.add_argument("--ha", type=float, required=True)
    parser.add_argument("--reference-run-dir", type=Path, required=True)
    parser.add_argument("--ny", type=int, default=16)
    parser.add_argument("--nz", type=int, default=16)
    parser.add_argument("--dt", type=float, default=1e-5)
    parser.add_argument("--t-final", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--forcing", type=float, default=None)
    parser.add_argument("--initial-velocity", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    recovered_inlet_flow_rate = infer_inlet_flow_rate(args.reference_run_dir) if args.case_kind == "hunt" else None
    reduced_inlet_flow_rate = None
    if args.case_kind == "hunt":
        initial_velocity = infer_initial_velocity_x(args.reference_run_dir) or 0.0
        geometry_case = build_case(
            args.case_kind,
            ha=args.ha,
            ny=args.ny,
            nz=args.nz,
            initial_velocity=initial_velocity,
            dt=args.dt,
            t_final=args.t_final,
            max_steps=args.max_steps,
            forcing=0.0,
        )
        reduced_inlet_flow_rate = infer_reduced_inlet_flow_rate(
            args.reference_run_dir,
            reduced_area=geometry_case.geometry.width * geometry_case.geometry.height,
            initial_velocity=initial_velocity,
        )
    case = build_case_from_freemhd_reference(
        case_kind=args.case_kind,
        ha=args.ha,
        ny=args.ny,
        nz=args.nz,
        dt=args.dt,
        t_final=args.t_final,
        max_steps=args.max_steps,
        reference_run_dir=args.reference_run_dir,
        forcing=args.forcing,
    )
    report = compare_with_reference_outputs(case, args.reference_run_dir)
    write_validation_report(report, args.output)
    payload = {
        "case_kind": args.case_kind,
        "ha": args.ha,
        "initial_velocity": case.initial_velocity,
        "forcing": case.forcing,
        "drive_mode": "explicit_forcing" if args.forcing is not None else ((infer_inlet_drive_mode(args.reference_run_dir) or "inlet_velocity") if args.case_kind == "hunt" else "none"),
        "recovered_inlet_flow_rate": recovered_inlet_flow_rate,
        "reduced_inlet_flow_rate": reduced_inlet_flow_rate,
        "magnetic_ramp_start": case.magnetic_field.ramp_start,
        "magnetic_ramp_duration": case.magnetic_field.ramp_duration,
        "reference_run_dir": _portable_path(args.reference_run_dir),
        "output": _portable_path(args.output),
        "metrics": report.metrics,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
