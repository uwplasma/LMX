#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lmx.cases import make_hartmann_case, make_hunt_case, make_shercliff_case
from lmx.specs import BoundaryCondition
from lmx.validation import ValidationReport, compare_with_freemhd, write_validation_report


def infer_initial_velocity_x(case_dir: str | Path) -> float | None:
    candidates = [
        Path(case_dir) / "0" / "liquid" / "U",
        Path(case_dir) / "0" / "fluid" / "U",
    ]
    pattern = re.compile(r"internalField\s+uniform\s+\(\s*(\S+)\s+\S+\s+\S+\s*\)")
    for path in candidates:
        if not path.exists():
            continue
        text = path.read_text()
        match = pattern.search(text)
        if match is None:
            continue
        return float(match.group(1))
    return None


def build_case(
    case_kind: str,
    ha: float,
    ny: int,
    nz: int,
    initial_velocity: float,
    dt: float,
    t_final: float,
    max_steps: int,
    forcing: float,
):
    factories = {
        "hartmann": make_hartmann_case,
        "shercliff": make_shercliff_case,
        "hunt": make_hunt_case,
    }
    case = factories[case_kind](ha=ha, ny=ny, nz=nz)
    return replace(
        case,
        initial_velocity=initial_velocity,
        forcing=forcing,
        time_stepper=replace(case.time_stepper, dt=dt, t_final=t_final, max_steps=max_steps),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build an LMX case, compare it with a FreeMHD run directory, and write a parity JSON report.")
    parser.add_argument("--case-kind", choices=("hartmann", "shercliff", "hunt"), required=True)
    parser.add_argument("--ha", type=float, required=True)
    parser.add_argument("--freemhd-run-dir", type=Path, required=True)
    parser.add_argument("--ny", type=int, default=16)
    parser.add_argument("--nz", type=int, default=16)
    parser.add_argument("--dt", type=float, default=1e-5)
    parser.add_argument("--t-final", type=float, default=1e-4)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--forcing", type=float, default=None)
    parser.add_argument("--initial-velocity", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    initial_velocity = args.initial_velocity
    if initial_velocity is None:
        initial_velocity = infer_initial_velocity_x(args.freemhd_run_dir) or 0.0

    case = build_case(
        case_kind=args.case_kind,
        ha=args.ha,
        ny=args.ny,
        nz=args.nz,
        initial_velocity=initial_velocity,
        dt=args.dt,
        t_final=args.t_final,
        max_steps=args.max_steps,
        forcing=0.0 if args.forcing is None else args.forcing,
    )
    drive_mode = "explicit_forcing"
    if args.forcing is None and args.case_kind == "hunt":
        inlet_bc = BoundaryCondition("inlet", "inlet_velocity", value=(initial_velocity, 0.0, 0.0), axis="x")
        case = replace(case, boundary_conditions=case.boundary_conditions + (inlet_bc,))
        drive_mode = "inlet_velocity"
    elif args.forcing is None:
        drive_mode = "none"
    report = compare_with_freemhd(case, args.freemhd_run_dir)
    write_validation_report(report, args.output)
    payload = {
        "case_kind": args.case_kind,
        "ha": args.ha,
        "initial_velocity": initial_velocity,
        "forcing": case.forcing,
        "drive_mode": drive_mode,
        "freemhd_run_dir": str(args.freemhd_run_dir.resolve()),
        "output": str(args.output.resolve()),
        "metrics": report.metrics,
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
