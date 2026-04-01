from __future__ import annotations

import argparse
import json
import sys
from dataclasses import is_dataclass, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lmx.cases import make_hartmann_case, make_hunt_case, make_shercliff_case
from lmx.solvers import solve_steady
from lmx.validation import (
    closed_channel_validation,
    combined_profile_error,
    duct_layer_resolution_metrics,
    hartmann_acceptance,
    hartmann_validation,
    processed_slice_validation,
    validation_summary,
)


def _replace_like(obj, **changes):
    if is_dataclass(obj):
        return replace(obj, **changes)
    if hasattr(obj, "__dict__"):
        values = {**vars(obj), **changes}
        return obj.__class__(**values)
    raise TypeError(f"Unsupported object replacement for {type(obj)!r}")


def _parse_values(raw: str, kind: str) -> list[float | int | str]:
    items = [item.strip() for item in raw.split(",") if item.strip()]
    if kind == "int":
        return [int(item) for item in items]
    if kind == "str":
        return items
    return [float(item) for item in items]


def _build_case(case_kind: str, ha: float, resolution: int, output_dir: Path, wall_cells: int | None):
    if case_kind == "hartmann":
        return make_hartmann_case(ha=ha, ny=resolution, nz=resolution, output_dir=str(output_dir))
    if case_kind == "shercliff":
        return make_shercliff_case(ha=ha, ny=resolution, nz=resolution, output_dir=str(output_dir))
    if case_kind == "hunt":
        return make_hunt_case(
            ha=ha,
            ny=resolution,
            nz=resolution,
            wall_cells=wall_cells if wall_cells is not None else max(2, round(resolution * 8 / 72)),
            output_dir=str(output_dir),
        )
    raise ValueError(case_kind)


def _collect_metrics(solution, case_kind: str, ha: float, *, reference_root: Path | None, x_slice: str) -> dict[str, float | str]:
    case_name = f"{case_kind}_ha{int(ha)}"
    metrics = validation_summary(solution, case_name, ha=ha)
    if case_kind == "hartmann":
        comparison = hartmann_validation(solution, ha)
        acceptance = hartmann_acceptance(solution, ha, l2_threshold=0.05, linf_threshold=0.1)
        metrics["l2_error"] = comparison.l2_error
        metrics["linf_error"] = comparison.linf_error
        metrics["accepted"] = float(acceptance.passed)
    elif reference_root is not None:
        comparison = closed_channel_validation(solution, case_kind, int(ha), reference_root=reference_root)
        metrics["y_l2_error"] = comparison.y_profile.l2_error
        metrics["z_l2_error"] = comparison.z_profile.l2_error
        metrics["combined_l2_error"] = combined_profile_error(
            comparison.y_profile.l2_error,
            comparison.z_profile.l2_error,
        )
        metrics["combined_linf_error"] = combined_profile_error(
            comparison.y_profile.linf_error,
            comparison.z_profile.linf_error,
        )
        try:
            slice_report = processed_slice_validation(
                solution,
                case_kind,
                int(ha),
                x_slice=x_slice,
                reference_root=reference_root,
            )
        except FileNotFoundError:
            slice_report = None
        if slice_report is not None:
            metrics["slice_y_l2_error"] = slice_report.y_profile.l2_error
            metrics["slice_z_l2_error"] = slice_report.z_profile.l2_error
            metrics["slice_combined_l2_error"] = combined_profile_error(
                slice_report.y_profile.l2_error,
                slice_report.z_profile.l2_error,
            )
            metrics["slice_combined_linf_error"] = combined_profile_error(
                slice_report.y_profile.linf_error,
                slice_report.z_profile.linf_error,
            )
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a native LMX solver-control sweep.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/control_sweep"))
    parser.add_argument("--case", choices=["hartmann", "shercliff", "hunt"], required=True)
    parser.add_argument("--ha", type=float, default=20.0)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--wall-cells", type=int, default=None)
    parser.add_argument(
        "--parameter",
        choices=[
            "outer_iterations",
            "potential_iterations",
            "potential_tolerance",
            "potential_relaxation",
            "potential_solver",
            "current_reconstruction",
            "velocity_update_limit",
            "relaxation",
            "dt",
        ],
        required=True,
    )
    parser.add_argument("--values", type=str, required=True)
    parser.add_argument("--value-type", choices=["float", "int", "str"], default="float")
    parser.add_argument("--reference-root", type=Path, default=None)
    parser.add_argument("--x-slice", type=str, default="1m")
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    values = _parse_values(args.values, args.value_type)
    payload: dict[str, object] = {
        "case": args.case,
        "ha": args.ha,
        "resolution": args.resolution,
        "parameter": args.parameter,
        "values": values,
        "levels": [],
    }

    for value in values:
        case_dir = args.output / f"{args.parameter}_{value}"
        case = _build_case(args.case, args.ha, args.resolution, case_dir, args.wall_cells)
        time_stepper = _replace_like(case.time_stepper, **{args.parameter: value})
        if args.parameter == "dt":
            max_steps = max(1, int(round(case.time_stepper.t_final / float(value))))
            time_stepper = _replace_like(time_stepper, max_steps=max_steps)
        case = _replace_like(case, time_stepper=time_stepper)
        solution = solve_steady(case)
        metrics = _collect_metrics(solution, args.case, args.ha, reference_root=args.reference_root, x_slice=args.x_slice)
        payload["levels"].append(
            {
                "parameter_value": value,
                "dt": case.time_stepper.dt,
                "max_steps": float(case.time_stepper.max_steps),
                **duct_layer_resolution_metrics(case, solution.mesh),
                **metrics,
            }
        )

    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2))
    print(summary_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
