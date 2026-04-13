from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_solver_control_sweep import _build_case, _collect_metrics, _parse_values, _replace_like
from lmx.solvers import _bounded_time_step_count, solve_steady
from lmx.validation import duct_layer_resolution_metrics


def _apply_parameter(case, parameter: str, value):
    time_stepper = _replace_like(case.time_stepper, **{parameter: value})
    if parameter == "dt":
        max_steps = _bounded_time_step_count(
            start_time=0.0,
            dt=float(value),
            t_final=case.time_stepper.t_final,
            max_steps=case.time_stepper.max_steps,
        )
        time_stepper = _replace_like(time_stepper, max_steps=max_steps)
    return _replace_like(case, time_stepper=time_stepper)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a two-parameter native LMX solver-control grid sweep.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/control_grid"))
    parser.add_argument("--case", choices=["hartmann", "shercliff", "hunt"], required=True)
    parser.add_argument("--ha", type=float, default=20.0)
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--wall-cells", type=int, default=None)
    parser.add_argument("--parameter-a", required=True)
    parser.add_argument("--values-a", type=str, required=True)
    parser.add_argument("--type-a", choices=["float", "int", "str"], default="float")
    parser.add_argument("--parameter-b", required=True)
    parser.add_argument("--values-b", type=str, required=True)
    parser.add_argument("--type-b", choices=["float", "int", "str"], default="float")
    parser.add_argument("--reference-root", type=Path, default=None)
    parser.add_argument("--x-slice", type=str, default="1m")
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    values_a = _parse_values(args.values_a, args.type_a)
    values_b = _parse_values(args.values_b, args.type_b)

    payload: dict[str, object] = {
        "case": args.case,
        "ha": args.ha,
        "resolution": args.resolution,
        "parameter_a": args.parameter_a,
        "values_a": values_a,
        "parameter_b": args.parameter_b,
        "values_b": values_b,
        "levels": [],
    }

    for value_a in values_a:
        for value_b in values_b:
            case_dir = args.output / f"{args.parameter_a}_{value_a}__{args.parameter_b}_{value_b}"
            case = _build_case(args.case, args.ha, args.resolution, case_dir, args.wall_cells)
            case = _apply_parameter(case, args.parameter_a, value_a)
            case = _apply_parameter(case, args.parameter_b, value_b)
            solution = solve_steady(case)
            metrics = _collect_metrics(solution, args.case, args.ha, reference_root=args.reference_root, x_slice=args.x_slice)
            payload["levels"].append(
                {
                    "parameter_a_value": value_a,
                    "parameter_b_value": value_b,
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
