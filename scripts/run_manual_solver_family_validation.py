from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lmx.cases import make_hartmann_case, make_hunt_case, make_shercliff_case
from lmx.fringing import build_square_duct_extruded_problem, solve_extruded_inductionless
from lmx.solvers import solve_steady
from lmx.validation import (
    closed_channel_validation,
    combined_profile_error,
    duct_layer_resolution_metrics,
    hartmann_acceptance,
    validation_summary,
)


def _cases(ha: float, resolution: int):
    return [
        make_hartmann_case(ha=ha, ny=resolution, nz=resolution),
        make_shercliff_case(ha=ha, ny=resolution, nz=resolution),
        make_hunt_case(ha=ha, ny=resolution, nz=resolution, wall_cells=2),
    ]


def _bounded_case(case, *, max_steps: int, potential_iterations: int, coupling_iterations: int):
    return replace(
        case,
        time_stepper=replace(
            case.time_stepper,
            max_steps=min(case.time_stepper.max_steps, max_steps),
            potential_iterations=min(case.time_stepper.potential_iterations, potential_iterations),
        ),
        solver=replace(case.solver, coupling_iterations=min(case.solver.coupling_iterations, coupling_iterations)),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the heavier manual LMX solver-family validation lane.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/manual_validation/solver_family_summary.json"))
    parser.add_argument("--ha-values", type=str, default="10,20")
    parser.add_argument("--resolution", type=int, default=24)
    parser.add_argument("--reference-root", type=Path, default=None)
    parser.add_argument("--hartmann-l2-threshold", type=float, default=0.05)
    parser.add_argument("--hartmann-linf-threshold", type=float, default=0.1)
    parser.add_argument("--include-fringing", action="store_true")
    parser.add_argument("--fringing-nx", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--potential-iterations", type=int, default=80)
    parser.add_argument("--coupling-iterations", type=int, default=8)
    args = parser.parse_args(argv)

    ha_values = [float(item) for item in args.ha_values.split(",") if item]
    summary: dict[str, dict[str, float | str]] = {}

    for ha in ha_values:
        for case in _cases(ha, args.resolution):
            case = _bounded_case(
                case,
                max_steps=args.max_steps,
                potential_iterations=args.potential_iterations,
                coupling_iterations=args.coupling_iterations,
            )
            solution = solve_steady(case)
            metrics = validation_summary(solution, case.name, ha=ha)
            metrics.update(duct_layer_resolution_metrics(case, solution.mesh))
            if case.name.startswith("hartmann"):
                acceptance = hartmann_acceptance(
                    solution,
                    ha,
                    l2_threshold=args.hartmann_l2_threshold,
                    linf_threshold=args.hartmann_linf_threshold,
                )
                metrics["accepted"] = float(acceptance.passed)
                metrics["acceptance_l2_error"] = acceptance.l2_error
                metrics["acceptance_linf_error"] = acceptance.linf_error
            elif args.reference_root is not None:
                comparison = closed_channel_validation(solution, case.name.split("_", 1)[0], int(ha), reference_root=args.reference_root)
                metrics["combined_l2_error"] = combined_profile_error(
                    comparison.y_profile.l2_error,
                    comparison.z_profile.l2_error,
                )
                metrics["combined_linf_error"] = combined_profile_error(
                    comparison.y_profile.linf_error,
                    comparison.z_profile.linf_error,
                )
            summary[case.name] = metrics

    if args.include_fringing:
        for ha in ha_values:
            problem = build_square_duct_extruded_problem(
                ha_peak=ha,
                ny=max(6, args.resolution // 2),
                nz=max(6, args.resolution // 2),
                nx_stations=args.fringing_nx,
            )
            problem = replace(
                problem,
                case=_bounded_case(
                    problem.case,
                    max_steps=args.max_steps,
                    potential_iterations=args.potential_iterations,
                    coupling_iterations=args.coupling_iterations,
                ),
            )
            solution = solve_extruded_inductionless(problem)
            summary[f"fringing_ha{int(ha)}"] = {
                "station_count": float(solution.validation.station_count),
                "max_residual": solution.validation.max_residual,
                "max_charge_balance_residual": solution.validation.max_charge_balance_residual,
                "mean_velocity_span": solution.validation.mean_velocity_span,
                "volumetric_flow_rate_span": solution.validation.volumetric_flow_rate_span,
                "field_mean_velocity_correlation": solution.validation.field_mean_velocity_correlation,
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(args.output.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
