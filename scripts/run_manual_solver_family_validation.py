from __future__ import annotations

import argparse
from dataclasses import replace
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lmx.cases import make_hartmann_case, make_hunt_case, make_shercliff_case
from lmx.fringing import (
    build_layered_duct_extruded_problem,
    build_pipe_ogrid_extruded_problem,
    build_square_duct_extruded_problem,
    solve_extruded_inductionless,
)
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
    parser.add_argument("--fringing-geometries", type=str, default="rect_duct,layered_duct,pipe_ogrid")
    parser.add_argument("--fringing-nx", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--potential-iterations", type=int, default=80)
    parser.add_argument("--coupling-iterations", type=int, default=8)
    parser.add_argument("--max-charge-balance", type=float, default=1.0e-5)
    parser.add_argument("--max-interface-current", type=float, default=1.0e-4)
    parser.add_argument("--max-fringing-wall-current-leakage", type=float, default=5.0e-4)
    parser.add_argument("--max-fringing-boundary-current", type=float, default=5.0e-4)
    parser.add_argument("--fail-on-threshold", action="store_true")
    args = parser.parse_args(argv)

    ha_values = [float(item) for item in args.ha_values.split(",") if item]
    summary: dict[str, dict[str, float | str]] = {}
    failures: list[str] = []

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
            conservation_pass = (
                float(metrics.get("charge_balance_residual", 0.0)) <= args.max_charge_balance
                and float(metrics.get("interface_current_residual", 0.0)) <= args.max_interface_current
            )
            metrics["conservation_pass"] = float(conservation_pass)
            metrics["charge_balance_threshold"] = float(args.max_charge_balance)
            metrics["interface_current_threshold"] = float(args.max_interface_current)
            if not conservation_pass:
                failures.append(case.name)
            summary[case.name] = metrics

    if args.include_fringing:
        geometry_kinds = [item.strip() for item in args.fringing_geometries.split(",") if item.strip()]
        for ha in ha_values:
            for geometry_kind in geometry_kinds:
                if geometry_kind == "rect_duct":
                    problem = build_square_duct_extruded_problem(
                        ha_peak=ha,
                        ny=max(6, args.resolution // 2),
                        nz=max(6, args.resolution // 2),
                        nx_stations=args.fringing_nx,
                    )
                elif geometry_kind == "layered_duct":
                    problem = build_layered_duct_extruded_problem(
                        ha_peak=ha,
                        ny=max(6, args.resolution // 2),
                        nz=max(6, args.resolution // 2),
                        wall_cells=1,
                        insulator_cells=1,
                        nx_stations=args.fringing_nx,
                    )
                elif geometry_kind == "pipe_ogrid":
                    problem = build_pipe_ogrid_extruded_problem(
                        ha_peak=ha,
                        nr=max(6, args.resolution // 2),
                        ntheta=max(12, args.resolution),
                        nx_stations=args.fringing_nx,
                    )
                else:
                    raise ValueError(f"Unsupported fringing geometry {geometry_kind!r}")
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
                key = f"fringing_{geometry_kind}_ha{int(ha)}"
                summary[key] = {
                    "station_count": float(solution.validation.station_count),
                    "max_residual": solution.validation.max_residual,
                    "max_charge_balance_residual": solution.validation.max_charge_balance_residual,
                    "mean_velocity_span": solution.validation.mean_velocity_span,
                    "volumetric_flow_rate_span": solution.validation.volumetric_flow_rate_span,
                    "axial_current_span": solution.validation.axial_current_span,
                    "max_wall_current_leakage": solution.validation.max_wall_current_leakage,
                    "net_boundary_current_residual": solution.validation.net_boundary_current_residual,
                    "field_mean_velocity_correlation": solution.validation.field_mean_velocity_correlation,
                    "conservation_pass": float(
                        solution.validation.max_charge_balance_residual <= args.max_charge_balance
                        and solution.validation.max_wall_current_leakage <= args.max_fringing_wall_current_leakage
                        and solution.validation.net_boundary_current_residual <= args.max_fringing_boundary_current
                    ),
                    "charge_balance_threshold": float(args.max_charge_balance),
                    "wall_current_leakage_threshold": float(args.max_fringing_wall_current_leakage),
                    "boundary_current_threshold": float(args.max_fringing_boundary_current),
                }
                if not bool(summary[key]["conservation_pass"]):
                    failures.append(key)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(args.output.read_text())
    if failures and args.fail_on_threshold:
        print(f"Conservation thresholds failed for: {', '.join(failures)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
