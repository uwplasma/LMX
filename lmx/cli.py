from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp

from .benchmarks import benchmark_solver, write_benchmark_report
from .cases import make_hartmann_case, make_hunt_case, make_shercliff_case
from .io import write_paraview
from .solvers import solve_steady
from .validation import (
    closed_channel_validation,
    extract_centerline,
    extract_midplane_profile,
    hartmann_validation,
    validation_summary,
    write_analytic_comparison,
    write_closed_channel_validation,
    write_metrics_json,
    write_profile_csv,
)


def _build_case(args: argparse.Namespace):
    if args.case == "hartmann":
        return make_hartmann_case(ha=args.ha, output_dir=args.output)
    if args.case == "shercliff":
        return make_shercliff_case(ha=args.ha, output_dir=args.output)
    if args.case == "hunt":
        return make_hunt_case(ha=args.ha, output_dir=args.output)
    raise ValueError(args.case)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="lmx")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("case", choices=["hartmann", "shercliff", "hunt"])
    run_parser.add_argument("--ha", type=float, default=20.0)
    run_parser.add_argument("--output", type=str, default="./out")

    bench_parser = subparsers.add_parser("benchmark")
    bench_parser.add_argument("--repeats", type=int, default=3)
    bench_parser.add_argument("--ha", type=float, default=20.0)
    bench_parser.add_argument("--ny", type=int, default=48)
    bench_parser.add_argument("--nz", type=int, default=48)
    bench_parser.add_argument("--output", type=str, default="")

    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("case", choices=["hartmann", "shercliff", "hunt"])
    validate_parser.add_argument("--ha", type=float, default=20.0)
    validate_parser.add_argument("--output", type=str, default="./out")
    validate_parser.add_argument("--reference-root", type=str, default="")

    args = parser.parse_args(argv)

    if args.command == "benchmark":
        payload = benchmark_solver(repeats=args.repeats, ha=args.ha, ny=args.ny, nz=args.nz)
        if args.output:
            write_benchmark_report(payload, args.output)
        print(json.dumps(payload, indent=2))
        return 0

    if args.command == "validate":
        case = _build_case(args)
        solution = solve_steady(case)
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        write_paraview(solution, out_dir)
        write_profile_csv(out_dir / f"{case.name}_centerline.csv", extract_centerline(solution))
        z_profile = extract_midplane_profile(solution, axis="z")
        write_profile_csv(out_dir / f"{case.name}_midplane_z.csv", z_profile)
        payload = validation_summary(solution, case.name, ha=args.ha)
        if args.case == "hartmann":
            comparison = hartmann_validation(solution, args.ha)
            write_analytic_comparison(comparison, out_dir / f"{case.name}_analytic.json", axis_name="y")
        elif args.reference_root:
            comparison = closed_channel_validation(solution, args.case, int(args.ha), reference_root=args.reference_root)
            write_closed_channel_validation(comparison, out_dir / f"{case.name}_analytic.json")
            payload["y_l2_error"] = comparison.y_profile.l2_error
            payload["y_linf_error"] = comparison.y_profile.linf_error
            payload["z_l2_error"] = comparison.z_profile.l2_error
            payload["z_linf_error"] = comparison.z_profile.linf_error
        write_metrics_json(payload, out_dir / f"{case.name}_metrics.json")
        print(json.dumps(payload, indent=2))
        return 0

    case = _build_case(args)
    solution = solve_steady(case)
    out_dir = Path(args.output)
    write_paraview(solution, out_dir)
    write_profile_csv(out_dir / f"{case.name}_centerline.csv", extract_centerline(solution))
    print(
        json.dumps(
            {
                "case": case.name,
                "time": solution.state.time,
                "residual": solution.state.residual,
                "output": str(out_dir.resolve()),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
