from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lmx.cases import make_hartmann_case, make_hunt_case, make_shercliff_case
from lmx.io import write_paraview
from lmx.solvers import solve_steady
from lmx.validation import (
    closed_channel_validation,
    combined_profile_error,
    extract_centerline,
    extract_midplane_profile,
    hartmann_acceptance,
    hartmann_validation,
    processed_slice_validation,
    validation_summary,
    write_acceptance_report,
    write_analytic_comparison,
    write_closed_channel_validation,
    write_metrics_json,
    write_processed_slice_validation,
    write_profile_csv,
)


def _cases(ha: float, output_dir: Path):
    return [
        make_hartmann_case(ha=ha, ny=16, nz=16, output_dir=str(output_dir / "hartmann")),
        make_shercliff_case(ha=ha, ny=16, nz=16, output_dir=str(output_dir / "shercliff")),
        make_hunt_case(ha=ha, ny=16, nz=16, wall_cells=2, output_dir=str(output_dir / "hunt")),
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the LMX validation artifact suite.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/validation"))
    parser.add_argument("--ha", type=float, default=5.0)
    parser.add_argument("--reference-root", type=Path, default=None)
    parser.add_argument("--x-slice", type=str, default="1m")
    parser.add_argument("--hartmann-l2-threshold", type=float, default=0.05)
    parser.add_argument("--hartmann-linf-threshold", type=float, default=0.1)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict[str, float | str]] = {}

    for case in _cases(args.ha, args.output):
        solution = solve_steady(case)
        case_dir = args.output / case.name
        case_dir.mkdir(parents=True, exist_ok=True)
        write_paraview(solution, case_dir)
        write_profile_csv(case_dir / "centerline.csv", extract_centerline(solution))
        write_profile_csv(case_dir / "midplane_z.csv", extract_midplane_profile(solution, axis="z"))
        metrics = validation_summary(solution, case.name, ha=args.ha)
        write_metrics_json(metrics, case_dir / "metrics.json")
        if case.name.startswith("hartmann"):
            comparison = hartmann_validation(solution, args.ha)
            write_analytic_comparison(comparison, case_dir / "analytic.json", axis_name="y")
            acceptance = hartmann_acceptance(
                solution,
                args.ha,
                l2_threshold=args.hartmann_l2_threshold,
                linf_threshold=args.hartmann_linf_threshold,
            )
            write_acceptance_report(acceptance, case_dir / "acceptance.json")
            metrics["accepted"] = float(acceptance.passed)
            metrics["acceptance_l2_threshold"] = acceptance.l2_threshold
            metrics["acceptance_linf_threshold"] = acceptance.linf_threshold
        elif args.reference_root is not None:
            comparison = closed_channel_validation(
                solution,
                case.name.split("_", 1)[0],
                int(args.ha),
                reference_root=args.reference_root,
            )
            write_closed_channel_validation(comparison, case_dir / "analytic.json")
            metrics["y_l2_error"] = comparison.y_profile.l2_error
            metrics["y_linf_error"] = comparison.y_profile.linf_error
            metrics["z_l2_error"] = comparison.z_profile.l2_error
            metrics["z_linf_error"] = comparison.z_profile.linf_error
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
                    case.name.split("_", 1)[0],
                    int(args.ha),
                    x_slice=args.x_slice,
                    reference_root=args.reference_root,
                )
            except FileNotFoundError:
                slice_report = None
            if slice_report is not None:
                write_processed_slice_validation(slice_report, case_dir / "slice.json")
                metrics["slice_y_l2_error"] = slice_report.y_profile.l2_error
                metrics["slice_y_linf_error"] = slice_report.y_profile.linf_error
                metrics["slice_z_l2_error"] = slice_report.z_profile.l2_error
                metrics["slice_z_linf_error"] = slice_report.z_profile.linf_error
                metrics["slice_combined_l2_error"] = combined_profile_error(
                    slice_report.y_profile.l2_error,
                    slice_report.z_profile.l2_error,
                )
                metrics["slice_combined_linf_error"] = combined_profile_error(
                    slice_report.y_profile.linf_error,
                    slice_report.z_profile.linf_error,
                )
            write_metrics_json(metrics, case_dir / "metrics.json")
        summary[case.name] = metrics

    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(summary_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
