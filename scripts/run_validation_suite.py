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
    extract_centerline,
    extract_midplane_profile,
    hartmann_validation,
    validation_summary,
    write_analytic_comparison,
    write_metrics_json,
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
        summary[case.name] = metrics

    summary_path = args.output / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(summary_path.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
