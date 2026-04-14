from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lmx.cases import make_hartmann_case, make_hunt_case, make_shercliff_case
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the heavier manual LMX solver-family validation lane.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/manual_validation/solver_family_summary.json"))
    parser.add_argument("--ha-values", type=str, default="10,20")
    parser.add_argument("--resolution", type=int, default=24)
    parser.add_argument("--reference-root", type=Path, default=None)
    parser.add_argument("--hartmann-l2-threshold", type=float, default=0.05)
    parser.add_argument("--hartmann-linf-threshold", type=float, default=0.1)
    args = parser.parse_args(argv)

    ha_values = [float(item) for item in args.ha_values.split(",") if item]
    summary: dict[str, dict[str, float | str]] = {}

    for ha in ha_values:
        for case in _cases(ha, args.resolution):
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

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2))
    print(args.output.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
