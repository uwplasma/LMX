#!/usr/bin/env python3
"""Build or verify the deterministic Benchmark B specification-freeze index."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from lmx.benchmarks import (
    BENCHMARK_B_SPEC_FILES,
    load_benchmark_b_reference,
    load_benchmark_b_spec,
)


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_specification_index(root: Path = ROOT) -> dict[str, Any]:
    cases = []
    for case_id, filename in BENCHMARK_B_SPEC_FILES.items():
        spec = load_benchmark_b_spec(case_id, root)
        reference = load_benchmark_b_reference(case_id, root)
        spec_path = root / "benchmarks" / "specs" / filename
        data_path = root / spec["reference"]["data_path"]
        cases.append(
            {
                "id": case_id,
                "spec_path": str(spec_path.relative_to(root)),
                "spec_sha256": _sha256(spec_path),
                "data_path": str(data_path.relative_to(root)),
                "data_sha256": _sha256(data_path),
                "reference_row_count": len(reference["x_over_L"]),
                "parameters": {
                    "hartmann_number": spec["physics"]["hartmann_number"],
                    "interaction_parameter": spec["physics"]["interaction_parameter"],
                    "wall_conductance_ratio": spec["wall"]["wall_conductance_ratio"],
                },
                "primary_observable": spec["reference"]["primary_observable"],
                "combined_uncertainty_absolute": spec["reference"][
                    "combined_uncertainty_absolute"
                ],
                "numerical_independence": {
                    "steady_residual_max": spec["solver"]["steady_residual_max"],
                    "steady_residual_uncertainty_fraction_max": spec["solver"][
                        "steady_residual_uncertainty_fraction_max"
                    ],
                    "coupling_acceleration": spec["solver"]["coupling_acceleration"],
                    "coupling_history_depth": spec["solver"]["coupling_history_depth"],
                    "electric_iterations_min": spec["solver"][
                        "electric_iterations_min"
                    ],
                    "electric_tolerance_max": spec["solver"]["electric_tolerance_max"],
                    "projection_iterations_min": spec["solver"][
                        "projection_iterations_min"
                    ],
                    "projection_tolerance_max": spec["solver"][
                        "projection_tolerance_max"
                    ],
                    "tolerance_factor": spec["solver"]["tolerance_independence_factor"],
                    "tolerance_uncertainty_fraction_max": spec["solver"][
                        "tolerance_independence_uncertainty_fraction_max"
                    ],
                    "iteration_factor": spec["solver"]["iteration_independence_factor"],
                    "iteration_uncertainty_fraction_max": spec["solver"][
                        "iteration_independence_uncertainty_fraction_max"
                    ],
                    "thin_wall_relative_max": spec["wall"][
                        "thickness_independence_relative_max"
                    ],
                },
                "mesh_levels": [level["name"] for level in spec["mesh"]["levels"]],
                "source_ids": [source["id"] for source in spec["sources"]],
                "tolerances_frozen_before_production": spec[
                    "tolerances_frozen_before_production"
                ],
                **(
                    {"harness_smoke_execution": spec["harness_smoke_execution"]}
                    if case_id == "B2-fringing-square"
                    else {}
                ),
            }
        )
    return {
        "schema_version": 1,
        "benchmark": "B: ALEX 3D laminar fringing-field validation",
        "specification_freeze_pass": len(cases) == 2
        and all(case["tolerances_frozen_before_production"] for case in cases),
        "production_results_included": False,
        "cases": cases,
        "production_blockers": [
            "Pass steady/tolerance and nominal-versus-confirmation thin-wall independence for both frozen builders.",
            "Run three meshes and one exactly matched FreeMHD case before accepting Benchmark B.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/benchmark-b-specification.json"),
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = build_specification_index()
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.check:
        if (
            not args.output.is_file()
            or args.output.read_text(encoding="utf-8") != encoded
        ):
            raise SystemExit("Benchmark B specification index is stale")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(
        f"{args.output} specification_freeze_pass="
        f"{payload['specification_freeze_pass']}"
    )
    return 0 if payload["specification_freeze_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
