from __future__ import annotations

import json
from pathlib import Path
import shutil

from lmx import (
    load_magnetic_obstacle_votyakov_digitized_curve,
    magnetic_obstacle_votyakov_curve_observables,
    write_magnetic_obstacle_votyakov_curve_comparison,
)


OUTPUT_DIR = Path("artifacts/examples/magnetic_obstacle_votyakov_curve_validation")
DOCS_OUTPUT_DIR = Path("docs/_static/generated")
VOTYAKOV_DIGITIZED_CURVE = DOCS_OUTPUT_DIR / "magnetic_obstacle_votyakov_fig7a_digitized.csv"
BENCHMARK_SUMMARY_PATH = DOCS_OUTPUT_DIR / "magnetic_obstacle_benchmark_summary.json"
OUTPUT_STEM = "magnetic_obstacle_votyakov_curve_comparison"
COPY_TO_DOCS = True


def run_magnetic_obstacle_votyakov_curve_validation(
    *,
    out_dir: Path = OUTPUT_DIR,
    docs_output_dir: Path = DOCS_OUTPUT_DIR,
    digitized_curve_path: Path = VOTYAKOV_DIGITIZED_CURVE,
    benchmark_summary_path: Path = BENCHMARK_SUMMARY_PATH,
    copy_to_docs: bool = COPY_TO_DOCS,
) -> dict[str, object]:
    """Write a full digitized-curve magnetic-obstacle mismatch artifact."""

    out_dir.mkdir(parents=True, exist_ok=True)
    docs_output_dir.mkdir(parents=True, exist_ok=True)
    records = load_magnetic_obstacle_votyakov_digitized_curve(digitized_curve_path)
    benchmark_summary = json.loads(Path(benchmark_summary_path).read_text(encoding="utf-8"))
    lmx_observables = dict(benchmark_summary.get("external_readiness", {}).get("observables", {}))
    if not lmx_observables:
        raise ValueError("magnetic obstacle benchmark summary has no external_readiness observables")

    derived = magnetic_obstacle_votyakov_curve_observables(records)
    paths = write_magnetic_obstacle_votyakov_curve_comparison(
        records,
        lmx_observables,
        out_dir,
        output_stem=OUTPUT_STEM,
    )
    copied: list[str] = []
    if copy_to_docs:
        for path in paths:
            target = docs_output_dir / path.name
            shutil.copy2(path, target)
            copied.append(target.name)

    experiment = next(row for row in derived if row["series"] == "experiment_Ha140")
    lmx_minimum = float(lmx_observables["minimum_centerline_velocity_ratio"])
    target_plateau = float(experiment["plateau_minimum_centerline_velocity_ratio"])
    summary = {
        "case": "magnetic_obstacle_votyakov_curve_validation",
        "status": "literature_curve_compared_mismatch",
        "strict_blocker_closed": False,
        "digitized_curve": str(digitized_curve_path),
        "benchmark_summary": str(benchmark_summary_path),
        "derived_reference_observables": derived,
        "lmx_minimum_centerline_velocity_ratio": lmx_minimum,
        "target_plateau_minimum_centerline_velocity_ratio": target_plateau,
        "absolute_gap_to_plateau": abs(lmx_minimum - target_plateau),
        "plots": [path.name for path in paths if path.suffix in {".png", ".pdf"}],
        "tables": [path.name for path in paths if path.suffix == ".csv"],
        "docs_artifacts": copied,
        "notes": (
            "This is a literature-curve diagnostic, not a closed validation. "
            "Votyakov reverse-flow onset and high-N plateau require inertial "
            "recirculating magnetic-obstacle physics, while the current LMX "
            "localized-field gate remains a conservative inductionless response solve."
        ),
    }
    summary_path = out_dir / "magnetic_obstacle_votyakov_curve_validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if copy_to_docs:
        shutil.copy2(summary_path, docs_output_dir / summary_path.name)
    return summary


if __name__ == "__main__":
    run_magnetic_obstacle_votyakov_curve_validation()
