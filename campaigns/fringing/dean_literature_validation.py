"""Generate the Dean-flow literature validation gate.

This example validates the reduced Dean-flow correlation and visualization
surface used to plan the higher-inertia bent-pipe lane. It does not mark the
current inductionless bent-pipe solve as a resolved Dean-vortex validation.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

from lmx.dean import (
    DeanVelocityPoint,
    compare_dean_velocity_points,
    dean_velocity_reference_rows,
    write_dean_literature_validation_plots,
)


OUTPUT_DIR = Path("artifacts/examples/dean_literature_validation")
DOCS_OUTPUT_DIR = Path("docs/_static/generated")
OUTPUT_STEM = "dean_literature_validation"
COPY_TO_DOCS = True

KINEMATIC_VISCOSITY = 1.0e-6
LARGEST_CHANNEL_DIMENSION = 150.0e-6
AXIAL_VELOCITIES = (0.15, 0.22, 0.44, 0.66, 0.74)
DEAN_NUMBERS = (2.73, 6.82, 10.0, 20.0, 30.0)


def run_dean_literature_validation(
    *,
    out_dir: Path = OUTPUT_DIR,
    docs_output_dir: Path = DOCS_OUTPUT_DIR,
    copy_to_docs: bool = COPY_TO_DOCS,
) -> dict[str, object]:
    """Write Dean-flow literature correlation plots, CSV, and JSON."""

    out_dir.mkdir(parents=True, exist_ok=True)
    docs_output_dir.mkdir(parents=True, exist_ok=True)
    points = [
        DeanVelocityPoint(
            dean_number=dean,
            axial_velocity=axial,
            kinematic_viscosity=KINEMATIC_VISCOSITY,
            largest_channel_dimension=LARGEST_CHANNEL_DIMENSION,
        )
        for dean, axial in zip(DEAN_NUMBERS, AXIAL_VELOCITIES, strict=True)
    ]
    comparison = compare_dean_velocity_points(points)
    reference_rows = dean_velocity_reference_rows(
        DEAN_NUMBERS,
        kinematic_viscosity=KINEMATIC_VISCOSITY,
        largest_channel_dimension=LARGEST_CHANNEL_DIMENSION,
    )
    table_path = _write_dean_reference_table(
        reference_rows, out_dir / "dean_literature_reference_observables.csv"
    )
    plots = write_dean_literature_validation_plots(
        comparison, out_dir, output_stem=OUTPUT_STEM
    )
    summary = {
        **comparison,
        "case": OUTPUT_STEM,
        "status": "literature_correlation_gate_passed_current_bent_solver_still_open",
        "reference_table": table_path.name,
        "plots": [path.name for path in plots],
        "docs_artifacts": [],
        "notes": (
            "This closes the Dean correlation/test-data side of the blocker. "
            "The strict bent-pipe lane still requires a solved LMX secondary-flow "
            "state and external curved-pipe parity before release readiness can "
            "mark higher-inertia Dean-vortex validation closed."
        ),
    }
    summary_path = out_dir / f"{OUTPUT_STEM}_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    copied: list[str] = []
    if copy_to_docs:
        for path in [table_path, *plots, summary_path]:
            target = docs_output_dir / path.name
            shutil.copy2(path, target)
            copied.append(target.name)
        summary["docs_artifacts"] = copied
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        shutil.copy2(summary_path, docs_output_dir / summary_path.name)

    print(f"Dean literature validation written to {out_dir}")
    print(f"validation_pass = {summary['validation_pass']}")
    return summary


def _write_dean_reference_table(rows: list[dict[str, float | str]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "observable",
        "value",
        "tolerance",
        "relative_tolerance",
        "units",
        "source",
        "note",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return path


if __name__ == "__main__":
    run_dean_literature_validation()
