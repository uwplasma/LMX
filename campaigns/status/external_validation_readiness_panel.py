from __future__ import annotations

import json
from pathlib import Path

from lmx.external_validation import (
    external_validation_readiness_rows,
    summarize_external_validation_readiness,
    write_external_validation_readiness_panel,
)


OUTPUT_DIR = Path("artifacts/examples/external_validation_readiness")
OUTPUT_STEM = "external_validation_readiness"


def run_external_validation_readiness_panel(
    *,
    out_dir: Path = OUTPUT_DIR,
) -> dict[str, object]:
    """Write the external-code readiness plot and machine-readable summary."""

    out_dir.mkdir(parents=True, exist_ok=True)
    rows = external_validation_readiness_rows()
    plot_paths = write_external_validation_readiness_panel(
        rows, out_dir, output_stem=OUTPUT_STEM
    )
    summary = {
        "case": "external_validation_readiness",
        "plots": [path.name for path in plot_paths],
        "rows": rows,
        "summary": summarize_external_validation_readiness(rows),
        "notes": (
            "This panel records executable external-code readiness. It is not a "
            "physics acceptance claim for open lanes until the referenced "
            "observable CSVs or field-comparison artifacts are filled."
        ),
    }
    (out_dir / "external_validation_readiness_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


if __name__ == "__main__":
    run_external_validation_readiness_panel()
