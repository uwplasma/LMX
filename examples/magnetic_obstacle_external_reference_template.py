from __future__ import annotations

import json
from pathlib import Path

from lmx import (
    magnetic_obstacle_literature_reference_cases,
    magnetic_obstacle_reference_template_rows,
    write_magnetic_obstacle_reference_template,
)


OUTPUT_DIR = Path("artifacts/examples/magnetic_obstacle_external_reference")
TEMPLATE_PATH = OUTPUT_DIR / "magnetic_obstacle_reference_observables_template.csv"


def run_magnetic_obstacle_external_reference_template() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    template_path = write_magnetic_obstacle_reference_template(TEMPLATE_PATH)
    references = magnetic_obstacle_literature_reference_cases()
    summary = {
        "case": "magnetic_obstacle_external_reference_template",
        "status": "template_only_no_external_reference_claim",
        "template": template_path.name,
        "template_observables": [row["observable"] for row in magnetic_obstacle_reference_template_rows()],
        "registered_reference_cases": sorted(references),
        "notes": (
            "Fill this CSV with digitized literature or experimental scalar observables "
            "before promoting the magnetic-obstacle lane from internal response to "
            "external parity validation."
        ),
    }
    (OUTPUT_DIR / "magnetic_obstacle_external_reference_template_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    run_magnetic_obstacle_external_reference_template()
