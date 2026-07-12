from __future__ import annotations

import json
from pathlib import Path

from lmx.external_validation import write_dean_vortex_reference_template


OUTPUT_DIR = Path("artifacts/examples/dean_vortex_external_reference_template")
TEMPLATE_FILENAME = "dean_vortex_reference_observables.csv"


def run_dean_vortex_external_reference_template() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    template = write_dean_vortex_reference_template(OUTPUT_DIR / TEMPLATE_FILENAME)
    summary = {
        "case": "dean_vortex_external_reference_template",
        "status": "template_only_no_external_reference_claim",
        "template": template.name,
        "notes": (
            "Fill this CSV with matched higher-inertia curved-pipe or curved-duct "
            "observables before claiming Dean-vortex validation."
        ),
    }
    (OUTPUT_DIR / "dean_vortex_external_reference_template_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    run_dean_vortex_external_reference_template()
