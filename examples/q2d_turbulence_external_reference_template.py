from __future__ import annotations

import json
from pathlib import Path

from lmx import write_q2d_turbulence_reference_template


OUTPUT_DIR = Path("artifacts/examples/q2d_turbulence_external_reference_template")
TEMPLATE_FILENAME = "q2d_turbulence_reference_observables.csv"


def run_q2d_turbulence_external_reference_template() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    template = write_q2d_turbulence_reference_template(OUTPUT_DIR / TEMPLATE_FILENAME)
    summary = {
        "case": "q2d_turbulence_external_reference_template",
        "status": "template_only_no_external_reference_claim",
        "template": template.name,
        "notes": (
            "Fill this CSV with matched Sommeria-Moreau-style turbulent "
            "observables before claiming external Q2D turbulence parity."
        ),
    }
    (OUTPUT_DIR / "q2d_turbulence_external_reference_template_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


if __name__ == "__main__":
    run_q2d_turbulence_external_reference_template()
