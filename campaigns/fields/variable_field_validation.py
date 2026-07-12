from __future__ import annotations

import json
from pathlib import Path

from lmx.field_models import (
    cross_section_divergence_metrics,
    make_divergence_free_cross_section_field,
    sample_cross_section_field,
    save_cross_section_divergence_report,
)
from lmx.plotting import write_cross_section_field_plots


OUTPUT_DIR = Path("artifacts/examples/variable_field_validation")
WIDTH = 2.4
HEIGHT = 1.6
BASE_BZ = 12.0
PERTURBATION = 0.12
NY = 81
NZ = 81


def run_variable_field_validation() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    field_fn = make_divergence_free_cross_section_field(
        width=WIDTH,
        height=HEIGHT,
        base_bz=BASE_BZ,
        perturbation=PERTURBATION,
    )
    y, z, field = sample_cross_section_field(
        field_fn, width=WIDTH, height=HEIGHT, ny=NY, nz=NZ
    )
    plots = write_cross_section_field_plots(
        y=y,
        z=z,
        field=field,
        out_dir=OUTPUT_DIR,
        title="Analytic divergence-free cross-sectional magnetic field",
    )
    metrics = cross_section_divergence_metrics(
        field_fn, width=WIDTH, height=HEIGHT, ny=NY, nz=NZ
    )
    metrics_path = save_cross_section_divergence_report(
        metrics, OUTPUT_DIR / "field_divergence_metrics.json"
    )
    summary = {
        "case": "variable_field_validation",
        "plots": [path.name for path in plots],
        "metrics_path": metrics_path.name,
        "metrics": metrics,
    }
    (OUTPUT_DIR / "variable_field_validation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


if __name__ == "__main__":
    run_variable_field_validation()
