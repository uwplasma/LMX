"""Post-process the bent-pipe baseline against a Bayat-Rezai Dean target.

The current bent-pipe inductionless example is a low-Dean-number current-closure
gate. This lightweight artifact writes a moderate-Dean-number Bayat-Rezai
reference target and compares it to the current LMX secondary-flow observables.
The comparison is expected to fail until LMX has a resolved or explicitly
reduced higher-inertia Dean-flow model coupled to the curved-pipe solve.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import numpy as np

from lmx.dean import bayat_rezai_dean_velocity, dean_secondary_flow_field
from lmx.external_validation import (
    compare_scalar_reference_observables,
    load_scalar_reference_observables,
    write_scalar_reference_comparison_plots,
    write_scalar_reference_comparison_table,
)


OUTPUT_DIR = Path("artifacts/examples/dean_vortex_bayat_rezai_strict_attempt")
DOCS_OUTPUT_DIR = Path("docs/_static/generated")
BENT_PIPE_SUMMARY_PATH = DOCS_OUTPUT_DIR / "bent_pipe_inductionless_summary.json"
STRICT_REFERENCE_FILENAME = "dean_vortex_reference_observables.csv"
COPY_TO_DOCS = True

REFERENCE_DEAN_NUMBER = 20.0
REFERENCE_AXIAL_VELOCITY = 0.66
KINEMATIC_VISCOSITY = 1.0e-6
LARGEST_CHANNEL_DIMENSION = 150.0e-6
RELATIVE_TOLERANCE = 0.15


def run_dean_vortex_bayat_rezai_strict_attempt(
    *,
    out_dir: Path = OUTPUT_DIR,
    docs_output_dir: Path = DOCS_OUTPUT_DIR,
    bent_pipe_summary_path: Path = BENT_PIPE_SUMMARY_PATH,
    copy_to_docs: bool = COPY_TO_DOCS,
) -> dict[str, object]:
    """Write the current Bayat-Rezai strict-attempt comparison artifacts."""

    out_dir.mkdir(parents=True, exist_ok=True)
    docs_output_dir.mkdir(parents=True, exist_ok=True)
    bent_summary = json.loads(Path(bent_pipe_summary_path).read_text(encoding="utf-8"))
    validation = dict(bent_summary.get("validation", {}))
    if not validation:
        raise ValueError("bent-pipe summary has no validation payload")

    reference_path = _write_bayat_rezai_reference_rows(
        out_dir / STRICT_REFERENCE_FILENAME
    )
    reference_observables = load_scalar_reference_observables(
        reference_path, context="Dean-vortex reference CSV"
    )
    lmx_observables = {
        key: float(value)
        for key, value in validation.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    comparison = compare_scalar_reference_observables(
        lmx_observables, reference_observables
    )
    table_path = write_scalar_reference_comparison_table(
        comparison,
        out_dir / "dean_vortex_reference_comparison.csv",
    )
    plot_paths = write_scalar_reference_comparison_plots(
        comparison,
        out_dir,
        output_stem="dean_vortex_reference_comparison",
        title="Dean-vortex Bayat-Rezai strict observables",
        no_data_label="No compared Dean-vortex observables",
    )
    external_reference_comparison = {
        "status": "external_reference_compared",
        "validation_pass": bool(comparison["validation_pass"]),
        "reference_path": reference_path.name,
        "comparison_table": table_path.name,
        "plots": [path.name for path in plot_paths],
        "comparison": comparison,
        "comparison_source": (
            "Postprocessed from bent_pipe_inductionless_summary.json using a "
            "moderate-Dean-number Bayat-Rezai target."
        ),
    }
    patched_bent_summary = {
        **bent_summary,
        "external_reference_comparison": external_reference_comparison,
    }
    patched_summary_path = out_dir / "bent_pipe_inductionless_summary.json"
    patched_summary_path.write_text(
        json.dumps(patched_bent_summary, indent=2) + "\n", encoding="utf-8"
    )

    summary = {
        "case": "dean_vortex_bayat_rezai_strict_attempt",
        "status": "external_reference_compared_mismatch",
        "strict_blocker_closed": False,
        "reference_dean_number": REFERENCE_DEAN_NUMBER,
        "current_lmx_dean_number": float(validation.get("dean_number", 0.0)),
        "external_reference_comparison": external_reference_comparison,
        "notes": (
            "The current LMX bent-pipe baseline is intentionally low-De and has "
            "near-zero secondary flow. The Bayat-Rezai target is a moderate-De "
            "secondary-flow scale, so this artifact quantifies the remaining "
            "higher-inertia Dean-vortex blocker."
        ),
        "docs_artifacts": [],
    }
    summary_path = out_dir / "dean_vortex_bayat_rezai_strict_attempt_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    copied: list[str] = []
    if copy_to_docs:
        for path in [
            reference_path,
            table_path,
            *plot_paths,
            patched_summary_path,
            summary_path,
        ]:
            target = docs_output_dir / path.name
            shutil.copy2(path, target)
            copied.append(target.name)
        summary["docs_artifacts"] = copied
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        shutil.copy2(summary_path, docs_output_dir / summary_path.name)

    print(f"Dean-vortex Bayat-Rezai strict attempt written to {out_dir}")
    print(f"validation_pass = {comparison['validation_pass']}")
    return summary


def _write_bayat_rezai_reference_rows(path: Path) -> Path:
    dean_velocity = float(
        bayat_rezai_dean_velocity(
            REFERENCE_DEAN_NUMBER,
            kinematic_viscosity=KINEMATIC_VISCOSITY,
            largest_channel_dimension=LARGEST_CHANNEL_DIMENSION,
        )
    )
    rms_ratio = dean_velocity / REFERENCE_AXIAL_VELOCITY
    y = np.linspace(-1.0, 1.0, 81)
    z = np.linspace(-1.0, 1.0, 81)
    field = dean_secondary_flow_field(
        y, z, tube_radius=1.0, target_rms_velocity=dean_velocity
    )
    peak_ratio = float(field["peak_velocity"]) / REFERENCE_AXIAL_VELOCITY
    rows = [
        {
            "observable": "secondary_flow_rms_ratio",
            "value": f"{rms_ratio:.16g}",
            "tolerance": f"{RELATIVE_TOLERANCE * rms_ratio:.16g}",
            "relative_tolerance": f"{RELATIVE_TOLERANCE:.16g}",
            "units": "dimensionless",
            "source": "Bayat & Rezai, Scientific Reports 7, 13655 (2017), Eq. 8 and Eq. 9",
            "note": f"Moderate-De target at De={REFERENCE_DEAN_NUMBER:g}, U={REFERENCE_AXIAL_VELOCITY:g} m/s.",
        },
        {
            "observable": "secondary_flow_peak_ratio",
            "value": f"{peak_ratio:.16g}",
            "tolerance": f"{RELATIVE_TOLERANCE * peak_ratio:.16g}",
            "relative_tolerance": f"{RELATIVE_TOLERANCE:.16g}",
            "units": "dimensionless",
            "source": "Bayat-Rezai average Dean velocity projected through the LMX reduced two-cell field",
            "note": "Peak/rms ratio comes from dean_secondary_flow_field on the reference cross-section.",
        },
    ]
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
    run_dean_vortex_bayat_rezai_strict_attempt()
