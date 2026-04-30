"""Post-process the magnetic-obstacle benchmark against a digitized Votyakov target.

This example is intentionally lightweight: it does not rerun the localized-field
solve. It reads the current benchmark summary, filters the already documented
Votyakov Fig. 7(a) candidate row into the strict reference CSV contract, and
writes the failed comparison artifacts used to track the open magnetic-obstacle
lane.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

from lmx import (
    compare_magnetic_obstacle_reference_observables,
    load_magnetic_obstacle_reference_observables,
    write_magnetic_obstacle_reference_comparison_plots,
    write_magnetic_obstacle_reference_comparison_table,
)


OUTPUT_DIR = Path("artifacts/examples/magnetic_obstacle_votyakov_strict_attempt")
DOCS_OUTPUT_DIR = Path("docs/_static/generated")
BENCHMARK_SUMMARY_PATH = DOCS_OUTPUT_DIR / "magnetic_obstacle_benchmark_summary.json"
CANDIDATE_REFERENCE_PATH = DOCS_OUTPUT_DIR / "magnetic_obstacle_reference_observables_candidate.csv"
STRICT_REFERENCE_FILENAME = "magnetic_obstacle_reference_observables.csv"
COPY_TO_DOCS = True


def run_magnetic_obstacle_votyakov_strict_attempt(
    *,
    out_dir: Path = OUTPUT_DIR,
    docs_output_dir: Path = DOCS_OUTPUT_DIR,
    benchmark_summary_path: Path = BENCHMARK_SUMMARY_PATH,
    candidate_reference_path: Path = CANDIDATE_REFERENCE_PATH,
    copy_to_docs: bool = COPY_TO_DOCS,
) -> dict[str, object]:
    """Write the current Votyakov strict-attempt comparison artifacts."""

    out_dir.mkdir(parents=True, exist_ok=True)
    docs_output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_summary = json.loads(Path(benchmark_summary_path).read_text(encoding="utf-8"))
    lmx_observables = dict(benchmark_summary.get("external_readiness", {}).get("observables", {}))
    if not lmx_observables:
        raise ValueError("magnetic obstacle benchmark summary has no external_readiness observables")

    reference_path = _write_filled_candidate_rows(candidate_reference_path, out_dir / STRICT_REFERENCE_FILENAME)
    reference_observables = load_magnetic_obstacle_reference_observables(reference_path)
    comparison = compare_magnetic_obstacle_reference_observables(lmx_observables, reference_observables)
    table_path = write_magnetic_obstacle_reference_comparison_table(
        comparison,
        out_dir / "magnetic_obstacle_reference_comparison.csv",
    )
    plot_paths = write_magnetic_obstacle_reference_comparison_plots(comparison, out_dir)
    external_reference_comparison = {
        "status": "external_reference_compared",
        "validation_pass": bool(comparison["validation_pass"]),
        "reference_path": reference_path.name,
        "comparison_table": table_path.name,
        "plots": [path.name for path in plot_paths],
        "comparison": comparison,
        "comparison_source": (
            "Postprocessed from magnetic_obstacle_benchmark_summary.json using "
            "the digitized Votyakov Fig. 7(a) candidate target."
        ),
    }
    patched_benchmark_summary = {
        **benchmark_summary,
        "external_reference_comparison": external_reference_comparison,
    }
    patched_summary_path = out_dir / "magnetic_obstacle_benchmark_summary.json"
    patched_summary_path.write_text(json.dumps(patched_benchmark_summary, indent=2) + "\n", encoding="utf-8")
    summary = {
        "case": "magnetic_obstacle_votyakov_strict_attempt",
        "status": "external_reference_compared_mismatch",
        "strict_blocker_closed": False,
        "reference_source": "Votyakov et al. magnetic-obstacle centerline digitization",
        "benchmark_summary": patched_summary_path.name,
        "external_reference_comparison": external_reference_comparison,
        "notes": (
            "The current LMX localized-field reduced case does not reproduce the "
            "negative centerline velocity required by the Votyakov recirculation "
            "target. This artifact narrows the magnetic-obstacle lane from "
            "template-only to an explicit external-observable mismatch."
        ),
        "docs_artifacts": [],
    }
    summary_path = out_dir / "magnetic_obstacle_votyakov_strict_attempt_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    copied: list[str] = []
    if copy_to_docs:
        for path in [reference_path, table_path, *plot_paths, patched_summary_path, summary_path]:
            target = docs_output_dir / path.name
            shutil.copy2(path, target)
            copied.append(target.name)
        summary["docs_artifacts"] = copied
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        shutil.copy2(summary_path, docs_output_dir / summary_path.name)

    print(f"Magnetic-obstacle Votyakov strict attempt written to {out_dir}")
    print(f"validation_pass = {comparison['validation_pass']}")
    return summary


def _write_filled_candidate_rows(candidate_path: Path, output_path: Path) -> Path:
    rows: list[dict[str, str]] = []
    with Path(candidate_path).open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if (row.get("value") or "").strip() and (row.get("tolerance") or "").strip():
                rows.append(row)
    if not rows:
        raise ValueError(f"{candidate_path} does not contain any filled candidate rows")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    columns = ("observable", "value", "tolerance", "relative_tolerance", "units", "source", "note")
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return output_path


if __name__ == "__main__":
    run_magnetic_obstacle_votyakov_strict_attempt()
