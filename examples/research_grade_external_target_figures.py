"""Generate publication-facing target figures for open research-grade lanes."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from lmx.research_figures import (
    write_research_grade_external_target_panel,
    write_research_grade_external_target_tables,
)


OUTPUT_DIR = Path("artifacts/examples/research_grade_external_targets")
DOCS_OUTPUT_DIR = Path("docs/_static/generated")
MAGNETIC_SUMMARY = DOCS_OUTPUT_DIR / "magnetic_obstacle_benchmark_summary.json"
Q2DMHDFOAM_SUMMARY = DOCS_OUTPUT_DIR / "q2dmhdfoam_external_reference_summary.json"
DEAN_SUMMARY = DOCS_OUTPUT_DIR / "bent_pipe_inductionless_summary.json"
OUTPUT_STEM = "research_grade_external_targets"
COPY_TO_DOCS = True


def run_research_grade_external_target_figures(
    *,
    out_dir: Path = OUTPUT_DIR,
    docs_output_dir: Path = DOCS_OUTPUT_DIR,
    copy_to_docs: bool = COPY_TO_DOCS,
) -> dict[str, object]:
    """Write open-lane target panel plus candidate reference-data tables."""

    out_dir.mkdir(parents=True, exist_ok=True)
    docs_output_dir.mkdir(parents=True, exist_ok=True)
    tables = write_research_grade_external_target_tables(
        out_dir,
        q2dmhdfoam_summary_path=Q2DMHDFOAM_SUMMARY,
        dean_summary_path=DEAN_SUMMARY,
        output_stem=OUTPUT_STEM,
    )
    plots = write_research_grade_external_target_panel(
        out_dir,
        magnetic_summary_path=MAGNETIC_SUMMARY,
        q2dmhdfoam_summary_path=Q2DMHDFOAM_SUMMARY,
        dean_summary_path=DEAN_SUMMARY,
        output_stem=OUTPUT_STEM,
    )
    copied: list[str] = []
    if copy_to_docs:
        for path in [*tables, *plots]:
            target = docs_output_dir / path.name
            shutil.copy2(path, target)
            copied.append(target.name)
        copied.append(f"{OUTPUT_STEM}_example_summary.json")
    summary = {
        "case": OUTPUT_STEM,
        "status": "publication_targets_generated_no_strict_closure",
        "plots": [path.name for path in plots],
        "tables": [path.name for path in tables],
        "docs_artifacts": copied,
        "notes": (
            "These figures document external targets and current gaps. They do "
            "not fill the strict matched reference CSVs used by release readiness."
        ),
    }
    summary_path = out_dir / f"{OUTPUT_STEM}_example_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if copy_to_docs:
        shutil.copy2(summary_path, docs_output_dir / summary_path.name)
    print(f"Research-grade external target figures written to {out_dir}")
    print(f"plots = {[path.name for path in plots]}")
    return summary


if __name__ == "__main__":
    run_research_grade_external_target_figures()
