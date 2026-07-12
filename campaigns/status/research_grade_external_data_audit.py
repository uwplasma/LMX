"""Write the external-data audit for strict research-grade LMX blockers."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from lmx.research_closure import write_research_grade_external_data_audit


OUTPUT_DIR = Path("artifacts/examples/research_grade_external_data_audit")
DOCS_OUTPUT_DIR = Path("docs/_static/generated")
COPY_TO_DOCS = True


def run_research_grade_external_data_audit(
    *,
    out_dir: Path = OUTPUT_DIR,
    docs_output_dir: Path = DOCS_OUTPUT_DIR,
    copy_to_docs: bool = COPY_TO_DOCS,
) -> dict[str, object]:
    """Record which local external-code/data inputs exist for open blockers."""

    out_dir.mkdir(parents=True, exist_ok=True)
    docs_output_dir.mkdir(parents=True, exist_ok=True)
    audit_path = write_research_grade_external_data_audit(
        out_dir, static_dir=docs_output_dir
    )
    summary = json.loads(audit_path.read_text(encoding="utf-8"))
    copied: list[str] = []
    if copy_to_docs:
        target = docs_output_dir / audit_path.name
        shutil.copy2(audit_path, target)
        copied.append(target.name)
    summary["docs_artifacts"] = copied
    audit_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if copy_to_docs:
        shutil.copy2(audit_path, docs_output_dir / audit_path.name)

    print(f"Research-grade external-data audit written to {out_dir}")
    print(
        f"available_sources = {summary['available_source_count']}/{summary['source_count']}"
    )
    print(f"matched_reference_csvs = {summary['matched_reference_csv_count']}")
    return summary


if __name__ == "__main__":
    run_research_grade_external_data_audit()
