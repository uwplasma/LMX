"""Write the strict research-grade closure status artifacts for LMX."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from lmx.research_closure import write_research_grade_closure_status


OUTPUT_DIR = Path("artifacts/examples/research_grade_closure_status")
DOCS_OUTPUT_DIR = Path("docs/_static/generated")
COPY_TO_DOCS = True


def run_research_grade_closure_status(
    *,
    out_dir: Path = OUTPUT_DIR,
    docs_output_dir: Path = DOCS_OUTPUT_DIR,
    copy_to_docs: bool = COPY_TO_DOCS,
) -> dict[str, object]:
    """Collect current strict blocker status and write JSON/CSV artifacts."""

    out_dir.mkdir(parents=True, exist_ok=True)
    docs_output_dir.mkdir(parents=True, exist_ok=True)
    outputs = write_research_grade_closure_status(out_dir, static_dir=docs_output_dir)
    copied: list[str] = []
    if copy_to_docs:
        for path in outputs:
            target = docs_output_dir / path.name
            shutil.copy2(path, target)
            copied.append(target.name)

    summary_path = out_dir / "research_grade_closure_status.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["docs_artifacts"] = copied
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if copy_to_docs:
        shutil.copy2(summary_path, docs_output_dir / summary_path.name)

    print(f"Research-grade closure status written to {out_dir}")
    print(f"closed_lanes = {summary['closed_lane_count']}/{summary['lane_count']}")
    print(f"research_grade_ready = {summary['research_grade_ready']}")
    return summary


if __name__ == "__main__":
    run_research_grade_closure_status()
