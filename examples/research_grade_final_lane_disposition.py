"""Generate the final strict-lane disposition artifact."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from lmx.research_closure import write_research_grade_final_disposition


OUTPUT_DIR = Path("artifacts/examples/research_grade_final_disposition")
DOCS_OUTPUT_DIR = Path("docs/_static/generated")
OUTPUT_STEM = "research_grade_final_disposition"
COPY_TO_DOCS = True


def run_research_grade_final_lane_disposition(
    *,
    out_dir: Path = OUTPUT_DIR,
    docs_output_dir: Path = DOCS_OUTPUT_DIR,
    copy_to_docs: bool = COPY_TO_DOCS,
) -> dict[str, object]:
    """Write final strict-lane JSON/CSV/PNG artifacts and docs copies."""

    out_dir.mkdir(parents=True, exist_ok=True)
    docs_output_dir.mkdir(parents=True, exist_ok=True)
    outputs = write_research_grade_final_disposition(
        out_dir,
        static_dir=docs_output_dir,
        filename_stem=OUTPUT_STEM,
    )
    copied: list[str] = []
    if copy_to_docs:
        for path in outputs:
            target = docs_output_dir / path.name
            shutil.copy2(path, target)
            copied.append(target.name)

    summary_path = out_dir / f"{OUTPUT_STEM}.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["docs_artifacts"] = copied
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if copy_to_docs:
        shutil.copy2(summary_path, docs_output_dir / summary_path.name)

    print(f"Research-grade final disposition written to {out_dir}")
    print(f"decision = {summary['final_push_decision']}")
    return summary


if __name__ == "__main__":
    run_research_grade_final_lane_disposition()
