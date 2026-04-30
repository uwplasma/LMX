"""Generate the publication-facing strict closure dashboard."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from lmx.research_figures import write_research_grade_closure_dashboard


OUTPUT_DIR = Path("artifacts/examples/research_grade_closure_dashboard")
DOCS_OUTPUT_DIR = Path("docs/_static/generated")
Q2D_SIDEWALL_SUMMARY = DOCS_OUTPUT_DIR / "q2d_lmx_q2dmhdfoam_lid_driven_parity_summary.json"
MAGNETIC_STRICT_SUMMARY = DOCS_OUTPUT_DIR / "magnetic_obstacle_votyakov_strict_attempt_summary.json"
DEAN_STRICT_SUMMARY = DOCS_OUTPUT_DIR / "dean_vortex_bayat_rezai_strict_attempt_summary.json"
CLOSURE_STATUS = DOCS_OUTPUT_DIR / "research_grade_closure_status.json"
OUTPUT_STEM = "research_grade_closure_dashboard"
COPY_TO_DOCS = True


def run_research_grade_closure_dashboard(
    *,
    out_dir: Path = OUTPUT_DIR,
    docs_output_dir: Path = DOCS_OUTPUT_DIR,
    copy_to_docs: bool = COPY_TO_DOCS,
) -> dict[str, object]:
    """Write the current strict research-lane dashboard and docs copies."""

    out_dir.mkdir(parents=True, exist_ok=True)
    docs_output_dir.mkdir(parents=True, exist_ok=True)
    outputs = write_research_grade_closure_dashboard(
        out_dir,
        q2d_sidewall_summary_path=Q2D_SIDEWALL_SUMMARY,
        magnetic_strict_summary_path=MAGNETIC_STRICT_SUMMARY,
        dean_strict_summary_path=DEAN_STRICT_SUMMARY,
        closure_status_path=CLOSURE_STATUS,
        output_stem=OUTPUT_STEM,
    )
    copied: list[str] = []
    if copy_to_docs:
        for path in outputs:
            target = docs_output_dir / path.name
            shutil.copy2(path, target)
            copied.append(target.name)

    summary_path = out_dir / f"{OUTPUT_STEM}_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["docs_artifacts"] = copied
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if copy_to_docs:
        shutil.copy2(summary_path, docs_output_dir / summary_path.name)

    print(f"Research-grade closure dashboard written to {out_dir}")
    print(f"strict lanes closed = {summary['closed_lane_count']}/{summary['lane_count']}")
    print(f"research_grade_ready = {summary['research_grade_ready']}")
    return summary


if __name__ == "__main__":
    run_research_grade_closure_dashboard()
