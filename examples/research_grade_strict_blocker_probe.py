"""Write the final strict research-blocker closure-attempt artifacts.

This example is intentionally lightweight. It records the outcome of the latest
manual closure attempts and audits local external-code inputs; it does not run
the heavy external solvers or promote unmatched candidate data into validation.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from lmx.research_blockers import write_strict_blocker_closure_attempt


OUTPUT_DIR = Path("artifacts/examples/research_grade_strict_blocker_attempt")
DOCS_OUTPUT_DIR = Path("docs/_static/generated")
EXTERNAL_CODES_ROOT = Path("/Users/rogerio/local/tests/lmx_external_codes")
OUTPUT_STEM = "research_grade_strict_blocker_attempt"
COPY_TO_DOCS = True


def run_research_grade_strict_blocker_probe(
    *,
    out_dir: Path = OUTPUT_DIR,
    docs_output_dir: Path = DOCS_OUTPUT_DIR,
    external_codes_root: Path = EXTERNAL_CODES_ROOT,
    copy_to_docs: bool = COPY_TO_DOCS,
) -> dict[str, object]:
    """Write the strict blocker attempt report and copy it to docs."""

    out_dir.mkdir(parents=True, exist_ok=True)
    docs_output_dir.mkdir(parents=True, exist_ok=True)
    outputs = write_strict_blocker_closure_attempt(
        out_dir,
        static_dir=docs_output_dir,
        external_codes_root=external_codes_root,
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

    print(f"Strict blocker attempt written to {out_dir}")
    print(f"research_grade_ready = {summary['research_grade_ready']}")
    print(f"strict_open_lanes = {summary['strict_open_lanes']}")
    return summary


if __name__ == "__main__":
    run_research_grade_strict_blocker_probe()
