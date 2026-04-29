"""Write the bounded publication-figure manifest for LMX.

This example does not run the heavy external-code or high-resolution campaigns.
It records the manuscript-facing artifacts that are already generated under
``docs/_static/generated`` and writes a machine-readable readiness table. Set
``REFRESH_FAST_FIGURES`` to true when you want this script to refresh the
bounded WHAM blanket pressure/movie artifacts before collecting the manifest.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from lmx.publication import write_publication_figure_manifest


OUTPUT_DIR = Path("artifacts/examples/publication_figure_campaign")
DOCS_OUTPUT_DIR = Path("docs/_static/generated")
REFRESH_FAST_FIGURES = False
COPY_TO_DOCS = True


def run_publication_figure_campaign(
    *,
    out_dir: Path = OUTPUT_DIR,
    docs_output_dir: Path = DOCS_OUTPUT_DIR,
    refresh_fast_figures: bool = REFRESH_FAST_FIGURES,
    copy_to_docs: bool = COPY_TO_DOCS,
) -> dict[str, object]:
    """Collect publication-figure readiness and write JSON/CSV manifests."""

    out_dir.mkdir(parents=True, exist_ok=True)
    docs_output_dir.mkdir(parents=True, exist_ok=True)
    refreshed: list[str] = []
    if refresh_fast_figures:
        from wham_blanket_flow_demo import run_wham_blanket_flow_demo

        run_wham_blanket_flow_demo()
        refreshed.append("examples/wham_blanket_flow_demo.py")

    outputs = write_publication_figure_manifest(out_dir, static_dir=docs_output_dir)
    copied: list[str] = []
    if copy_to_docs:
        for path in outputs:
            target = docs_output_dir / path.name
            shutil.copy2(path, target)
            copied.append(target.name)

    summary = json.loads((out_dir / "publication_figure_campaign_summary.json").read_text(encoding="utf-8"))
    summary["refreshed_generators"] = refreshed
    summary["docs_artifacts"] = copied
    (out_dir / "publication_figure_campaign_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if copy_to_docs:
        shutil.copy2(out_dir / "publication_figure_campaign_summary.json", docs_output_dir / "publication_figure_campaign_summary.json")

    print(f"Publication figure manifest written to {out_dir}")
    print(f"figures_present = {summary['artifact_count']}/{summary['figure_count']}")
    print(f"paper_ready = {summary['paper_ready']}")
    return summary


if __name__ == "__main__":
    run_publication_figure_campaign()
