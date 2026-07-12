"""Run the Li/AlN wall-stack Phase 0-2 reduced study.

This example keeps all case inputs at the top so users can adapt it directly.
The outputs are reduced MHD electrical-performance artifacts: they do not claim
AlN/lithium material compatibility, corrosion resistance, coating adhesion, or
printability.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from lmx.wall_study import (
    LithiumMaterial,
    WallStackStudyCase,
    write_li_aln_phase0_2_artifacts,
)


OUTPUT_DIR = Path("studies/li_aln_wall_mhd/results/processed/phase0_2")
FIGURE_DIR = Path("studies/li_aln_wall_mhd/figures")
DOCS_OUTPUT_DIR = Path("docs/_static/generated")
COPY_TO_DOCS = True

LITHIUM = LithiumMaterial(
    temperature_c=250.0,
    density=500.0,
    dynamic_viscosity=4.0e-4,
    electrical_conductivity=3.2e6,
)

CASE = WallStackStudyCase(
    name="li_aln_rectangular_wall_stack_phase0_2",
    length_scale=0.05,
    velocity=0.04,
    magnetic_field=2.0,
    lithium=LITHIUM,
    aln_thickness=2.0e-4,
    aln_cells=4,
    metal_name="316L",
    metal_conductivity=1.35e6,
    metal_thickness=1.0e-3,
    metal_cells=8,
    intact_aln_conductivity=1.0e-8,
    degraded_aln_conductivity=1.0e-3,
)

CONDUCTANCE_RATIOS = (
    0.0,
    1.0e-10,
    1.0e-9,
    1.0e-8,
    1.0e-7,
    1.0e-6,
    1.0e-5,
    1.0e-4,
    1.0e-3,
    1.0e-2,
    1.0e-1,
    1.0,
    10.0,
)
PINHOLE_FRACTIONS = (0.0, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0)


def run_li_aln_wall_stack_phase0_2(
    *,
    output_dir: Path = OUTPUT_DIR,
    figure_dir: Path = FIGURE_DIR,
    docs_output_dir: Path = DOCS_OUTPUT_DIR,
    copy_to_docs: bool = COPY_TO_DOCS,
) -> dict[str, object]:
    """Write Phase 0-2 CSV/JSON/PNG artifacts for the Li/AlN wall-stack study."""

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    docs_output_dir.mkdir(parents=True, exist_ok=True)
    outputs = write_li_aln_phase0_2_artifacts(
        output_dir,
        case=CASE,
        conductance_ratios=CONDUCTANCE_RATIOS,
        pinhole_fractions=PINHOLE_FRACTIONS,
    )
    summary_path = output_dir / "li_aln_wall_stack_phase0_2_summary.json"
    figure_path = output_dir / "li_aln_wall_stack_phase0_2.png"
    figure_copy = figure_dir / figure_path.name
    shutil.copy2(figure_path, figure_copy)

    copied: list[str] = []
    if copy_to_docs:
        for path in [summary_path, figure_path]:
            target = docs_output_dir / path.name
            shutil.copy2(path, target)
            copied.append(target.name)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["outputs"] = [str(path) for path in outputs]
    summary["figure_copy"] = str(figure_copy)
    summary["docs_artifacts"] = copied
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if copy_to_docs:
        shutil.copy2(summary_path, docs_output_dir / summary_path.name)

    print(f"Li/AlN Phase 0-2 artifacts written to {output_dir}")
    print(f"Ha = {summary['unit_audit']['hartmann_number']:.3g}")
    print(f"Rm = {summary['unit_audit']['magnetic_reynolds_number']:.3g}")
    return summary


if __name__ == "__main__":
    run_li_aln_wall_stack_phase0_2()
