"""Run the Li/AlN wall-stack Phase 3-6 reduced parametric study.

Inputs live at the top so this file can be copied into a project and edited
directly. The outputs are reduced MHD electrical-performance artifacts; they do
not claim AlN/lithium material compatibility, corrosion resistance, coating
adhesion, wetting, irradiation tolerance, or manufacturability.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from lmx.wall_study import (
    LithiumMaterial,
    WallStackStudyCase,
    write_li_aln_phase3_6_artifacts,
)


OUTPUT_DIR = Path("studies/li_aln_wall_mhd/results/processed/phase3_6")
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
    name="li_aln_rectangular_wall_stack_phase3_6",
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

MAGNETIC_FIELDS_T = (1.0, 2.0, 4.0, 6.0)
VELOCITIES_M_S = (0.01, 0.02, 0.03, 0.04)
SUBSTRATE_CONDUCTIVITIES_S_M = {
    "316L": 1.35e6,
    "IN625": 0.80e6,
    "molybdenum": 1.87e7,
}
ALN_CONDUCTIVITIES_S_M = (
    1.0e-10,
    1.0e-9,
    1.0e-8,
    1.0e-7,
    1.0e-6,
    1.0e-5,
    1.0e-4,
    1.0e-3,
)
PINHOLE_FRACTIONS = (0.0, 1.0e-7, 1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1)


def run_li_aln_wall_stack_phase3_6(
    *,
    output_dir: Path = OUTPUT_DIR,
    figure_dir: Path = FIGURE_DIR,
    docs_output_dir: Path = DOCS_OUTPUT_DIR,
    copy_to_docs: bool = COPY_TO_DOCS,
) -> dict[str, object]:
    """Write Phase 3-6 CSV/JSON/PNG artifacts for the Li/AlN wall-stack study."""

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    docs_output_dir.mkdir(parents=True, exist_ok=True)
    outputs = write_li_aln_phase3_6_artifacts(
        output_dir,
        case=CASE,
        magnetic_fields=MAGNETIC_FIELDS_T,
        velocities=VELOCITIES_M_S,
        substrate_conductivities=SUBSTRATE_CONDUCTIVITIES_S_M,
        aln_conductivities=ALN_CONDUCTIVITIES_S_M,
        pinhole_fractions=PINHOLE_FRACTIONS,
    )
    summary_path = output_dir / "li_aln_wall_stack_phase3_6_summary.json"
    figure_path = output_dir / "li_aln_wall_stack_phase3_6.png"
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

    ten_pct = [
        row
        for row in summary["threshold_rows"]
        if abs(float(row["tolerance_fraction"]) - 0.10) < 1.0e-12
        and row["substrate"] == "316L"
    ][0]
    print(f"Li/AlN Phase 3-6 artifacts written to {output_dir}")
    print(
        f"maximum 316L pinhole fraction for 10% deviation = {float(ten_pct['maximum_pinhole_fraction']):.3e}"
    )
    print(
        "Scope: MHD electrical performance only; true multilayer geometry remains a solver-extension lane."
    )
    return summary


if __name__ == "__main__":
    run_li_aln_wall_stack_phase3_6()
