"""Generate explicit Li/AlN/metal multilayer wall-stack mesh QA artifacts.

This example builds a true ``fluid | AlN | metal`` rectangular cross-section
with faces aligned at material interfaces. It is a geometry and electrical
mesh-quality gate; it does not yet claim a solved multilayer validation against
an external code.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from lmx.wall_study import (
    LithiumMaterial,
    WallStackStudyCase,
    write_li_aln_multilayer_mesh_artifacts,
)


OUTPUT_DIR = Path("studies/li_aln_wall_mhd/results/processed/multilayer_mesh")
FIGURE_DIR = Path("studies/li_aln_wall_mhd/figures")
DOCS_OUTPUT_DIR = Path("docs/_static/generated")
COPY_TO_DOCS = True

FLUID_CELLS_Y = 48
FLUID_CELLS_Z = 48

LITHIUM = LithiumMaterial(
    temperature_c=250.0,
    density=500.0,
    dynamic_viscosity=4.0e-4,
    electrical_conductivity=3.2e6,
)

CASE = WallStackStudyCase(
    name="li_aln_rectangular_multilayer_wall_stack",
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


def run_li_aln_multilayer_mesh_qa(
    *,
    output_dir: Path = OUTPUT_DIR,
    figure_dir: Path = FIGURE_DIR,
    docs_output_dir: Path = DOCS_OUTPUT_DIR,
    copy_to_docs: bool = COPY_TO_DOCS,
) -> dict[str, object]:
    """Write explicit multilayer mesh QA JSON/CSV/PNG artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    docs_output_dir.mkdir(parents=True, exist_ok=True)
    outputs = write_li_aln_multilayer_mesh_artifacts(
        output_dir,
        case=CASE,
        ny=FLUID_CELLS_Y,
        nz=FLUID_CELLS_Z,
    )
    summary_path = output_dir / "li_aln_multilayer_mesh_qa_summary.json"
    figure_path = output_dir / "li_aln_multilayer_mesh_qa.png"
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

    print(f"Li/AlN multilayer mesh QA artifacts written to {output_dir}")
    print(f"interfaces aligned = {summary['qa']['interface_faces_aligned']}")
    print(
        f"ready for current diagnostics = {summary['qa']['ready_for_conservative_current_diagnostics']}"
    )
    return summary


if __name__ == "__main__":
    run_li_aln_multilayer_mesh_qa()
