"""Run explicit Li/AlN/metal multilayer wall-stack limiting-case solves.

The example compares ideal-insulator, intact-AlN, degraded-AlN, and bare-metal
electrical wall models on the same finite-volume mesh family.  It is a bounded
MHD electrical-performance gate with conservative current diagnostics; it is
not a material-compatibility claim and not an external-code parity claim.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from lmx.wall_study import (
    LithiumMaterial,
    WallStackStudyCase,
    write_li_aln_multilayer_solve_artifacts,
)


OUTPUT_DIR = Path("studies/li_aln_wall_mhd/results/processed/multilayer_solve")
FIGURE_DIR = Path("studies/li_aln_wall_mhd/figures")
DOCS_OUTPUT_DIR = Path("docs/_static/generated")
COPY_TO_DOCS = True

WALL_MODELS = ("ideal_insulator", "intact_aln", "degraded_aln", "bare_metal")
FLUID_CELLS_Y = 18
FLUID_CELLS_Z = 18
MAGNETIC_FIELD_T = 5.0e-2
MEAN_VELOCITY_M_S = 1.0e-2
DT_S = 1.0e-3
T_FINAL_S = 1.2e-2
MAX_STEPS = 12
POTENTIAL_ITERATIONS = 80

LITHIUM = LithiumMaterial(
    temperature_c=250.0,
    density=500.0,
    dynamic_viscosity=4.0e-4,
    electrical_conductivity=3.2e6,
)

CASE = WallStackStudyCase(
    name="li_aln_rectangular_multilayer_wall_stack",
    length_scale=0.05,
    velocity=MEAN_VELOCITY_M_S,
    magnetic_field=MAGNETIC_FIELD_T,
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


def run_li_aln_multilayer_solve(
    *,
    output_dir: Path = OUTPUT_DIR,
    figure_dir: Path = FIGURE_DIR,
    docs_output_dir: Path = DOCS_OUTPUT_DIR,
    copy_to_docs: bool = COPY_TO_DOCS,
) -> dict[str, object]:
    """Write solved multilayer limiting-case JSON/CSV/PNG artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    docs_output_dir.mkdir(parents=True, exist_ok=True)
    outputs = write_li_aln_multilayer_solve_artifacts(
        output_dir,
        case=CASE,
        wall_models=WALL_MODELS,
        ny=FLUID_CELLS_Y,
        nz=FLUID_CELLS_Z,
        magnetic_field=MAGNETIC_FIELD_T,
        velocity=MEAN_VELOCITY_M_S,
        dt=DT_S,
        t_final=T_FINAL_S,
        max_steps=MAX_STEPS,
        potential_iterations=POTENTIAL_ITERATIONS,
    )
    summary_path = output_dir / "li_aln_multilayer_solve_summary.json"
    figure_path = output_dir / "li_aln_multilayer_solve.png"
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

    print(f"Li/AlN multilayer solve artifacts written to {output_dir}")
    print(f"charge balance pass = {summary['qa']['charge_balance_pass']}")
    print(f"interface current pass = {summary['qa']['interface_current_bounded_pass']}")
    return summary


if __name__ == "__main__":
    run_li_aln_multilayer_solve()
