from __future__ import annotations

import json
from pathlib import Path

from lmx import generate_layered_duct_mesh
from lmx.showcase import (
    write_lm_duct_geometry_setup_figure,
    write_structured_mesh_figure,
)


OUTPUT_DIR = Path("artifacts/examples/straight_duct_geometry_and_mesh")
WIDTH = 0.2
HEIGHT = 0.2
FLUID_NY = 96
FLUID_NZ = 96
WALL_CELLS = 16
WALL_THICKNESS = 0.02
AXIAL_CELLS = 64
DUCT_LENGTH = 1.0

FLUID_CONDUCTIVITY = 1.0e6
CONDUCTING_WALL_CONDUCTIVITY = 1.0e7
INSULATING_WALL_CONDUCTIVITY = 1.0e-6
DENSITY = 1.0e4
VISCOSITY = 1.0e-3


def run_straight_duct_geometry_and_mesh_demo(
    *,
    out_dir: Path = OUTPUT_DIR,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = generate_layered_duct_mesh(
        width=WIDTH,
        height=HEIGHT,
        length=DUCT_LENGTH,
        nx=1,
        ny=FLUID_NY,
        nz=FLUID_NZ,
        wall_thickness=(WALL_THICKNESS, WALL_THICKNESS, WALL_THICKNESS, WALL_THICKNESS),
        wall_cells=(WALL_CELLS, WALL_CELLS, WALL_CELLS, WALL_CELLS),
        target_ha=20.0,
    )

    geometry_outputs = write_lm_duct_geometry_setup_figure(
        out_dir,
        width=WIDTH,
        height=HEIGHT,
        fluid_conductivity=FLUID_CONDUCTIVITY,
        conducting_wall_conductivity=CONDUCTING_WALL_CONDUCTIVITY,
        insulating_wall_conductivity=INSULATING_WALL_CONDUCTIVITY,
        density=DENSITY,
        viscosity=VISCOSITY,
    )
    mesh_outputs = write_structured_mesh_figure(
        mesh,
        out_dir,
        title="Test Case Mesh - Ha = 20",
        nx=AXIAL_CELLS,
        length=DUCT_LENGTH,
    )

    summary = {
        "case": "straight_duct_geometry_and_mesh",
        "geometry_outputs": [path.name for path in geometry_outputs],
        "mesh_outputs": [path.name for path in mesh_outputs],
    }
    (out_dir / "straight_duct_geometry_and_mesh_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


if __name__ == "__main__":
    run_straight_duct_geometry_and_mesh_demo()
