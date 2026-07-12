"""Preview a liquid-metal blanket pipe routed around WHAM before simulation.

This script is intentionally a geometry-review driver: all parameters are near
the top of the file, and the output is a PNG/PDF panel plus JSON dimensions.
After the route is approved, the same centerline can be promoted into a
solver-facing mapped-pipe mesh and WHAM field-response study.
"""

from __future__ import annotations

from pathlib import Path
import shutil

from lmx.blanket_geometry import (
    WhamBlanketLoop,
    build_wham_blanket_centerline,
    wham_blanket_clearance_metrics,
    write_wham_blanket_geometry_preview,
)
from lmx.field_models import load_wham_coil_model_script


OUTPUT_DIR = Path("artifacts/examples/wham_blanket_geometry_preview")
DOCS_OUTPUT_DIR = Path("docs/_static/generated")
WHAM_COIL_MODEL_SCRIPT = Path("/Users/rogerio/Downloads/coil_model_WHAM-1.txt")

# Geometry proposal for review. The WHAM mirror axis is z; this first blanket
# loop lies in the z=0 midplane, enters from negative x, wraps around the
# central-cell clearance envelope, and returns on the opposite side.
PIPE_RADIUS = 0.12
BEND_RADIUS = 0.90
ENTRY_LENGTH = 1.35
CENTRAL_CELL_RADIUS = 0.42
Z_OFFSET = 0.0

# Sampling density controls only the preview smoothness. It is not the solver
# mesh resolution.
STRAIGHT_POINTS = 84
BEND_POINTS = 168


def _coil_parameters() -> dict[str, float]:
    if not WHAM_COIL_MODEL_SCRIPT.exists():
        return {
            "coil_separation": 1.96,
            "coil_inner_radius": 0.043,
            "coil_outer_radius": 0.365,
            "coil_axial_thickness": 0.1144,
        }
    parsed = load_wham_coil_model_script(
        WHAM_COIL_MODEL_SCRIPT, radial_loops=12, axial_loops=4
    )
    return {
        "coil_separation": float(parsed["coil_separation"]),
        "coil_inner_radius": float(parsed["inner_radius"]),
        "coil_outer_radius": float(parsed["outer_radius"]),
        "coil_axial_thickness": float(parsed["coil_axial_thickness"]),
    }


coil_parameters = _coil_parameters()
geometry = WhamBlanketLoop(
    pipe_radius=PIPE_RADIUS,
    bend_radius=BEND_RADIUS,
    entry_length=ENTRY_LENGTH,
    central_cell_radius=CENTRAL_CELL_RADIUS,
    z_offset=Z_OFFSET,
    **coil_parameters,
)
centerline = build_wham_blanket_centerline(
    geometry,
    straight_points=STRAIGHT_POINTS,
    bend_points=BEND_POINTS,
)

outputs = write_wham_blanket_geometry_preview(centerline, OUTPUT_DIR, geometry=geometry)
DOCS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
for output in outputs:
    if output.suffix.lower() in {".png", ".pdf", ".json"}:
        shutil.copy2(output, DOCS_OUTPUT_DIR / output.name)

metrics = wham_blanket_clearance_metrics(centerline, geometry)
print(f"WHAM blanket geometry preview written to {OUTPUT_DIR}")
print(f"path_length = {metrics['path_length']:.3f} m")
print(f"tube_to_cell_clearance = {metrics['tube_to_cell_clearance']:.3f} m")
