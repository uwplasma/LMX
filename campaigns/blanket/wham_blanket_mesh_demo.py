"""Generate the mapped pipe mesh for the approved WHAM blanket route."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from lmx.blanket_geometry import (
    WhamBlanketLoop,
    build_wham_blanket_centerline,
    write_centerline_pipe_mesh_preview,
)
from lmx.mesh import centerline_pipe_mesh_quality_metrics, generate_centerline_pipe_mesh
from lmx.field_models import load_wham_coil_model_script
from lmx.io import write_vtu


OUTPUT_DIR = Path("artifacts/examples/wham_blanket_mesh")
DOCS_OUTPUT_DIR = Path("docs/_static/generated")
WHAM_COIL_MODEL_SCRIPT = Path("/Users/rogerio/Downloads/coil_model_WHAM-1.txt")

PIPE_RADIUS = 0.12
BEND_RADIUS = 0.90
ENTRY_LENGTH = 1.35
CENTRAL_CELL_RADIUS = 0.42

# These are mesh-preview settings, not final solver-resolution choices.
NX_STATIONS = 64
NR = 18
NTHETA = 48


def _coil_geometry() -> dict[str, float]:
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


def run_wham_blanket_mesh_demo() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    geometry = WhamBlanketLoop(
        pipe_radius=PIPE_RADIUS,
        bend_radius=BEND_RADIUS,
        entry_length=ENTRY_LENGTH,
        central_cell_radius=CENTRAL_CELL_RADIUS,
        **_coil_geometry(),
    )
    centerline = build_wham_blanket_centerline(
        geometry, straight_points=96, bend_points=192
    )
    mesh = generate_centerline_pipe_mesh(
        centerline,
        tube_radius=PIPE_RADIUS,
        nx=NX_STATIONS,
        nr=NR,
        ntheta=NTHETA,
        geometry="wham_blanket_centerline_pipe",
    )
    metrics = centerline_pipe_mesh_quality_metrics(mesh)
    vtk_path = write_vtu(mesh, OUTPUT_DIR, name="wham_blanket_centerline_pipe_mesh")
    plot_outputs = write_centerline_pipe_mesh_preview(
        mesh,
        OUTPUT_DIR,
        filename_stem="wham_blanket_mesh_preview",
        title="WHAM blanket mapped centerline-pipe mesh",
    )
    summary = {
        "case": "wham_blanket_mesh",
        "geometry": geometry.__dict__,
        "mesh": {
            "nx_stations": NX_STATIONS,
            "nr": NR,
            "ntheta": NTHETA,
            "vtu": vtk_path.name,
        },
        "metrics": metrics,
        "plots": [path.name for path in plot_outputs],
        "notes": (
            "This is the mapped geometry/mesh handoff for the approved WHAM "
            "blanket loop. It is ready for ParaView inspection and future "
            "curved-pipe MHD operator work; it is not a solve artifact."
        ),
    }
    summary_path = OUTPUT_DIR / "wham_blanket_mesh_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for output in [*plot_outputs, summary_path]:
        if output.suffix.lower() in {".png", ".json"}:
            shutil.copy2(output, DOCS_OUTPUT_DIR / output.name)
    print(f"WHAM blanket mapped pipe mesh written to {OUTPUT_DIR}")
    print(f"cells = {metrics['cell_count']}")
    print(f"validation_pass = {metrics['validation_pass']}")
    return summary


if __name__ == "__main__":
    run_wham_blanket_mesh_demo()
