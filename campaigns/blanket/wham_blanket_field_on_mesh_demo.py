"""Sample the WHAM mirror field on the approved blanket mapped-pipe mesh."""

from __future__ import annotations

import json
from pathlib import Path
import shutil

from lmx.blanket_geometry import WhamBlanketLoop, build_wham_blanket_centerline
from lmx.centerline_fields import (
    centerline_field_quality_metrics,
    sample_wham_field_on_centerline_pipe_mesh,
    write_centerline_field_preview,
)
from lmx.mesh import generate_centerline_pipe_mesh
from lmx.field_models import load_wham_coil_model_script


OUTPUT_DIR = Path("artifacts/examples/wham_blanket_field_on_mesh")
DOCS_OUTPUT_DIR = Path("docs/_static/generated")
WHAM_COIL_MODEL_SCRIPT = Path("/Users/rogerio/Downloads/coil_model_WHAM-1.txt")

PIPE_RADIUS = 0.12
BEND_RADIUS = 0.90
ENTRY_LENGTH = 1.35
CENTRAL_CELL_RADIUS = 0.42

# Mesh and field settings match the current README blanket mesh handoff.
NX_STATIONS = 64
NR = 18
NTHETA = 48
RADIAL_LOOPS = 12
AXIAL_LOOPS = 4
FIELD_SCALE = 8.0


def _coil_parameters() -> dict[str, float | int]:
    if not WHAM_COIL_MODEL_SCRIPT.exists():
        return {
            "coil_separation": 1.96,
            "inner_radius": 0.043,
            "outer_radius": 0.365,
            "coil_axial_thickness": 0.1144,
            "radial_loops": RADIAL_LOOPS,
            "axial_loops": AXIAL_LOOPS,
            "current_scale": 100323.62459546926,
        }
    parsed = load_wham_coil_model_script(
        WHAM_COIL_MODEL_SCRIPT,
        radial_loops=RADIAL_LOOPS,
        axial_loops=AXIAL_LOOPS,
    )
    return {
        "coil_separation": float(parsed["coil_separation"]),
        "inner_radius": float(parsed["inner_radius"]),
        "outer_radius": float(parsed["outer_radius"]),
        "coil_axial_thickness": float(parsed["coil_axial_thickness"]),
        "radial_loops": int(parsed["radial_loops"]),
        "axial_loops": int(parsed["axial_loops"]),
        "current_scale": float(parsed["current_scale"]),
    }


def run_wham_blanket_field_on_mesh_demo() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    coil_parameters = _coil_parameters()
    geometry = WhamBlanketLoop(
        pipe_radius=PIPE_RADIUS,
        bend_radius=BEND_RADIUS,
        entry_length=ENTRY_LENGTH,
        central_cell_radius=CENTRAL_CELL_RADIUS,
        coil_separation=float(coil_parameters["coil_separation"]),
        coil_inner_radius=float(coil_parameters["inner_radius"]),
        coil_outer_radius=float(coil_parameters["outer_radius"]),
        coil_axial_thickness=float(coil_parameters["coil_axial_thickness"]),
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
    sample = sample_wham_field_on_centerline_pipe_mesh(
        mesh,
        coil_parameters=coil_parameters,
        field_scale=FIELD_SCALE,
    )
    plot_outputs = write_centerline_field_preview(
        sample,
        OUTPUT_DIR,
        filename_stem="wham_blanket_field_on_mesh",
        title="WHAM field sampled on mapped blanket pipe mesh",
    )
    metrics = centerline_field_quality_metrics(sample)
    summary = {
        "case": "wham_blanket_field_on_mesh",
        "geometry": geometry.__dict__,
        "mesh": {
            "nx_stations": NX_STATIONS,
            "nr": NR,
            "ntheta": NTHETA,
        },
        "field": {
            "field_scale": FIELD_SCALE,
            "coil_parameters": coil_parameters,
        },
        "metrics": metrics,
        "artifacts": [path.name for path in plot_outputs],
        "notes": (
            "Solver-facing handoff: the global WHAM-like magnetic field is "
            "sampled on the mapped pipe mesh and projected into local "
            "streamwise/transverse components for future conservative phi/J "
            "assembly."
        ),
    }
    summary_path = OUTPUT_DIR / "wham_blanket_field_on_mesh_demo_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    for output in [*plot_outputs, summary_path]:
        if output.suffix.lower() in {".png", ".json", ".csv"}:
            shutil.copy2(output, DOCS_OUTPUT_DIR / output.name)
    print(f"WHAM blanket field-on-mesh artifacts written to {OUTPUT_DIR}")
    print(f"peak_centerline_B_perp = {metrics['peak_centerline_b_perp']:.3e} T")
    print(
        f"max_cross_section_relative_B_span = {metrics['max_cross_section_relative_b_span']:.3e}"
    )
    print(f"validation_pass = {metrics['validation_pass']}")
    return summary


if __name__ == "__main__":
    run_wham_blanket_field_on_mesh_demo()
