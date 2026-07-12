from __future__ import annotations

import json
from pathlib import Path

from lmx.mesh import generate_bent_pipe_mesh
from lmx.plotting import write_geometry_preview_plots


OUTPUT_DIR = Path("artifacts/examples/bent_pipe_preview")
TUBE_RADIUS = 0.30
BEND_RADIUS = 1.20
BEND_ANGLE = 1.35
NX = 24
NR = 18
NTHETA = 56


def run_bent_pipe_preview() -> dict[str, object]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mesh = generate_bent_pipe_mesh(
        tube_radius=TUBE_RADIUS,
        bend_radius=BEND_RADIUS,
        bend_angle=BEND_ANGLE,
        nx=NX,
        nr=NR,
        ntheta=NTHETA,
    )
    plots = write_geometry_preview_plots(
        mesh, OUTPUT_DIR, case_title="Bent pipe geometry"
    )
    summary = {
        "case": "bent_pipe_preview",
        "geometry_kind": mesh.geometry,
        "plots": [path.name for path in plots],
        "mesh": {"nx": mesh.nx, "nr": mesh.ny, "ntheta": mesh.nz},
        "tube_radius": TUBE_RADIUS,
        "bend_radius": BEND_RADIUS,
        "bend_angle": BEND_ANGLE,
    }
    (OUTPUT_DIR / "bent_pipe_preview_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


if __name__ == "__main__":
    run_bent_pipe_preview()
