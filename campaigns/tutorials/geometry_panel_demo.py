from __future__ import annotations

import argparse
import json
from pathlib import Path

from lmx.mesh import (
    generate_bent_pipe_mesh,
    generate_layered_duct_mesh,
    generate_pipe_ogrid_mesh,
    generate_rect_duct_mesh,
)
from lmx.plotting import write_geometry_gallery_plots


def run_geometry_panel_demo(*, out_dir: Path) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)

    rect_mesh = generate_rect_duct_mesh(
        width=1.0, height=1.0, length=2.0, nx=12, ny=24, nz=24
    )
    layered_mesh = generate_layered_duct_mesh(
        width=1.0,
        height=1.0,
        length=2.0,
        nx=12,
        ny=18,
        nz=18,
        wall_thickness=(0.08, 0.08, 0.06, 0.06),
        wall_cells=(2, 2, 1, 1),
    )
    pipe_mesh = generate_pipe_ogrid_mesh(
        radius=0.5, length=2.0, nx=12, nr=18, ntheta=48
    )
    bent_pipe_mesh = generate_bent_pipe_mesh(
        tube_radius=0.28, bend_radius=1.1, bend_angle=1.2, nx=16, nr=16, ntheta=48
    )

    outputs = write_geometry_gallery_plots(
        [
            ("Rectangular duct", rect_mesh, rect_mesh.fluid_mask),
            ("Layered duct", layered_mesh, layered_mesh.fluid_mask),
            ("Mapped pipe O-grid", pipe_mesh, pipe_mesh.fluid_mask),
            ("Bent pipe", bent_pipe_mesh, bent_pipe_mesh.fluid_mask),
        ],
        out_dir,
        title="LMX geometry panel",
    )

    summary = {
        "case": "geometry_panel_demo",
        "plots": [path.name for path in outputs],
        "geometries": {
            "rect_duct": {"nx": rect_mesh.nx, "ny": rect_mesh.ny, "nz": rect_mesh.nz},
            "layered_duct": {
                "nx": layered_mesh.nx,
                "ny": layered_mesh.ny,
                "nz": layered_mesh.nz,
            },
            "pipe_ogrid": {
                "nx": pipe_mesh.nx,
                "nr": pipe_mesh.ny,
                "ntheta": pipe_mesh.nz,
            },
            "bent_pipe": {
                "nx": bent_pipe_mesh.nx,
                "nr": bent_pipe_mesh.ny,
                "ntheta": bent_pipe_mesh.nz,
            },
        },
    }
    (out_dir / "geometry_panel_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Write a compact panel with the current LMX geometries."
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/examples/geometry_panel")
    )
    args = parser.parse_args(argv)
    run_geometry_panel_demo(out_dir=args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
