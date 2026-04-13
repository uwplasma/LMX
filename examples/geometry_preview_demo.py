from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from lmx.cases import make_hartmann_case, make_hunt_case
from lmx.mesh import generate_pipe_ogrid_mesh
from lmx.plotting import write_case_overview_plots, write_geometry_preview_plots
from lmx.solvers import _build_mesh, solve_steady


def build_hartmann_preview_case():
    return make_hartmann_case(ha=20.0, ny=24, nz=18)


def build_hunt_preview_case():
    return make_hunt_case(ha=20.0, ny=12, nz=10, wall_cells=1)


def build_pipe_preview_mesh():
    return generate_pipe_ogrid_mesh(radius=0.5, length=2.0, nx=16, nr=18, ntheta=48)


def build_postprocessing_case(kind: str = "hartmann"):
    if kind == "hartmann":
        case = make_hartmann_case(ha=20.0, ny=12, nz=10)
        return replace(
            case,
            time_stepper=replace(case.time_stepper, max_steps=6, potential_iterations=80, steady_tolerance=1e-6),
        )
    if kind == "hunt":
        case = make_hunt_case(ha=20.0, ny=8, nz=6, wall_cells=1)
        return replace(
            case,
            time_stepper=replace(case.time_stepper, max_steps=4, potential_iterations=60, steady_tolerance=1e-5),
        )
    raise ValueError(f"Unsupported postprocessing case kind {kind!r}")


def write_preview_bundle(*, out_dir: Path, with_post_run: bool = False, post_case_kind: str = "hartmann") -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)

    hartmann = build_hartmann_preview_case()
    hunt = build_hunt_preview_case()
    pipe_mesh = build_pipe_preview_mesh()

    hartmann_preview = write_geometry_preview_plots(
        _build_mesh(hartmann),
        out_dir / "hartmann_geometry",
        case_title="Hartmann rect_duct geometry",
    )
    hunt_preview = write_geometry_preview_plots(
        _build_mesh(hunt),
        out_dir / "hunt_geometry",
        case_title="Hunt layered_duct geometry",
    )
    pipe_preview = write_geometry_preview_plots(
        pipe_mesh,
        out_dir / "pipe_geometry",
        case_title="Pipe O-grid geometry",
    )

    post_outputs: list[str] = []
    if with_post_run:
        post_case = build_postprocessing_case(post_case_kind)
        post_solution = solve_steady(post_case)
        post_paths = write_case_overview_plots(
            post_solution,
            out_dir / f"{post_case_kind}_post",
            case_title=f"{post_case_kind.capitalize()} postprocessing",
        )
        post_outputs = [path.name for path in post_paths]

    summary = {
        "hartmann_geometry": [path.name for path in hartmann_preview],
        "hunt_geometry": [path.name for path in hunt_preview],
        "pipe_geometry": [path.name for path in pipe_preview],
        "with_post_run": with_post_run,
        "post_case_kind": post_case_kind if with_post_run else None,
        "postprocessing": post_outputs,
    }
    (out_dir / "geometry_preview_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Preview LMX geometries and generate matching postprocessing figures.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/examples/geometry_preview"))
    parser.add_argument(
        "--with-post-run",
        action="store_true",
        help="Run a short steady solve after the geometry preview and write matching overview plots.",
    )
    parser.add_argument(
        "--post-case",
        choices=("hartmann", "hunt"),
        default="hartmann",
        help="Choose which short benchmark case to solve when --with-post-run is enabled.",
    )
    args = parser.parse_args(argv)
    write_preview_bundle(out_dir=args.output, with_post_run=args.with_post_run, post_case_kind=args.post_case)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
