from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

from lmx.cases import make_hunt_case
from lmx.example_runner import solve_case_snapshots
from lmx.plotting import write_transient_movies

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from geometry_panel_demo import run_geometry_panel_demo
from extruded_paper_figures import run_extruded_paper_figures


def run_readme_showcase_demo(
    *,
    out_dir: Path,
    movie_ha: float = 20.0,
    movie_resolution: int = 18,
    movie_frames: int = 8,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)

    geometry_summary = run_geometry_panel_demo(out_dir=out_dir)

    movie_case = make_hunt_case(ha=movie_ha, ny=movie_resolution, nz=movie_resolution, wall_cells=1)
    movie_case = replace(
        movie_case,
        time_stepper=replace(
            movie_case.time_stepper,
            dt=1.0e-3,
            t_final=1.0e-1,
            max_steps=12,
            potential_iterations=80,
        ),
    )
    movie_frames_payload = solve_case_snapshots(movie_case, frame_count=movie_frames)
    movie_outputs = write_transient_movies(
        movie_frames_payload,
        out_dir,
        case_title="LMX Hunt startup showcase",
        fps=6,
        field_mode="bulk_deviation",
        output_stem="readme_hunt_startup",
    )

    paper_summary = run_extruded_paper_figures(
        out_dir=out_dir,
        ha_peak=20.0,
        ny=10,
        nz=10,
        nx_stations=7,
    )

    summary = {
        "case": "readme_showcase_demo",
        "geometry": geometry_summary,
        "movie_outputs": [path.name for path in movie_outputs],
        "paper_figures": paper_summary["plots"],
    }
    (out_dir / "readme_showcase_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the retained README showcase media bundle.")
    parser.add_argument("--output", type=Path, default=Path("docs/_static/generated"))
    parser.add_argument("--movie-ha", type=float, default=20.0)
    parser.add_argument("--movie-resolution", type=int, default=18)
    parser.add_argument("--movie-frames", type=int, default=8)
    args = parser.parse_args(argv)
    run_readme_showcase_demo(
        out_dir=args.output,
        movie_ha=args.movie_ha,
        movie_resolution=args.movie_resolution,
        movie_frames=args.movie_frames,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
