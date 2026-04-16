from __future__ import annotations

import argparse
import json
from dataclasses import replace
import math
from pathlib import Path
import sys

from lmx.cases import make_hartmann_case
from lmx.example_runner import solve_case_snapshots
from lmx.plotting import write_transient_movies

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from geometry_panel_demo import run_geometry_panel_demo


def run_readme_showcase_demo(
    *,
    out_dir: Path,
    movie_ha: float = 20.0,
    movie_resolution: int = 48,
    movie_dt: float = 2.0e-5,
    movie_t_final: float = 2.0e-3,
    movie_fps: int = 28,
    movie_view: str = "both",
    include_geometry: bool = True,
    include_movie: bool = True,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)

    geometry_summary: dict[str, object] | None = None
    if include_geometry:
        geometry_summary = run_geometry_panel_demo(out_dir=out_dir)

    movie_outputs: list[Path] = []
    if include_movie:
        movie_steps = max(1, int(math.floor(movie_t_final / movie_dt)))
        movie_case = make_hartmann_case(ha=movie_ha, ny=movie_resolution, nz=movie_resolution)
        movie_case = replace(
            movie_case,
            solver=replace(
                movie_case.solver,
                coupling_iterations=8,
                coupling_tolerance=2.5e-4,
            ),
            time_stepper=replace(
                movie_case.time_stepper,
                dt=movie_dt,
                t_final=movie_t_final,
                max_steps=movie_steps,
                potential_iterations=48,
            ),
        )
        movie_frames_payload = solve_case_snapshots(movie_case, frame_count=movie_steps)
        include_2d = movie_view in {"both", "2d"}
        include_3d = movie_view in {"both", "3d"}
        movie_outputs = write_transient_movies(
            movie_frames_payload,
            out_dir,
            case_title="LMX Hartmann startup",
            fps=movie_fps,
            field_mode="raw",
            output_stem="readme_hartmann_startup",
            include_2d=include_2d,
            include_3d=include_3d,
        )

    summary = {
        "case": "readme_showcase_demo",
        "geometry": geometry_summary,
        "movie_outputs": [path.name for path in movie_outputs],
    }
    (out_dir / "readme_showcase_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the README showcase media bundle.")
    parser.add_argument("--output", type=Path, default=Path("docs/_static/generated"))
    parser.add_argument("--movie-ha", type=float, default=20.0)
    parser.add_argument("--movie-resolution", type=int, default=48)
    parser.add_argument("--movie-dt", type=float, default=2.0e-5)
    parser.add_argument("--movie-t-final", type=float, default=2.0e-3)
    parser.add_argument("--movie-fps", type=int, default=28)
    parser.add_argument("--movie-view", choices=("both", "2d", "3d"), default="both")
    parser.add_argument("--skip-geometry", action="store_true")
    parser.add_argument("--skip-movie", action="store_true")
    args = parser.parse_args(argv)
    run_readme_showcase_demo(
        out_dir=args.output,
        movie_ha=args.movie_ha,
        movie_resolution=args.movie_resolution,
        movie_dt=args.movie_dt,
        movie_t_final=args.movie_t_final,
        movie_fps=args.movie_fps,
        movie_view=args.movie_view,
        include_geometry=not args.skip_geometry,
        include_movie=not args.skip_movie,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
