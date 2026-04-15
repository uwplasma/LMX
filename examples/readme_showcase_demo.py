from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np

from lmx.cases import make_hunt_case
from lmx.example_runner import solve_case_snapshots
from lmx.plotting import write_transient_movies

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from geometry_panel_demo import run_geometry_panel_demo


_FIELD_KEYS = ("u", "phi", "jy", "jz", "lorentz_x")
_SCALAR_KEYS = (
    "time",
    "residual",
    "potential_residual",
    "potential_iterations",
    "face_current_max",
    "emf_max",
    "face_lorentz_max",
    "mean_velocity",
    "applied_forcing",
    "pressure_proxy",
)


def _densify_movie_frames(
    frames: list[dict[str, object]],
    *,
    inbetween_count: int,
) -> list[dict[str, object]]:
    if inbetween_count <= 0 or len(frames) < 2:
        return frames

    dense_frames: list[dict[str, object]] = [frames[0]]
    for left, right in zip(frames, frames[1:], strict=True):
        for substep in range(1, inbetween_count + 1):
            alpha = substep / float(inbetween_count + 1)
            blended: dict[str, object] = {
                "mesh": left["mesh"],
                "fluid_mask": left["fluid_mask"],
                "case": left.get("case", right.get("case")),
            }
            for key in _FIELD_KEYS:
                left_array = np.asarray(left[key], dtype=float)
                right_array = np.asarray(right[key], dtype=float)
                blended[key] = (1.0 - alpha) * left_array + alpha * right_array
            for key in _SCALAR_KEYS:
                left_value = float(left[key])
                right_value = float(right[key])
                blended[key] = (1.0 - alpha) * left_value + alpha * right_value
            dense_frames.append(blended)
        dense_frames.append(right)
    return dense_frames


def run_readme_showcase_demo(
    *,
    out_dir: Path,
    movie_ha: float = 20.0,
    movie_resolution: int = 12,
    movie_frames: int = 16,
    movie_inbetween_frames: int = 10,
    movie_fps: int = 4,
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
        movie_case = make_hunt_case(ha=movie_ha, ny=movie_resolution, nz=movie_resolution, wall_cells=1)
        movie_case = replace(
            movie_case,
            time_stepper=replace(
                movie_case.time_stepper,
                dt=3.75e-4,
                t_final=6.0e-2,
                max_steps=160,
                potential_iterations=80,
            ),
        )
        movie_frames_payload = solve_case_snapshots(movie_case, frame_count=movie_frames)
        movie_frames_payload = _densify_movie_frames(movie_frames_payload, inbetween_count=movie_inbetween_frames)
        include_2d = movie_view in {"both", "2d"}
        include_3d = movie_view in {"both", "3d"}
        movie_outputs = write_transient_movies(
            movie_frames_payload,
            out_dir,
            case_title="LMX Hunt startup showcase",
            fps=movie_fps,
            field_mode="bulk_deviation",
            output_stem="readme_hunt_startup",
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
    parser = argparse.ArgumentParser(description="Generate the retained README showcase media bundle.")
    parser.add_argument("--output", type=Path, default=Path("docs/_static/generated"))
    parser.add_argument("--movie-ha", type=float, default=20.0)
    parser.add_argument("--movie-resolution", type=int, default=12)
    parser.add_argument("--movie-frames", type=int, default=16)
    parser.add_argument("--movie-inbetween-frames", type=int, default=10)
    parser.add_argument("--movie-fps", type=int, default=4)
    parser.add_argument("--movie-view", choices=("both", "2d", "3d"), default="both")
    parser.add_argument("--skip-geometry", action="store_true")
    parser.add_argument("--skip-movie", action="store_true")
    args = parser.parse_args(argv)
    run_readme_showcase_demo(
        out_dir=args.output,
        movie_ha=args.movie_ha,
        movie_resolution=args.movie_resolution,
        movie_frames=args.movie_frames,
        movie_inbetween_frames=args.movie_inbetween_frames,
        movie_fps=args.movie_fps,
        movie_view=args.movie_view,
        include_geometry=not args.skip_geometry,
        include_movie=not args.skip_movie,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
