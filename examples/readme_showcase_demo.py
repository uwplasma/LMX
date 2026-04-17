from __future__ import annotations

import argparse
import json
from dataclasses import replace
import math
from pathlib import Path
import sys

from lmx.cases import make_hartmann_case, make_hunt_case
from lmx.example_runner import solve_case_snapshots
from lmx.plotting import write_transient_movies
from lmx.specs import MagneticFieldSpec

EXAMPLES_DIR = Path(__file__).resolve().parent
if str(EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_DIR))

from geometry_panel_demo import run_geometry_panel_demo


def _hartmann_b_from_ha(*, ha: float, hartmann_spacing: float, conductivity: float, density: float, viscosity: float) -> float:
    return ha / (hartmann_spacing * ((conductivity / (density * viscosity)) ** 0.5))


def run_readme_showcase_demo(
    *,
    out_dir: Path,
    movie_case_kind: str = "hunt",
    movie_ha: float = 20.0,
    movie_width: float = 2.0,
    movie_height: float = 2.0,
    movie_ny: int = 49,
    movie_nz: int = 49,
    movie_dt: float = 1.0e-5,
    movie_t_final: float = 2.0e-3,
    movie_fps: int = 12,
    movie_view: str = "both",
    movie_coupling_iterations: int = 6,
    movie_coupling_tolerance: float = 1.0e-6,
    movie_potential_iterations: int = 48,
    movie_wall_cells: int = 6,
    movie_initial_velocity: float = 1.0,
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
        if movie_case_kind == "hartmann":
            movie_case = make_hartmann_case(
                ha=movie_ha,
                width=movie_width,
                height=movie_height,
                ny=movie_ny,
                nz=movie_nz,
            )
            fluid = movie_case.regions[0]
            corrected_bmag = _hartmann_b_from_ha(
                ha=movie_ha,
                hartmann_spacing=0.5 * movie_width,
                conductivity=fluid.conductivity,
                density=fluid.density,
                viscosity=fluid.viscosity,
            )
            movie_case = replace(
                movie_case,
                magnetic_field=MagneticFieldSpec(kind="constant", value=(0.0, corrected_bmag, 0.0)),
            )
            output_stem = "readme_hartmann_startup"
            case_title = "LMX Hartmann startup"
        elif movie_case_kind == "hunt":
            movie_case = make_hunt_case(
                ha=movie_ha,
                width=movie_width,
                height=movie_height,
                ny=movie_ny,
                nz=movie_nz,
                wall_cells=movie_wall_cells,
            )
            output_stem = "readme_hunt_startup"
            case_title = "LMX Hunt startup"
        else:
            raise ValueError(f"Unsupported README movie case {movie_case_kind!r}")
        movie_case = replace(
            movie_case,
            solver=replace(
                movie_case.solver,
                coupling_iterations=movie_coupling_iterations,
                coupling_tolerance=movie_coupling_tolerance,
            ),
            time_stepper=replace(
                movie_case.time_stepper,
                dt=movie_dt,
                t_final=movie_t_final,
                max_steps=movie_steps,
                potential_iterations=movie_potential_iterations,
            ),
            initial_velocity=movie_initial_velocity,
        )
        movie_frames_payload = solve_case_snapshots(movie_case, frame_count=movie_steps)
        include_2d = movie_view in {"both", "2d"}
        include_3d = movie_view in {"both", "3d"}
        movie_outputs = write_transient_movies(
            movie_frames_payload,
            out_dir,
            case_title=case_title,
            fps=movie_fps,
            field_mode="raw",
            output_stem=output_stem,
            include_2d=include_2d,
            include_3d=include_3d,
            symmetry_average_axes=("y", "z") if movie_case_kind in {"hunt", "shercliff", "hartmann"} else (),
        )

    summary = {
        "case": "readme_showcase_demo",
        "movie_case_kind": movie_case_kind,
        "geometry": geometry_summary,
        "movie_outputs": [path.name for path in movie_outputs],
    }
    (out_dir / "readme_showcase_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate the README showcase media bundle.")
    parser.add_argument("--output", type=Path, default=Path("docs/_static/generated"))
    parser.add_argument("--movie-case-kind", choices=("hartmann", "hunt"), default="hunt")
    parser.add_argument("--movie-ha", type=float, default=20.0)
    parser.add_argument("--movie-width", type=float, default=2.0)
    parser.add_argument("--movie-height", type=float, default=2.0)
    parser.add_argument("--movie-ny", type=int, default=49)
    parser.add_argument("--movie-nz", type=int, default=49)
    parser.add_argument("--movie-dt", type=float, default=1.0e-5)
    parser.add_argument("--movie-t-final", type=float, default=2.0e-3)
    parser.add_argument("--movie-fps", type=int, default=12)
    parser.add_argument("--movie-view", choices=("both", "2d", "3d"), default="both")
    parser.add_argument("--movie-coupling-iterations", type=int, default=6)
    parser.add_argument("--movie-coupling-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--movie-potential-iterations", type=int, default=48)
    parser.add_argument("--movie-wall-cells", type=int, default=6)
    parser.add_argument("--movie-initial-velocity", type=float, default=1.0)
    parser.add_argument("--skip-geometry", action="store_true")
    parser.add_argument("--skip-movie", action="store_true")
    args = parser.parse_args(argv)
    run_readme_showcase_demo(
        out_dir=args.output,
        movie_case_kind=args.movie_case_kind,
        movie_ha=args.movie_ha,
        movie_width=args.movie_width,
        movie_height=args.movie_height,
        movie_ny=args.movie_ny,
        movie_nz=args.movie_nz,
        movie_dt=args.movie_dt,
        movie_t_final=args.movie_t_final,
        movie_fps=args.movie_fps,
        movie_view=args.movie_view,
        movie_coupling_iterations=args.movie_coupling_iterations,
        movie_coupling_tolerance=args.movie_coupling_tolerance,
        movie_potential_iterations=args.movie_potential_iterations,
        movie_wall_cells=args.movie_wall_cells,
        movie_initial_velocity=args.movie_initial_velocity,
        include_geometry=not args.skip_geometry,
        include_movie=not args.skip_movie,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
