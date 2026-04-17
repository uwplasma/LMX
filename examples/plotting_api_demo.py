from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path

from lmx import (
    make_hartmann_case,
    solve_case_snapshots,
    solve_steady,
    write_case_overview_plots,
    write_geometry_preview_plots,
    write_transient_movies,
)


def run_plotting_api_demo(
    *,
    out_dir: Path,
    ha: float = 20.0,
    ny: int = 24,
    nz: int = 24,
    movie_dt: float = 2.0e-5,
    movie_t_final: float = 2.0e-4,
) -> dict[str, object]:
    out_dir.mkdir(parents=True, exist_ok=True)
    case = make_hartmann_case(ha=ha, ny=ny, nz=nz, output_dir=str(out_dir))

    steady = solve_steady(case)
    geometry_paths = write_geometry_preview_plots(steady.mesh, out_dir / "geometry", case_title=case.name)
    overview_paths = write_case_overview_plots(steady, out_dir / "steady", case_title=case.name)

    transient_case = replace(
        case,
        time_stepper=replace(
            case.time_stepper,
            dt=movie_dt,
            t_final=movie_t_final,
            max_steps=max(1, int(movie_t_final / movie_dt)),
        ),
    )
    frames = solve_case_snapshots(transient_case, frame_count=max(1, int(movie_t_final / movie_dt)))
    movie_paths = write_transient_movies(
        frames,
        out_dir / "movies",
        case_title=f"{case.name} plotting API demo",
        output_stem="plotting_api_demo",
        fps=18,
    )

    summary = {
        "case": case.name,
        "geometry": [str(path.name) for path in geometry_paths],
        "steady": [str(path.name) for path in overview_paths],
        "movies": [str(path.name) for path in movie_paths],
    }
    (out_dir / "plotting_api_demo_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Demonstrate the importable LMX plotting APIs.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/examples/plotting_api_demo"))
    parser.add_argument("--ha", type=float, default=20.0)
    parser.add_argument("--ny", type=int, default=24)
    parser.add_argument("--nz", type=int, default=24)
    parser.add_argument("--movie-dt", type=float, default=2.0e-5)
    parser.add_argument("--movie-t-final", type=float, default=2.0e-4)
    args = parser.parse_args(argv)
    run_plotting_api_demo(
        out_dir=args.output,
        ha=args.ha,
        ny=args.ny,
        nz=args.nz,
        movie_dt=args.movie_dt,
        movie_t_final=args.movie_t_final,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
