from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import numpy as np

from lmx import (
    build_q2d_wall_driven_cavity_case,
    compare_q2d_wall_driven_observables,
    load_q2dmhdfoam_vtk_vector_field,
    q2d_wall_driven_cavity_observables,
    q2dmhdfoam_vtk_velocity_observables,
    solve_q2d_wall_driven_cavity,
    write_q2d_wall_driven_comparison_plots,
)


OUTPUT_DIR = Path("artifacts/examples/q2d_lmx_q2dmhdfoam_lid_driven_parity")
DOCS_OUTPUT_DIR = Path("docs/_static/generated")
Q2DMHDFOAM_OUTPUT_DIR = Path("artifacts/external/q2dmhdfoam_lid_driven_smoke")
COPY_TO_DOCS = True

NX = 201
NY = 101
VISCOSITY = 2.27e-7
HARTMANN_FRICTION = 1.7025e-2
RIGHT_WALL_VELOCITY = 0.1
DT = 5.0e-4
FRAME_COUNT = 48
RELATIVE_TOLERANCE = 0.20


def run_q2d_lmx_q2dmhdfoam_lid_driven_parity(
    *,
    out_dir: Path = OUTPUT_DIR,
    docs_output_dir: Path = DOCS_OUTPUT_DIR,
    q2dmhdfoam_output_dir: Path = Q2DMHDFOAM_OUTPUT_DIR,
    copy_to_docs: bool = COPY_TO_DOCS,
) -> dict[str, object]:
    """Run the first matched LMX-vs-Q2DmhdFoam side-wall Q2D comparison."""

    out_dir.mkdir(parents=True, exist_ok=True)
    docs_output_dir.mkdir(parents=True, exist_ok=True)
    vtk_candidates = sorted(Path(q2dmhdfoam_output_dir).glob("VTK/*.vtk"))
    summary_path = Path(q2dmhdfoam_output_dir) / "summary.json"
    if not vtk_candidates:
        summary = {
            "case": "q2d_lmx_q2dmhdfoam_lid_driven_parity",
            "status": "q2dmhdfoam_vtk_outputs_missing",
            "required_glob": str(Path(q2dmhdfoam_output_dir) / "VTK/*.vtk"),
            "run_command": (
                "docker run --rm --platform linux/amd64 "
                "-e CASE_RELATIVE_PATH=run/lidDriven -e RANKS=2 "
                "-e FORCE_END_TIME=1 -e END_TIME=1.0 -e DELTA_T=0.005 "
                "-e WRITE_CONTROL=timeStep -e WRITE_INTERVAL=20 -e EXTRACT_PROFILE=0 "
                f"-v $PWD/{q2dmhdfoam_output_dir}:/output lmx-q2dmhdfoam:fe41"
            ),
        }
        (out_dir / "q2d_lmx_q2dmhdfoam_lid_driven_parity_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        return summary

    run_summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    t_final = float(run_summary.get("final_time", 1.0))
    case = build_q2d_wall_driven_cavity_case(
        nx=NX,
        ny=NY,
        viscosity=VISCOSITY,
        hartmann_friction=HARTMANN_FRICTION,
        right_wall_velocity=RIGHT_WALL_VELOCITY,
        dt=DT,
        t_final=t_final,
        frame_count=FRAME_COUNT,
    )
    solution = solve_q2d_wall_driven_cavity(case)
    lmx_observables = q2d_wall_driven_cavity_observables(case, solution)
    reference_field = load_q2dmhdfoam_vtk_vector_field(vtk_candidates[-1])
    reference_observables = q2dmhdfoam_vtk_velocity_observables(reference_field)
    comparison = compare_q2d_wall_driven_observables(
        lmx_observables,
        reference_observables,
        relative_tolerance=RELATIVE_TOLERANCE,
    )
    reference_x, reference_y, reference_speed = _reference_speed_grid(reference_field)
    plots = write_q2d_wall_driven_comparison_plots(
        case,
        solution,
        comparison,
        out_dir,
        reference_x=reference_x,
        reference_y=reference_y,
        reference_speed_grid=reference_speed,
    )
    comparison_table = _write_comparison_table(comparison, out_dir / "q2d_lmx_q2dmhdfoam_lid_driven_observables.csv")

    copied: list[str] = []
    if copy_to_docs:
        for path in [comparison_table, *plots]:
            target = docs_output_dir / path.name
            shutil.copy2(path, target)
            copied.append(target.name)

    summary = {
        "case": "q2d_lmx_q2dmhdfoam_lid_driven_parity",
        "status": "matched_comparison_written",
        "q2dmhdfoam_output_dir": str(q2dmhdfoam_output_dir),
        "q2dmhdfoam_vtk_file": str(vtk_candidates[-1]),
        "q2dmhdfoam_run_summary": run_summary,
        "lmx_case": {
            "nx": NX,
            "ny": NY,
            "viscosity": VISCOSITY,
            "hartmann_friction": HARTMANN_FRICTION,
            "right_wall_velocity": RIGHT_WALL_VELOCITY,
            "dt": DT,
            "t_final": t_final,
        },
        "lmx_observables": lmx_observables,
        "q2dmhdfoam_observables": reference_observables,
        "comparison": comparison,
        "matched_parity": bool(comparison["matched_parity"]),
        "strict_blocker_closed": bool(comparison["matched_parity"]),
        "plots": [path.name for path in plots],
        "observable_table": comparison_table.name,
        "docs_artifacts": copied,
        "notes": (
            "This is the first matched geometry/forcing LMX-vs-Q2DmhdFoam "
            "side-wall Q2D comparison. Any failed observable remains an "
            "offender for the strict turbulence/parity lane."
        ),
    }
    summary_path_out = out_dir / "q2d_lmx_q2dmhdfoam_lid_driven_parity_summary.json"
    summary_path_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if copy_to_docs:
        shutil.copy2(summary_path_out, docs_output_dir / summary_path_out.name)
    return summary


def _reference_speed_grid(field: dict[str, object]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(field["points"], dtype=float)
    vectors = np.asarray(field["vectors"], dtype=float)
    z_values = np.unique(points[:, 2])
    z_mid = z_values[int(np.argmin(np.abs(z_values - float(np.median(z_values)))))]
    z_mask = np.abs(points[:, 2] - z_mid) <= 1.0e-12
    section = points[z_mask]
    section_vectors = vectors[z_mask]
    x = np.unique(section[:, 0])
    y = np.unique(section[:, 1])
    speed = np.full((y.size, x.size), np.nan, dtype=float)
    ix = np.searchsorted(x, section[:, 0])
    iy = np.searchsorted(y, section[:, 1])
    speed[iy, ix] = np.linalg.norm(section_vectors[:, : min(3, section_vectors.shape[1])], axis=1)
    return x, y, speed


def _write_comparison_table(comparison: dict[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(comparison.get("rows", []))
    columns = (
        "observable",
        "lmx_value",
        "reference_value",
        "absolute_error",
        "relative_error",
        "relative_tolerance",
        "validation_pass",
    )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
    return path


if __name__ == "__main__":
    run_q2d_lmx_q2dmhdfoam_lid_driven_parity()
