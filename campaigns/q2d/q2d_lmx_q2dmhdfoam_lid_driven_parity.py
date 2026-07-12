from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

import numpy as np

from lmx.external_validation import (
    load_q2dmhdfoam_lid_driven_cell_field,
    load_q2dmhdfoam_vtk_vector_field,
    q2dmhdfoam_cell_velocity_observables,
    q2dmhdfoam_vtk_velocity_observables,
)
from lmx.q2d import (
    build_q2d_wall_driven_cavity_case,
    compare_q2d_wall_driven_observables,
    q2d_wall_driven_cavity_observables,
    q2d_wall_driven_cavity_observables_on_grid,
    solve_q2d_wall_driven_cavity,
    write_q2d_wall_driven_comparison_plots,
)


OUTPUT_DIR = Path("artifacts/examples/q2d_lmx_q2dmhdfoam_lid_driven_parity")
DOCS_OUTPUT_DIR = Path("docs/_static/generated")
Q2DMHDFOAM_OUTPUT_DIR = Path("artifacts/external/q2dmhdfoam_lid_driven_isothermal")
COPY_TO_DOCS = True

NX = 201
NY = 201
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
                "-e FORCE_END_TIME=1 -e ZERO_THERMAL=1 "
                "-e END_TIME=1.0 -e DELTA_T=0.005 "
                "-e WRITE_CONTROL=timeStep -e WRITE_INTERVAL=20 -e EXTRACT_PROFILE=0 "
                f"-v $PWD/{q2dmhdfoam_output_dir}:/output lmx-q2dmhdfoam:fe41"
            ),
        }
        (out_dir / "q2d_lmx_q2dmhdfoam_lid_driven_parity_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        return summary

    run_summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {}
    )
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
    lmx_node_observables = q2d_wall_driven_cavity_observables(case, solution)
    reference_field = load_q2dmhdfoam_vtk_vector_field(vtk_candidates[-1])
    reference_vtk_observables = q2dmhdfoam_vtk_velocity_observables(reference_field)
    reference_cell_field = None
    reference_cell_observables = None
    lmx_cell_observables = None
    case_dir = Path(q2dmhdfoam_output_dir) / "case"
    if case_dir.exists():
        reference_cell_field = load_q2dmhdfoam_lid_driven_cell_field(case_dir)
        reference_cell_observables = q2dmhdfoam_cell_velocity_observables(
            reference_cell_field
        )
        lmx_cell_observables = q2d_wall_driven_cavity_observables_on_grid(
            case,
            solution,
            x=np.asarray(reference_cell_field["x"], dtype=float),
            y=np.asarray(reference_cell_field["y"], dtype=float),
            y_widths=np.asarray(reference_cell_field["y_widths"], dtype=float),
        )
    lmx_observables = (
        lmx_cell_observables
        if lmx_cell_observables is not None
        else lmx_node_observables
    )
    reference_observables = (
        reference_cell_observables
        if reference_cell_observables is not None
        else reference_vtk_observables
    )
    observable_keys = (
        ("speed_mean", "speed_rms", "vorticity_peak")
        if reference_cell_observables is not None
        else (
            "speed_mean",
            "speed_rms",
            "uy_mean",
            "vorticity_peak",
        )
    )
    comparison = compare_q2d_wall_driven_observables(
        lmx_observables,
        reference_observables,
        relative_tolerance=RELATIVE_TOLERANCE,
        observable_keys=observable_keys,
    )
    if reference_cell_field is not None:
        reference_x = np.asarray(reference_cell_field["x"], dtype=float)
        reference_y = np.asarray(reference_cell_field["y"], dtype=float)
        reference_speed = np.linalg.norm(
            np.asarray(reference_cell_field["vectors"], dtype=float), axis=2
        )
    else:
        reference_x, reference_y, reference_speed = _reference_speed_grid(
            reference_field
        )
    plots = write_q2d_wall_driven_comparison_plots(
        case,
        solution,
        comparison,
        out_dir,
        reference_x=reference_x,
        reference_y=reference_y,
        reference_speed_grid=reference_speed,
    )
    comparison_table = _write_comparison_table(
        comparison, out_dir / "q2d_lmx_q2dmhdfoam_lid_driven_observables.csv"
    )

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
        "lmx_node_observables": lmx_node_observables,
        "q2dmhdfoam_observables": reference_observables,
        "q2dmhdfoam_vtk_observables": reference_vtk_observables,
        "reference_sampling": str(
            reference_observables.get(
                "reference_gate", "q2dmhdfoam_vtk_field_ingestion"
            )
        ),
        "comparison_observable_keys": list(observable_keys),
        "comparison": comparison,
        "matched_parity": bool(comparison["matched_parity"]),
        "strict_blocker_closed": bool(comparison["matched_parity"]),
        "plots": [path.name for path in plots],
        "observable_table": comparison_table.name,
        "docs_artifacts": copied,
        "notes": (
            "This matched geometry/forcing LMX-vs-Q2DmhdFoam side-wall Q2D "
            "comparison uses cell-centered OpenFOAM observables when the "
            "reconstructed case is available. It is a side-wall field gate; "
            "nonlinear turbulent Q2D parity remains a separate lane."
        ),
    }
    summary_path_out = out_dir / "q2d_lmx_q2dmhdfoam_lid_driven_parity_summary.json"
    summary_path_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if copy_to_docs:
        shutil.copy2(summary_path_out, docs_output_dir / summary_path_out.name)
    return summary


def _reference_speed_grid(
    field: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    speed[iy, ix] = np.linalg.norm(
        section_vectors[:, : min(3, section_vectors.shape[1])], axis=1
    )
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
