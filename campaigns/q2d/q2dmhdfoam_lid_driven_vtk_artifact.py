from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil

from lmx.external_validation import (
    load_q2dmhdfoam_vtk_vector_field,
    q2dmhdfoam_vtk_velocity_observables,
    write_q2dmhdfoam_vtk_velocity_panel,
)


OUTPUT_DIR = Path("artifacts/examples/q2dmhdfoam_lid_driven_vtk_artifact")
DOCS_OUTPUT_DIR = Path("docs/_static/generated")
DOCKER_OUTPUT_DIR = Path("artifacts/external/q2dmhdfoam_lid_driven_smoke")
VTK_GLOB = "VTK/*.vtk"
COPY_TO_DOCS = True


def run_q2dmhdfoam_lid_driven_vtk_artifact(
    *,
    out_dir: Path = OUTPUT_DIR,
    docker_output_dir: Path = DOCKER_OUTPUT_DIR,
    docs_output_dir: Path = DOCS_OUTPUT_DIR,
    copy_to_docs: bool = COPY_TO_DOCS,
) -> dict[str, object]:
    """Create a field-level artifact from a Q2DmhdFoam generic Docker rerun."""

    out_dir.mkdir(parents=True, exist_ok=True)
    docs_output_dir.mkdir(parents=True, exist_ok=True)
    vtk_candidates = sorted(Path(docker_output_dir).glob(VTK_GLOB))
    summary_path = Path(docker_output_dir) / "summary.json"
    if not vtk_candidates:
        summary = {
            "case": "q2dmhdfoam_lid_driven_vtk_artifact",
            "status": "q2dmhdfoam_vtk_outputs_missing",
            "required_glob": str(Path(docker_output_dir) / VTK_GLOB),
            "run_command": (
                "docker run --rm --platform linux/amd64 "
                "-e CASE_RELATIVE_PATH=run/lidDriven -e RANKS=2 "
                "-e FORCE_END_TIME=1 -e END_TIME=0.1 -e DELTA_T=0.01 "
                "-e WRITE_CONTROL=timeStep -e WRITE_INTERVAL=1 -e EXTRACT_PROFILE=0 "
                f"-v $PWD/{docker_output_dir}:/output lmx-q2dmhdfoam:fe41"
            ),
        }
        (out_dir / "q2dmhdfoam_lid_driven_vtk_artifact_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        return summary

    vtk_path = vtk_candidates[-1]
    field = load_q2dmhdfoam_vtk_vector_field(vtk_path)
    observables = q2dmhdfoam_vtk_velocity_observables(field)
    plots = write_q2dmhdfoam_vtk_velocity_panel(field, observables, out_dir)
    table_path = _write_observable_table(
        observables, out_dir / "q2dmhdfoam_lid_driven_vtk_observables.csv"
    )

    run_summary = (
        json.loads(summary_path.read_text(encoding="utf-8"))
        if summary_path.exists()
        else {}
    )
    copied: list[str] = []
    if copy_to_docs:
        for path in [table_path, *plots]:
            target = docs_output_dir / path.name
            shutil.copy2(path, target)
            copied.append(target.name)

    summary = {
        "case": "q2dmhdfoam_lid_driven_vtk_artifact",
        "status": "external_vtk_artifacts_written",
        "docker_output_dir": str(docker_output_dir),
        "vtk_file": str(vtk_path),
        "run_summary": run_summary,
        "observables": observables,
        "plots": [path.name for path in plots],
        "observable_table": table_path.name,
        "docs_artifacts": copied,
        "matched_lmx_parity": False,
        "notes": (
            "This artifact proves that the generalized Q2DmhdFoam Docker path "
            "can run a non-default case and expose field-level observables. It "
            "is not yet a matched LMX-vs-Q2DmhdFoam turbulence validation."
        ),
    }
    output_summary = out_dir / "q2dmhdfoam_lid_driven_vtk_artifact_summary.json"
    output_summary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if copy_to_docs:
        shutil.copy2(output_summary, docs_output_dir / output_summary.name)
    return summary


def _write_observable_table(observables: dict[str, object], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["observable", "value"])
        for key in sorted(observables):
            writer.writerow([key, observables[key]])
    return path


if __name__ == "__main__":
    run_q2dmhdfoam_lid_driven_vtk_artifact()
