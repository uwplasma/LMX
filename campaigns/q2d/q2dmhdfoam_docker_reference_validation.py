from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

from lmx.external_validation import (
    load_q2dmhdfoam_docker_reference_profile,
    q2dmhdfoam_docker_reference_observables,
    write_q2dmhdfoam_docker_reference_panel,
    write_q2dmhdfoam_profile_observable_table,
)


OUTPUT_DIR = Path("artifacts/examples/q2dmhdfoam_docker_reference_validation")
DOCS_OUTPUT_DIR = Path("docs/_static/generated")
DOCKER_OUTPUT_DIR = Path("artifacts/external/q2dmhdfoam_reference")
DOCKER_IMAGE = "lmx-q2dmhdfoam:fe41"
DOCKER_CONTEXT = Path("docker/q2dmhdfoam")
DOCKER_RANKS = 2
RUN_DOCKER = False
COPY_TO_DOCS = True


def run_q2dmhdfoam_docker_reference_validation(
    *,
    out_dir: Path = OUTPUT_DIR,
    docker_output_dir: Path = DOCKER_OUTPUT_DIR,
    docs_output_dir: Path = DOCS_OUTPUT_DIR,
    run_docker: bool = RUN_DOCKER,
    copy_to_docs: bool = COPY_TO_DOCS,
) -> dict[str, object]:
    """Create a publication artifact from the Docker-rerun Q2DmhdFoam case."""

    out_dir.mkdir(parents=True, exist_ok=True)
    docs_output_dir.mkdir(parents=True, exist_ok=True)
    docker_output_dir.mkdir(parents=True, exist_ok=True)
    if run_docker:
        _run_docker_reference_case(docker_output_dir)

    profile_path = docker_output_dir / "profile.csv"
    summary_path = docker_output_dir / "summary.json"
    if not profile_path.exists() or not summary_path.exists():
        summary = {
            "case": "q2dmhdfoam_docker_reference_validation",
            "status": "docker_reference_outputs_missing",
            "required_files": [str(profile_path), str(summary_path)],
            "build_command": f"docker build --platform linux/amd64 -t {DOCKER_IMAGE} {DOCKER_CONTEXT}",
            "run_command": (
                "docker run --rm --platform linux/amd64 "
                f"-e RANKS={DOCKER_RANKS} -v $PWD/{docker_output_dir}:/output {DOCKER_IMAGE}"
            ),
        }
        (out_dir / "q2dmhdfoam_docker_reference_validation_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n",
            encoding="utf-8",
        )
        return summary

    profile = load_q2dmhdfoam_docker_reference_profile(profile_path, summary_path)
    observables = q2dmhdfoam_docker_reference_observables(profile)
    table_path = write_q2dmhdfoam_profile_observable_table(
        [observables],
        out_dir / "q2dmhdfoam_docker_reference_observables.csv",
    )
    plots = write_q2dmhdfoam_docker_reference_panel(profile, observables, out_dir)

    copied: list[str] = []
    if copy_to_docs:
        for path in [table_path, *plots]:
            target = docs_output_dir / path.name
            shutil.copy2(path, target)
            copied.append(target.name)

    run_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary = {
        "case": "q2dmhdfoam_docker_reference_validation",
        "status": "external_reference_artifacts_written",
        "docker_output_dir": str(docker_output_dir),
        "profile": profile_path.name,
        "run_summary": run_summary,
        "observables": observables,
        "plots": [path.name for path in plots],
        "observable_table": table_path.name,
        "docs_artifacts": copied,
        "notes": (
            "This is a reproducible external Q2DmhdFoam execution gate using "
            "foam-extend 4.1. The rerun validates the external-code path and "
            "exports VTK/profile artifacts; matched turbulent LMX parity remains "
            "a stricter follow-on gate."
        ),
    }
    summary_path_out = out_dir / "q2dmhdfoam_docker_reference_validation_summary.json"
    summary_path_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if copy_to_docs:
        shutil.copy2(summary_path_out, docs_output_dir / summary_path_out.name)
    return summary


def _run_docker_reference_case(docker_output_dir: Path) -> None:
    subprocess.run(
        [
            "docker",
            "build",
            "--platform",
            "linux/amd64",
            "-t",
            DOCKER_IMAGE,
            str(DOCKER_CONTEXT),
        ],
        check=True,
    )
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            "--platform",
            "linux/amd64",
            "-e",
            f"RANKS={DOCKER_RANKS}",
            "-v",
            f"{docker_output_dir.resolve()}:/output",
            DOCKER_IMAGE,
        ],
        check=True,
    )


if __name__ == "__main__":
    run_q2dmhdfoam_docker_reference_validation()
