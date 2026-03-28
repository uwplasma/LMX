from pathlib import Path

import pytest

from lmx.freemhd import (
    discover_freemhd_cases,
    discover_freemhd_cases_in_roots,
    dockerfile_base_image,
    freemhd_environment_report,
    freemhd_container_report,
    recommended_freemhd_target,
)


pytestmark = pytest.mark.unit


def test_discover_freemhd_cases_skips_embedded_openfoam_tree(tmp_path: Path):
    embedded = tmp_path / "OpenFOAM-v2206" / "tutorials" / "foo"
    (embedded / "system").mkdir(parents=True)
    (embedded / "constant").mkdir()
    assert discover_freemhd_cases(tmp_path) == []


def test_recommended_freemhd_target_prefers_real_case(tmp_path: Path):
    case_dir = tmp_path / "ShercliffCase"
    (case_dir / "system").mkdir(parents=True)
    (case_dir / "constant").mkdir()
    (case_dir / "0").mkdir()
    target = recommended_freemhd_target(tmp_path)
    assert target is not None
    assert target.kind == "freemhd_case"
    assert target.path == str(case_dir)


def test_discover_freemhd_cases_in_roots_keeps_extra_root_case(tmp_path: Path):
    repo_root = tmp_path / "external" / "FreeMHD"
    repo_root.mkdir(parents=True)
    extra_root = tmp_path / "recovered" / "Hunt"
    case_dir = extra_root / "hunt_Ha20"
    (case_dir / "system").mkdir(parents=True)
    (case_dir / "constant").mkdir()
    (case_dir / "0").mkdir()

    cases = discover_freemhd_cases_in_roots([repo_root, extra_root])
    assert len(cases) == 1
    assert cases[0].path == str(case_dir)
    assert cases[0].source_root == str(extra_root)


def test_recommended_freemhd_target_falls_back_to_openfoam_smoke_case(tmp_path: Path):
    smoke = tmp_path / "OpenFOAM-v2206" / "tutorials" / "electromagnetics" / "mhdFoam" / "hartmann"
    smoke.mkdir(parents=True)
    target = recommended_freemhd_target(tmp_path)
    assert target is not None
    assert target.kind == "openfoam_smoke_case"
    assert target.path == str(smoke)


def test_freemhd_environment_report_includes_recommended_target(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    smoke = tmp_path / "OpenFOAM-v2206" / "tutorials" / "electromagnetics" / "mhdFoam" / "hartmann"
    smoke.mkdir(parents=True)
    monkeypatch.setattr("lmx.freemhd.docker_cli_available", lambda: True)
    monkeypatch.setattr("lmx.freemhd.docker_daemon_available", lambda: False)
    report = freemhd_environment_report(tmp_path)
    assert report["docker_cli_available"] is True
    assert report["docker_daemon_available"] is False
    assert report["recommended_target"]["kind"] == "openfoam_smoke_case"
    assert "docker daemon is not reachable" in report["blockers"]


def test_freemhd_environment_report_includes_extra_case_roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo_root = tmp_path / "external" / "FreeMHD"
    repo_root.mkdir(parents=True)
    extra_root = tmp_path / "recovered" / "Shercliff"
    case_dir = extra_root / "shercliff_Ha20"
    (case_dir / "system").mkdir(parents=True)
    (case_dir / "constant").mkdir()
    (case_dir / "0").mkdir()

    monkeypatch.setattr("lmx.freemhd.docker_cli_available", lambda: True)
    monkeypatch.setattr("lmx.freemhd.docker_daemon_available", lambda: True)

    report = freemhd_environment_report(repo_root, extra_case_roots=[extra_root])

    assert report["discovered_case_count"] == 1
    assert report["searched_case_roots"] == [str(repo_root), str(extra_root)]
    assert report["extra_case_roots"] == [str(extra_root)]
    assert report["discovered_cases"][0]["source_root"] == str(extra_root)
    assert report["recommended_target"]["kind"] == "freemhd_case"
    assert report["recommended_target"]["path"] == str(case_dir)
    assert str(extra_root) in report["recommended_target"]["reason"]
    assert report["blockers"] == []


def test_dockerfile_base_image_parses_first_from(tmp_path: Path):
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "# comment\n"
        "FROM openfoam/openfoam2206-paraview:latest AS base\n"
        "RUN echo ok\n"
    )
    assert dockerfile_base_image(dockerfile) == "openfoam/openfoam2206-paraview:latest"


def test_freemhd_container_report_classifies_local_missing_and_registry_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    bundle_root = tmp_path / "docker"
    bundle_root.mkdir()
    (bundle_root / "Dockerfile").write_text("FROM openfoam/openfoam2206-paraview:latest\n")

    monkeypatch.setattr("lmx.freemhd.docker_cli_available", lambda: True)
    monkeypatch.setattr("lmx.freemhd.docker_daemon_available", lambda: True)

    def fake_local_report(image: str) -> dict[str, object]:
        return {
            "command": ["docker", "image", "inspect", image],
            "status": "failed",
            "returncode": 1,
            "stdout_tail": "[]\n",
            "stderr_tail": f"No such image: {image}\n",
        }

    monkeypatch.setattr("lmx.freemhd.docker_local_image_report", fake_local_report)
    monkeypatch.setattr(
        "lmx.freemhd.docker_registry_image_report",
        lambda image, timeout_seconds=20: {
            "command": ["docker", "manifest", "inspect", image],
            "status": "timeout",
            "returncode": None,
            "stdout_tail": "",
            "stderr_tail": "",
        },
    )

    report = freemhd_container_report(bundle_root=bundle_root, image="lmx-freemhd", timeout_seconds=5)

    assert report["base_image"] == "openfoam/openfoam2206-paraview:latest"
    assert report["local_image_report"]["status"] == "failed"
    assert report["base_image_registry_report"]["status"] == "timeout"
    assert "requested image is not available locally: lmx-freemhd" in report["blockers"]
    assert "base image registry resolution timed out: openfoam/openfoam2206-paraview:latest" in report["blockers"]
