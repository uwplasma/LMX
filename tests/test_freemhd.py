from pathlib import Path

import pytest

from lmx.freemhd import (
    discover_freemhd_cases,
    discover_freemhd_cases_in_roots,
    freemhd_environment_report,
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
