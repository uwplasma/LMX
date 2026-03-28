from pathlib import Path

import pytest

from lmx.freemhd import discover_freemhd_cases, freemhd_environment_report, recommended_freemhd_target


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
