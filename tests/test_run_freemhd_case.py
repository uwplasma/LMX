from pathlib import Path

import pytest

from scripts import run_freemhd_case as runner


pytestmark = pytest.mark.unit


def test_main_reports_missing_docker_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    case_dir = tmp_path / "case"
    bundle_root = tmp_path / "docker"
    case_dir.mkdir()
    bundle_root.mkdir()

    monkeypatch.setattr(runner, "docker_cli_available", lambda: True)
    monkeypatch.setattr(runner, "docker_daemon_available", lambda: True)
    monkeypatch.setattr(runner, "docker_image_available", lambda image: False)

    output = tmp_path / "report.json"
    exit_code = runner.main(
        [
            "--image",
            "missing-image:latest",
            "--case-dir",
            str(case_dir),
            "--bundle-root",
            str(bundle_root),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr().out
    payload = output.read_text()
    assert exit_code == 0
    assert '"status": "docker-image-unavailable"' in captured
    assert '"docker_image_available": false' in payload


def test_main_runs_container_when_image_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    case_dir = tmp_path / "case"
    bundle_root = tmp_path / "docker"
    case_dir.mkdir()
    bundle_root.mkdir()

    monkeypatch.setattr(runner, "docker_cli_available", lambda: True)
    monkeypatch.setattr(runner, "docker_daemon_available", lambda: True)
    monkeypatch.setattr(runner, "docker_image_available", lambda image: True)

    def fake_run(**kwargs):
        return runner.subprocess.CompletedProcess(args=["docker"], returncode=0, stdout="done", stderr="")

    monkeypatch.setattr(runner, "run_freemhd_case", fake_run)

    output = tmp_path / "report.json"
    exit_code = runner.main(
        [
            "--image",
            "available-image:latest",
            "--case-dir",
            str(case_dir),
            "--bundle-root",
            str(bundle_root),
            "--output",
            str(output),
        ]
    )

    captured = capsys.readouterr().out
    payload = output.read_text()
    assert exit_code == 0
    assert '"status": "ok"' in captured
    assert '"docker_image_available": true' in payload
