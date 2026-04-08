from pathlib import Path

import os

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
    (case_dir / "system").mkdir()
    (case_dir / "system" / "controlDict").write_text("application epotMultiRegionInterFoam;\n")
    (case_dir / "system" / "decomposeParDict").write_text("numberOfSubdomains 95;\n")

    def fake_run(**kwargs):
        assert kwargs["solver"] == "epotMultiRegionInterFoam"
        assert kwargs["platform"] == "linux/amd64"
        assert kwargs["cores"] == 95
        assert kwargs["start_from"] is None
        assert kwargs["log_coupled_iterations"] is False
        return runner.subprocess.CompletedProcess(
            args=["docker"],
            returncode=0,
            stdout="LMX_DIAG outer time=1e-05 oCorr=0 nOuterCorr=1 finalIter=1\n",
            stderr="warn",
        )

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
    assert '"solver": "epotMultiRegionInterFoam"' in payload
    assert '"platform": "linux/amd64"' in payload
    assert '"cores": 95' in payload
    assert '"run_stdout_log": "' in payload
    assert '"run_stderr_log": "' in payload
    assert output.with_suffix(".run.stdout.log").read_text() == "LMX_DIAG outer time=1e-05 oCorr=0 nOuterCorr=1 finalIter=1\n"
    assert output.with_suffix(".run.stderr.log").read_text() == "warn"
    assert '"run_diag_json": "' in payload
    assert '"run_diag_record_count": 1' in payload


def test_run_freemhd_case_uses_bash_entrypoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    bundle_root = tmp_path / "docker"
    case_dir = tmp_path / "case"
    bundle_root.mkdir()
    case_dir.mkdir()
    (bundle_root / "run_freemhd_case.sh").write_text("#!/usr/bin/env bash\n")

    recorded = {}

    def fake_subprocess_run(command, text, capture_output, check):
        recorded["command"] = command
        return runner.subprocess.CompletedProcess(args=command, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_subprocess_run)
    runner.run_freemhd_case(
        image="lmx-freemhd-smoke",
        case_dir=case_dir,
        bundle_root=bundle_root,
        solver="epotMultiRegionInterFoam",
        platform="linux/amd64",
        end_time="1e-3",
        write_interval="1e-3",
        delta_t="1e-4",
        start_from="startTime",
        log_coupled_iterations=True,
    )

    command = recorded["command"]
    assert command[:5] == ["docker", "run", "--rm", "--platform", "linux/amd64"]
    assert "--user" in command
    assert f"{os.getuid()}:{os.getgid()}" in command
    assert "HOME=/tmp" in command
    assert "LMX_END_TIME=1e-3" in command
    assert "LMX_WRITE_INTERVAL=1e-3" in command
    assert "LMX_DELTA_T=1e-4" in command
    assert "LMX_START_FROM=startTime" in command
    assert "LMX_LOG_COUPLED_ITERATIONS=true" in command
    assert "--entrypoint" in command
    assert "/opt/lmx/run_freemhd_case.sh" in command
    assert "epotMultiRegionInterFoam" in command


def test_main_uses_explicit_cores_when_requested(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    case_dir = tmp_path / "case"
    bundle_root = tmp_path / "docker"
    case_dir.mkdir()
    bundle_root.mkdir()
    (case_dir / "system").mkdir()
    (case_dir / "system" / "controlDict").write_text("application epotMultiRegionInterFoam;\n")
    (case_dir / "system" / "decomposeParDict").write_text("numberOfSubdomains 95;\n")

    monkeypatch.setattr(runner, "docker_cli_available", lambda: True)
    monkeypatch.setattr(runner, "docker_daemon_available", lambda: True)
    monkeypatch.setattr(runner, "docker_image_available", lambda image: True)

    def fake_run(**kwargs):
        assert kwargs["cores"] == 8
        assert kwargs["solver"] == "epotMultiRegionInterFoam"
        assert kwargs["end_time"] == "1e-3"
        assert kwargs["write_interval"] == "5e-4"
        assert kwargs["delta_t"] == "1e-4"
        assert kwargs["start_from"] == "startTime"
        assert kwargs["log_coupled_iterations"] is True
        return runner.subprocess.CompletedProcess(
            args=["docker"],
            returncode=0,
            stdout="LMX_DIAG outer time=1e-05 oCorr=0 nOuterCorr=1 finalIter=1\n",
            stderr="",
        )

    monkeypatch.setattr(runner, "run_freemhd_case", fake_run)

    exit_code = runner.main(
        [
            "--image",
            "available-image:latest",
            "--case-dir",
            str(case_dir),
            "--bundle-root",
            str(bundle_root),
            "--cores",
            "8",
            "--end-time",
            "1e-3",
            "--write-interval",
            "5e-4",
            "--delta-t",
            "1e-4",
            "--start-from",
            "startTime",
            "--log-coupled-iterations",
        ]
    )

    captured = capsys.readouterr().out
    assert exit_code == 0
    assert '"cores": 8' in captured
    assert '"end_time": "1e-3"' in captured
    assert '"write_interval": "5e-4"' in captured
    assert '"delta_t": "1e-4"' in captured
    assert '"start_from": "startTime"' in captured
    assert '"log_coupled_iterations": true' in captured


def test_main_writes_empty_diag_json_when_forced_logging_has_no_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    case_dir = tmp_path / "case"
    bundle_root = tmp_path / "docker"
    case_dir.mkdir()
    bundle_root.mkdir()
    (case_dir / "system").mkdir()
    (case_dir / "system" / "controlDict").write_text("application epotMultiRegionInterFoam;\n")

    monkeypatch.setattr(runner, "docker_cli_available", lambda: True)
    monkeypatch.setattr(runner, "docker_daemon_available", lambda: True)
    monkeypatch.setattr(runner, "docker_image_available", lambda image: True)

    def fake_run(**kwargs):
        assert kwargs["log_coupled_iterations"] is True
        return runner.subprocess.CompletedProcess(args=["docker"], returncode=0, stdout="plain log\n", stderr="")

    monkeypatch.setattr(runner, "run_freemhd_case", fake_run)

    output = tmp_path / "forced.json"
    exit_code = runner.main(
        [
            "--image",
            "available-image:latest",
            "--case-dir",
            str(case_dir),
            "--bundle-root",
            str(bundle_root),
            "--log-coupled-iterations",
            "--output",
            str(output),
        ]
    )

    payload = output.read_text()
    assert exit_code == 0
    assert '"run_diag_record_count": 0' in payload
    diag_payload = (output.with_suffix(".run.diag.json")).read_text()
    assert '"records": []' in diag_payload
    assert '"status": "ok"' in capsys.readouterr().out


def test_main_marks_failed_run_with_diag_records_as_partial_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    case_dir = tmp_path / "case"
    bundle_root = tmp_path / "docker"
    case_dir.mkdir()
    bundle_root.mkdir()
    (case_dir / "system").mkdir()
    (case_dir / "system" / "controlDict").write_text("application epotMultiRegionInterFoam;\n")

    monkeypatch.setattr(runner, "docker_cli_available", lambda: True)
    monkeypatch.setattr(runner, "docker_daemon_available", lambda: True)
    monkeypatch.setattr(runner, "docker_image_available", lambda image: True)

    def fake_run(**kwargs):
        return runner.subprocess.CompletedProcess(
            args=["docker"],
            returncode=137,
            stdout=(
                "LMX_DIAG outer time=3e-05 oCorr=0 nOuterCorr=1 finalIter=1\n"
                "LMX_DIAG epot time=3e-05 region=liquid oCorr=0 potEInitialResidual=0.1 "
                "potEFinalResidual=1e-8 potEIterations=6 maxPotE=0.001 maxJ=1 maxJxB=2\n"
            ),
            stderr="killed",
        )

    monkeypatch.setattr(runner, "run_freemhd_case", fake_run)

    output = tmp_path / "partial_failed.json"
    exit_code = runner.main(
        [
            "--image",
            "available-image:latest",
            "--case-dir",
            str(case_dir),
            "--bundle-root",
            str(bundle_root),
            "--log-coupled-iterations",
            "--output",
            str(output),
        ]
    )

    payload = output.read_text()
    assert exit_code == 137
    assert '"status": "partial-failed"' in payload
    assert '"run_diag_record_count": 2' in payload
    assert '"run_diag_last_time": 3e-05' in payload
    assert '"returncode": 137' in payload
    assert '"status": "partial-failed"' in capsys.readouterr().out
