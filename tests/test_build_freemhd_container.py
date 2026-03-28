from pathlib import Path

import pytest

from scripts import build_freemhd_container as builder


pytestmark = pytest.mark.unit


def test_main_reports_daemon_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    bundle_root = tmp_path / "docker"
    bundle_root.mkdir()

    monkeypatch.setattr(builder, "docker_cli_available", lambda: True)
    monkeypatch.setattr(builder, "docker_daemon_available", lambda: False)

    output = tmp_path / "build.json"
    exit_code = builder.main(["--bundle-root", str(bundle_root), "--output", str(output)])

    assert exit_code == 0
    assert '"status": "docker-daemon-unavailable"' in capsys.readouterr().out
    assert '"platform": "linux/amd64"' in output.read_text()


def test_main_runs_build_with_platform(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    bundle_root = tmp_path / "docker"
    bundle_root.mkdir()

    monkeypatch.setattr(builder, "docker_cli_available", lambda: True)
    monkeypatch.setattr(builder, "docker_daemon_available", lambda: True)

    def fake_build(image: str, bundle_root: Path, platform: str):
        assert image == "lmx-freemhd"
        assert platform == "linux/amd64"
        return builder.subprocess.CompletedProcess(args=["docker"], returncode=0, stdout="done", stderr="")

    monkeypatch.setattr(builder, "build_freemhd_container", fake_build)

    output = tmp_path / "build.json"
    exit_code = builder.main(["--bundle-root", str(bundle_root), "--output", str(output)])

    assert exit_code == 0
    assert '"status": "ok"' in capsys.readouterr().out
    assert '"platform": "linux/amd64"' in output.read_text()
