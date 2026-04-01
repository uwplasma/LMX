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

    def fake_build(image: str, bundle_root: Path, platform: str, local_freemhd_root=None):
        assert image == "lmx-freemhd"
        assert platform == "linux/amd64"
        assert local_freemhd_root is None
        return builder.subprocess.CompletedProcess(args=["docker"], returncode=0, stdout="done", stderr="")

    monkeypatch.setattr(builder, "build_freemhd_container", fake_build)

    output = tmp_path / "build.json"
    exit_code = builder.main(["--bundle-root", str(bundle_root), "--output", str(output)])

    assert exit_code == 0
    assert '"status": "ok"' in capsys.readouterr().out
    assert '"platform": "linux/amd64"' in output.read_text()


def test_main_reports_cli_unavailable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    bundle_root = tmp_path / "docker"
    bundle_root.mkdir()

    monkeypatch.setattr(builder, "docker_cli_available", lambda: False)

    output = tmp_path / "build.json"
    exit_code = builder.main(["--bundle-root", str(bundle_root), "--output", str(output)])

    assert exit_code == 0
    assert '"status": "docker-cli-unavailable"' in capsys.readouterr().out
    assert '"docker_available": false' in output.read_text()


def test_prepare_build_context_stages_minimal_local_freemhd_tree(tmp_path: Path):
    bundle_root = tmp_path / "docker"
    bundle_root.mkdir()
    (bundle_root / "Dockerfile").write_text("FROM test\nCOPY FreeMHD/ /opt/FreeMHD/\n")
    (bundle_root / "FreeMHD").mkdir()
    (bundle_root / "FreeMHD" / ".gitkeep").write_text("")

    local_root = tmp_path / "external" / "FreeMHD"
    (local_root / "MHD_Solvers" / "solvers").mkdir(parents=True)
    (local_root / "MHD_Solvers" / "solvers" / "marker.txt").write_text("ok")
    (local_root / "README.md").write_text("local")

    temp_context = builder.prepare_build_context(bundle_root, local_root)
    assert temp_context is not None
    try:
        context_root = Path(temp_context.name)
        assert (context_root / "Dockerfile").exists()
        assert (context_root / "FreeMHD" / "MHD_Solvers" / "solvers" / "marker.txt").read_text() == "ok"
        assert (context_root / "FreeMHD" / "README.md").read_text() == "local"
    finally:
        temp_context.cleanup()


def test_copy_local_freemhd_minimal_requires_mhd_solvers(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="MHD_Solvers"):
        builder._copy_local_freemhd_minimal(tmp_path / "missing", tmp_path / "out")


def test_build_freemhd_container_uses_buildx_load(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    calls = {}

    def fake_run(command, text, capture_output, check):
        calls["command"] = command
        return builder.subprocess.CompletedProcess(args=command, returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(builder.subprocess, "run", fake_run)

    result = builder.build_freemhd_container("lmx-freemhd", tmp_path)

    assert result.returncode == 0
    assert calls["command"][:4] == ["docker", "buildx", "build", "--load"]
