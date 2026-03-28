from pathlib import Path

import pytest

from scripts import probe_freemhd_container as probe


pytestmark = pytest.mark.unit


def test_probe_freemhd_container_writes_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    payload = {
        "bundle_root": str(tmp_path / "docker"),
        "image": "lmx-freemhd",
        "docker_cli_available": True,
        "docker_daemon_available": True,
        "dockerfile_exists": True,
        "dockerfile_path": str(tmp_path / "docker" / "Dockerfile"),
        "base_image": "openfoam/openfoam2206-paraview:latest",
        "local_image_report": {"status": "failed"},
        "base_image_local_report": {"status": "failed"},
        "base_image_registry_report": {"status": "timeout"},
        "blockers": ["requested image is not available locally: lmx-freemhd"],
    }
    monkeypatch.setattr(probe, "freemhd_container_report", lambda **kwargs: payload)

    output = tmp_path / "container_probe.json"
    exit_code = probe.main(["--bundle-root", str(tmp_path / "docker"), "--output", str(output)])

    assert exit_code == 0
    assert output.exists()
    assert '"image": "lmx-freemhd"' in capsys.readouterr().out
