from pathlib import Path

import pytest

from scripts import probe_freemhd_environment as probe


pytestmark = pytest.mark.unit


class _Result:
    def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_probe_freemhd_environment_collects_expected_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo_root = tmp_path / "FreeMHD"
    (repo_root / "OpenFOAM-v2206" / "etc").mkdir(parents=True)
    (repo_root / "OpenFOAM-v2206" / "etc" / "bashrc").write_text("# bashrc\n")
    (repo_root / "OpenFOAM-v2206" / "platforms" / "tools" / "darwin64Clang").mkdir(parents=True)
    (repo_root / "OpenFOAM-v2206" / "platforms" / "tools" / "darwin64Clang" / "wmkdepend").write_text("")
    (repo_root / "MHD_Solvers" / "solvers" / "epotMultiRegionFoam").mkdir(parents=True)

    calls: list[str] = []

    def fake_run_shell(command: str) -> _Result:
        calls.append(command)
        if "foamSystemCheck" in command:
            return _Result(0, stdout="System check: PASS")
        return _Result(2, stderr="wmkdepend missing")

    monkeypatch.setattr(probe, "_run_shell", fake_run_shell)
    monkeypatch.setattr(probe, "docker_cli_available", lambda: True)
    monkeypatch.setattr(probe, "docker_daemon_available", lambda: False)

    payload = probe.probe_freemhd_environment(repo_root)

    assert payload["bashrc_exists"] is True
    assert payload["wmkdepend_exists"] is True
    assert payload["docker_cli_available"] is True
    assert payload["docker_daemon_available"] is False
    assert payload["foam_system_check_returncode"] == 0
    assert payload["solver_build_probe_returncode"] == 2
    assert payload["solver_build_issue"] == "unknown-build-failure"
    assert len(calls) == 2


def test_classify_build_probe_detects_expected_failures():
    assert probe.classify_build_probe("wmkdepend: No such file or directory", wmkdepend_exists=False) == "missing-wmkdepend"
    assert (
        probe.classify_build_probe("<cstring> tried including <string.h> but didn't find libc++'s <string.h> header.", True)
        == "macos-libcxx-header-conflict"
    )
    assert probe.classify_build_probe("", wmkdepend_exists=True) == "ok"
