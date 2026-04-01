from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from scripts import probe_freemhd_environment as probe


pytestmark = pytest.mark.unit


def test_classify_build_probe_covers_known_failures():
    assert probe.classify_build_probe("wmkdepend missing", False) == "missing-wmkdepend"
    assert probe.classify_build_probe("fatal error: 'fvMesh.H' file not found", True, False) == "missing-openfoam-lninclude"
    assert (
        probe.classify_build_probe("fatal error: 'fvMesh.H' file not found", True, True)
        == "post-darwin-header-patch-include-regression"
    )
    assert probe.classify_build_probe("", True) == "ok"
    assert probe.classify_build_probe("something else", True) == "unknown-build-failure"


def test_detect_shadowed_c_headers_and_recommendation():
    stderr = "lnInclude/string.h\nlnInclude/wchar.h"

    assert probe.detect_shadowed_c_headers(stderr) == ["string.h", "wchar.h"]
    recommendation = probe.build_issue_recommendation("macos-libcxx-header-conflict", stderr)
    assert "Observed shadowed headers: string.h, wchar.h." in recommendation


def test_probe_freemhd_environment_reports_expected_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    repo_root = tmp_path / "FreeMHD"
    foam_root = repo_root / "OpenFOAM-v2206"
    solver_dir = repo_root / "MHD_Solvers" / "solvers" / "epotMultiRegionFoam"
    (foam_root / "etc").mkdir(parents=True)
    (foam_root / "etc" / "bashrc").write_text("source me")
    (foam_root / "platforms" / "tools" / "darwin64Clang").mkdir(parents=True)
    (foam_root / "platforms" / "tools" / "darwin64Clang" / "wmkdepend").write_text("wmkdepend")
    (foam_root / "wmake" / "rules" / "darwin64Clang").mkdir(parents=True)
    rule_text = "DARWIN_LIB_HEADER_DIRS := test\n"
    (foam_root / "wmake" / "rules" / "darwin64Clang" / "c++").write_text(rule_text)
    (foam_root / "wmake" / "rules" / "darwin64Clang" / "c").write_text(rule_text)
    solver_dir.mkdir(parents=True)

    results = iter(
        [
            subprocess.CompletedProcess(args=["bash"], returncode=0, stdout="foam-ok", stderr=""),
            subprocess.CompletedProcess(
                args=["bash"],
                returncode=1,
                stdout="",
                stderr="fatal error: 'fvMesh.H' file not found",
            ),
        ]
    )
    monkeypatch.setattr(probe, "_run_shell", lambda command: next(results))
    monkeypatch.setattr(probe, "docker_cli_available", lambda: True)
    monkeypatch.setattr(probe, "docker_daemon_available", lambda: False)

    payload = probe.probe_freemhd_environment(repo_root)

    assert payload["wmkdepend_exists"] is True
    assert payload["darwin_header_patch_detected"] is True
    assert payload["solver_build_issue"] == "post-darwin-header-patch-include-regression"
    assert payload["docker_cli_available"] is True
    assert payload["docker_daemon_available"] is False


def test_main_writes_probe_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    output = tmp_path / "probe.json"
    expected = {"status": "ok", "solver_build_issue": "ok"}
    monkeypatch.setattr(probe, "probe_freemhd_environment", lambda repo_root: expected)

    monkeypatch.setattr(
        probe.argparse.ArgumentParser,
        "parse_args",
        lambda self, argv=None: type("Args", (), {"repo_root": tmp_path / "FreeMHD", "output": output})(),
    )

    exit_code = probe.main([])

    assert exit_code == 0
    assert json.loads(output.read_text()) == expected
    assert json.loads(capsys.readouterr().out) == expected
