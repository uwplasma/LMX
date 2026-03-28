from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import inspect_freemhd_case as inspect_case
from scripts import inspect_freemhd_setup as inspect_setup


pytestmark = pytest.mark.unit


def test_inspect_freemhd_case_main_writes_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    case_dir = tmp_path / "case"
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        inspect_case,
        "inspect_freemhd_case",
        lambda path: SimpleNamespace(
            case_dir=str(path),
            control_dicts=("system/controlDict",),
            fv_schemes=("system/fvSchemes",),
            fv_solutions=("system/fvSolution",),
            region_properties=("constant/regionProperties",),
            block_mesh_dicts=("system/blockMeshDict",),
            boundary_field_dirs=("0/liquid",),
            latest_time_dirs=("0.0001",),
        ),
    )
    monkeypatch.setattr(inspect_case, "docker_cli_available", lambda: True)
    monkeypatch.setattr(inspect_case, "docker_daemon_available", lambda: False)
    monkeypatch.setattr(
        inspect_case.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(case_dir=case_dir, output=output),
    )

    exit_code = inspect_case.main()

    assert exit_code == 0
    assert output.exists()
    assert '"docker_cli_available": true' in capsys.readouterr().out


def test_inspect_freemhd_setup_main_writes_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    output = tmp_path / "setup.json"
    monkeypatch.setattr(inspect_setup, "freemhd_environment_report", lambda repo_root, reference_root, extra_case_roots=None: {"status": "ok", "cases": []})
    monkeypatch.setattr(
        inspect_setup.argparse.ArgumentParser,
        "parse_args",
        lambda self, argv=None: SimpleNamespace(repo_root=tmp_path / "repo", reference_root=tmp_path / "refs", extra_case_root=[], output=output),
    )

    exit_code = inspect_setup.main([])

    assert exit_code == 0
    assert output.exists()
    assert '"status": "ok"' in capsys.readouterr().out
