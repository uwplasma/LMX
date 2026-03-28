import tarfile
from pathlib import Path

import pytest

from scripts import materialize_starting_case as materialize


pytestmark = pytest.mark.unit


def _write_tar_gz(path: Path, member_name: str, content: str) -> None:
    payload = path.parent / f".{member_name.replace('/', '_')}"
    payload.write_text(content)
    with tarfile.open(path, "w:gz") as tar:
        tar.add(payload, arcname=member_name)
    payload.unlink()


def test_materialize_case_expands_expected_archives(tmp_path: Path):
    _write_tar_gz(tmp_path / "0.tar.gz", "0/U", "field")
    _write_tar_gz(tmp_path / "constant.tar.gz", "constant/regionProperties", "regions")
    _write_tar_gz(tmp_path / "system.tar.gz", "system/controlDict", "application test;")
    payload = materialize.materialize_case(tmp_path)
    assert payload["has_zero_dir"] is True
    assert payload["has_constant_dir"] is True
    assert payload["has_system_dir"] is True
    assert (tmp_path / "system" / "controlDict").read_text() == "application test;"


def test_materialize_case_reports_missing_archives(tmp_path: Path):
    payload = materialize.materialize_case(tmp_path)
    assert payload["missing_archives"] == ["0.tar.gz", "constant.tar.gz", "system.tar.gz"]
    assert payload["extracted_archives"] == []


def test_materialize_main_writes_output(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    output = tmp_path / "materialized.json"
    monkeypatch.setattr(materialize.argparse.ArgumentParser, "parse_args", lambda self: type("Args", (), {"case_dir": tmp_path, "output": output})())

    exit_code = materialize.main()

    assert exit_code == 0
    assert output.exists()
    assert '"case_dir"' in capsys.readouterr().out
