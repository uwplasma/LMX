import tarfile
from pathlib import Path

import pytest

from scripts.materialize_starting_case import materialize_case


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
    payload = materialize_case(tmp_path)
    assert payload["has_zero_dir"] is True
    assert payload["has_constant_dir"] is True
    assert payload["has_system_dir"] is True
    assert (tmp_path / "system" / "controlDict").read_text() == "application test;"
