from pathlib import Path

import pytest

from lmx.example_runner import run_case_example


pytestmark = pytest.mark.unit


def test_run_case_example_writes_hartmann_outputs(tmp_path: Path):
    report = run_case_example(
        case_kind="hartmann",
        ha=5.0,
        ny=12,
        nz=12,
        out_dir=tmp_path,
        reference_root=None,
    )

    assert report["case"] == "hartmann_ha5"
    assert (tmp_path / "overview.png").exists()
    assert (tmp_path / "overview.pdf").exists()
    assert (tmp_path / "diagnostics.png").exists()
    assert (tmp_path / "diagnostics.pdf").exists()
    assert (tmp_path / "example_report.json").exists()
    assert (tmp_path / "hartmann_ha5.vtr").exists()
    assert (tmp_path / "hartmann_ha5_centerline.csv").exists()
