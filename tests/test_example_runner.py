from pathlib import Path

import pytest

from lmx.example_runner import run_case_example, run_theory_meeting_demo


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


def test_run_theory_meeting_demo_writes_movies_and_reports(tmp_path: Path):
    report = run_theory_meeting_demo(
        out_dir=tmp_path,
        hartmann_ha=5.0,
        shercliff_ha=5.0,
        hunt_ha=5.0,
        resolution=12,
        hunt_resolution=10,
        hunt_dt=1e-5,
        hunt_t_final=3e-5,
        hunt_frames=3,
        reference_root=None,
    )

    assert "hartmann" in report
    assert "shercliff" in report
    assert "hunt" in report
    assert report["hunt"]["movie_mode"] == "bulk_deviation"
    assert (tmp_path / "meeting_demo_report.json").exists()
    assert (tmp_path / "hunt" / "hunt_boundary_layers_2d.gif").exists()
    assert (tmp_path / "hunt" / "hunt_boundary_layers_3d.gif").exists()
    assert (tmp_path / "hunt" / "hunt_boundary_layers_2d_poster.png").exists()
    assert (tmp_path / "hunt" / "hunt_boundary_layers_2d_poster.pdf").exists()
    assert (tmp_path / "hunt" / "hunt_boundary_layers_3d_poster.png").exists()
    assert (tmp_path / "hunt" / "hunt_boundary_layers_3d_poster.pdf").exists()
