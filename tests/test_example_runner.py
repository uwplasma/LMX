import json
from pathlib import Path
import subprocess
import sys

import pytest


pytestmark = pytest.mark.unit


def test_fringing_benchmark_demo_runs_real_bounded_diagnostic(tmp_path: Path):
    script = Path(__file__).resolve().parents[1] / "examples/fringing_benchmark_demo.py"
    subprocess.run([sys.executable, script], cwd=tmp_path, timeout=30, check=True)
    summary_path = next(
        (tmp_path / "artifacts").rglob("fringing_benchmark_summary.json")
    )
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "research-stage internal diagnostic"
    assert summary["geometry_kind"] == "rect_duct"
    assert summary["shape"] == [7, 8, 8]
    assert summary["validation"]["station_count"] == 7
    assert summary["validation"]["max_charge_balance_residual"] < 1.0e-10
    assert len(summary["mean_velocity"]) == len(summary["field_scale"]) == 7
    assert all((summary_path.parent / name).is_file() for name in summary["plots"])


def test_variable_field_extruded_demo_writes_summary(tmp_path: Path):
    script = (
        Path(__file__).resolve().parents[1] / "examples/variable_field_extruded_demo.py"
    )
    subprocess.run([sys.executable, script], cwd=tmp_path, timeout=60, check=True)
    summary_path = next(
        (tmp_path / "artifacts").rglob("variable_field_extruded_summary.json")
    )
    summary = json.loads(summary_path.read_text())
    validation = summary["validation"]
    assert summary["case"].startswith("variable_field_duct")
    assert summary["geometry_kind"] == "rect_duct"
    assert validation["finite_velocity"] is True
    assert validation["mean_velocity_change"] > 0.0
    assert validation["current_proxy_change"] > 0.0
    assert isinstance(validation["validation_pass"], bool)
    assert (summary_path.parent / "extruded_overview.png").exists()


def test_extruded_restart_demo_is_state_exact(tmp_path: Path):
    script = Path(__file__).resolve().parents[1] / "examples/extruded_restart_demo.py"
    subprocess.run([sys.executable, script], cwd=tmp_path, timeout=60, check=True)
    summary_path = next((tmp_path / "artifacts").rglob("extruded_restart_summary.json"))
    summary = json.loads(summary_path.read_text())
    assert summary["max_state_difference"] == 0.0
    assert summary["max_mean_velocity_difference"] == 0.0
    assert summary["max_charge_balance_difference"] == 0.0
    assert (summary_path.parent / "extruded_restart_demo.png").is_file()


def test_pipe_reference_comparison_demo_writes_summary(tmp_path: Path):
    reference_dir = (
        tmp_path
        / "external/FreeMHDPaperAllFigures/FreeMHDPaperAllFigures/FringingBPipe"
    )
    reference_dir.mkdir(parents=True)
    header = "Points:2,U:2,Points:0,potE\n"
    for stem, offset in (("CenterLine", 0.0), ("NegXLine", -0.3), ("PosXLine", 0.3)):
        rows = "-1,0.5,{0},-1\n0,1,{0},0\n1,0.5,{0},1\n".format(offset)
        (reference_dir / f"sample_{stem}_data.csv").write_text(header + rows)
    script = (
        Path(__file__).resolve().parents[1]
        / "examples/pipe_reference_comparison_demo.py"
    )
    subprocess.run([sys.executable, script], cwd=tmp_path, timeout=60, check=True)
    summary_path = next(
        (tmp_path / "artifacts").rglob("pipe_reference_comparison_summary.json")
    )
    summary = json.loads(summary_path.read_text())
    assert summary["geometry_kind"] == "pipe_ogrid"
    assert summary["normalization"]["center"] == "independent_peak_axial_velocity"
    assert (summary_path.parent / "pipe_reference_comparison.png").exists()


def test_li_aln_wall_stack_example_runs_explicit_models(tmp_path: Path):
    script = (
        Path(__file__).resolve().parents[1]
        / "examples/li_aln_wall_stack_example.py"
    )
    subprocess.run([sys.executable, script], cwd=tmp_path, timeout=60, check=True)
    summary_path = next(
        (tmp_path / "artifacts").rglob("li_aln_wall_stack_summary.json")
    )
    summary = json.loads(summary_path.read_text())

    assert summary["inductionless_assumption_pass"] is True
    assert set(summary["models"]) == {"intact_aln", "bare_metal"}
    assert (
        summary["models"]["bare_metal"]["tangential_conductance_ratio"]
        > summary["models"]["intact_aln"]["tangential_conductance_ratio"]
    )
    assert all(
        model["validation"]["linear_residual"] >= 0.0
        for model in summary["models"].values()
    )
    assert all(
        model["validation"]["charge_balance_relative"] < 1.0e-8
        and model["validation"]["interface_current_relative"] < 1.0e-5
        for model in summary["models"].values()
    )
    assert (summary_path.parent / "li_aln_wall_stack.png").is_file()
