import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.xdist_group(name="examples")]


@pytest.mark.curated
def test_portable_duct_tutorials_and_toml_first_run(tmp_path: Path):
    examples = Path(__file__).resolve().parents[1] / "examples"
    for script_name in ("hartmann_example.py", "hunt_example.py"):
        subprocess.run([sys.executable, examples / script_name], cwd=tmp_path, timeout=30, check=True)

    hartmann = json.loads(next((tmp_path / "artifacts").rglob("hartmann_summary.json")).read_text())
    hunt = json.loads(next((tmp_path / "artifacts").rglob("hunt_summary.json")).read_text())
    assert hartmann["analytical_profile"]["l2_error"] < 0.05
    assert hunt["validation"]["interface_current_residual"] < 1.0e-8

    source = Path(__file__).resolve().parents[1] / "examples/hartmann_case.toml"
    case_path = tmp_path / source.name
    case_path.write_text(
        source.read_text().replace("../artifacts/examples/toml_hartmann", "artifacts/toml_hartmann")
    )
    subprocess.run([sys.executable, "-m", "lmx", case_path], cwd=tmp_path, timeout=30, check=True)
    summary = json.loads(next((tmp_path / "artifacts").rglob("hartmann_ha20_toml_summary.json")).read_text())
    assert summary["converged"] is True
    assert summary["status"] == "converged"
    assert summary["residual"] < 1.0e-8


def test_fringing_benchmark_demo_runs_real_bounded_diagnostic(tmp_path: Path):
    script = Path(__file__).resolve().parents[1] / "examples/fringing_benchmark_demo.py"
    subprocess.run([sys.executable, script], cwd=tmp_path, timeout=60, check=True)
    summary_path = next((tmp_path / "artifacts").rglob("fringing_benchmark_summary.json"))
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "research-stage internal diagnostic"
    assert summary["geometry_kind"] == "rect_duct"
    assert summary["shape"] == [5, 6, 6]
    assert summary["validation"]["station_count"] == 5
    assert summary["validation"]["max_charge_balance_residual"] < 1.0e-10
    assert len(summary["mean_velocity"]) == len(summary["field_scale"]) == 5
    assert all((summary_path.parent / name).is_file() for name in summary["plots"])


def test_variable_field_extruded_demo_optimizes_with_checked_gradients(tmp_path: Path):
    script = Path(__file__).resolve().parents[1] / "examples/variable_field_extruded_demo.py"
    subprocess.run([sys.executable, script], cwd=tmp_path, timeout=120, check=True)
    summary_path = next((tmp_path / "artifacts").rglob("variable_field_extruded_summary.json"))
    summary = json.loads(summary_path.read_text())
    assert summary["shape"] == [7, 6, 6]
    assert summary["controls"]["field_mean"] == pytest.approx(1.0, abs=1.0e-12)
    assert all(0.8 < value < 1.2 for value in summary["controls"]["field_scale"])
    assert 0.5 < summary["controls"]["wall_conductivity_scale"] < 1.5
    assert all(
        1.0 - half_range < value < 1.0 + half_range
        for value, half_range in zip(summary["controls"]["geometry_scale"], (0.10, 0.05, 0.05), strict=True)
    )
    assert summary["optimization"]["final_loss"] < 0.8 * summary["optimization"]["initial_loss"]
    assert summary["gradient_check"]["relative_l2_error"] < 2.0e-3
    assert (
        min(summary["improvements"][name] for name in ("pumping_power_magnitude", "wall_current_density_rms"))
        > 0.4
    )
    assert abs(summary["improvements"]["flow_rate_relative_change"]) < 0.01
    assert (summary_path.parent / summary["plot"]).is_file()


def test_li_aln_wall_stack_example_runs_explicit_models(tmp_path: Path):
    script = Path(__file__).resolve().parents[1] / "examples/li_aln_wall_stack_example.py"
    subprocess.run([sys.executable, script], cwd=tmp_path, timeout=60, check=True)
    summary_path = next((tmp_path / "artifacts").rglob("li_aln_wall_stack_summary.json"))
    summary = json.loads(summary_path.read_text())

    assert summary["inductionless_assumption_pass"] is True
    assert set(summary["models"]) == {"intact_aln", "bare_metal"}
    assert (
        summary["models"]["bare_metal"]["tangential_conductance_ratio"]
        > summary["models"]["intact_aln"]["tangential_conductance_ratio"]
    )
    assert all(model["validation"]["linear_residual"] >= 0.0 for model in summary["models"].values())
    assert all(
        model["validation"]["charge_balance_relative"] < 1.0e-8
        and model["validation"]["interface_current_relative"] < 1.0e-5
        for model in summary["models"].values()
    )
    assert (summary_path.parent / "li_aln_wall_stack.png").is_file()
