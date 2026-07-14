from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.run_benchmark_suite as benchmark_suite
from scripts.summarize_ci_artifacts import (
    build_summary,
    main,
    render_markdown,
    summarize_benchmark_report,
    summarize_grid_report,
    summarize_parity_report,
    summarize_sweep_report,
    summarize_validation_summary,
)


pytestmark = pytest.mark.unit


def test_benchmark_workflow_writes_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "artifacts" / "benchmark.json"
    recorded: dict[str, object] = {}
    monkeypatch.setattr(
        benchmark_suite.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(output=output, repeats=2, ha=5.0, ny=8, nz=8),
    )
    monkeypatch.setattr(
        benchmark_suite,
        "benchmark_solver",
        lambda repeats, ha, ny, nz: {
            "case": "hartmann_ha5",
            "cold_seconds": 1.0,
            "warm_seconds": 0.5,
            "mean_seconds": 0.75,
            "repeats": float(repeats),
        },
    )
    monkeypatch.setattr(
        benchmark_suite,
        "write_benchmark_report",
        lambda payload, path: recorded.update(payload=payload, path=path) or Path(path),
    )
    assert benchmark_suite.main() == 0
    assert recorded["path"] == output
    assert recorded["payload"]["case"] == "hartmann_ha5"
    assert '"case": "hartmann_ha5"' in capsys.readouterr().out


def test_summarize_validation_summary(tmp_path: Path):
    path = tmp_path / "summary.json"
    path.write_text(
        """
        {
          "hartmann": {"case": "hartmann", "residual": 1.0, "potential_residual": 0.01, "potential_iterations_used": 50, "u_max": 2.0, "l2_error": 0.1},
          "shercliff": {
            "case": "shercliff",
            "residual": 0.5,
            "potential_residual": 0.02,
            "potential_iterations_used": 200,
            "u_max": 1.0,
            "y_l2_error": 0.2,
            "z_l2_error": 0.3,
            "combined_l2_error": 0.25,
            "slice_y_l2_error": 0.4,
            "slice_z_l2_error": 0.5,
            "slice_combined_l2_error": 0.45
          }
        }
        """
    )
    summaries = summarize_validation_summary(path)
    assert [item.case for item in summaries] == ["hartmann", "shercliff"]
    assert summaries[0].l2_error == pytest.approx(0.1)
    assert summaries[0].potential_residual == pytest.approx(0.01)
    assert summaries[0].potential_iterations_used == pytest.approx(50)
    assert summaries[1].y_l2_error == pytest.approx(0.2)
    assert summaries[1].combined_l2_error == pytest.approx(0.25)
    assert summaries[1].slice_y_l2_error == pytest.approx(0.4)
    assert summaries[1].slice_combined_l2_error == pytest.approx(0.45)


def test_summarize_benchmark_report(tmp_path: Path):
    path = tmp_path / "benchmark.json"
    path.write_text(
        """
        {
          "case": "hartmann_ha20",
          "cold_seconds": 1.5,
          "warm_seconds": 0.8,
          "mean_seconds": 1.0,
          "repeats": 2.0
        }
        """
    )
    summary = summarize_benchmark_report(path)
    assert summary.case == "hartmann_ha20"
    assert summary.warm_seconds == pytest.approx(0.8)


def test_render_markdown_and_build_summary(tmp_path: Path):
    validation = tmp_path / "validation.json"
    validation.write_text(
        """
        {
          "hartmann": {"case": "hartmann", "residual": 1.0, "potential_residual": 0.01, "potential_iterations_used": 50, "u_max": 2.0, "l2_error": 0.1}
        }
        """
    )
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        """
        {
          "case": "hartmann_ha20",
          "cold_seconds": 1.5,
          "warm_seconds": 0.8,
          "mean_seconds": 1.0,
          "repeats": 2.0
        }
        """
    )
    parity = tmp_path / "parity.json"
    parity.write_text(
        """
        {
          "status": "ok",
          "reason": "",
          "case_dir": "/tmp/case",
          "sample_output": "/tmp/sample.json",
          "parity_output": "/tmp/parity_report.json",
          "parity_report": {
            "metrics": {
              "u_max_abs_diff": 0.01,
              "reference_sample_y_l2_error": 0.02,
              "reference_sample_z_l2_error": 0.03
            },
            "observable_gate": {
              "research_grade_validation_pass": false,
              "observable_offender_count": 2,
              "missing_observable_count": 0,
              "low_signal_count": 1
            }
          }
        }
        """
    )
    time_convergence = tmp_path / "time_convergence.json"
    time_convergence.write_text(
        """
        {
          "case": "hunt",
          "parameter": "dt",
          "levels": [
            {"parameter_value": 0.002, "combined_l2_error": 0.22, "y_l2_error": 0.1, "z_l2_error": 0.3, "accepted": 1.0},
            {"parameter_value": 0.001, "combined_l2_error": 0.23, "y_l2_error": 0.2, "z_l2_error": 0.25, "accepted": 0.0}
          ]
        }
        """
    )
    control_sweep = tmp_path / "control_sweep.json"
    control_sweep.write_text(
        """
        {
          "case": "hunt",
          "parameter": "outer_iterations",
          "levels": [
            {"parameter_value": 2, "combined_l2_error": 0.35, "y_l2_error": 0.3, "z_l2_error": 0.4},
            {"parameter_value": 6, "combined_l2_error": 0.16, "y_l2_error": 0.1, "z_l2_error": 0.2}
          ]
        }
        """
    )
    control_grid = tmp_path / "control_grid.json"
    control_grid.write_text(
        """
        {
          "case": "hunt",
          "parameter_a": "outer_iterations",
          "parameter_b": "potential_relaxation",
          "levels": [
            {"parameter_a_value": 4, "parameter_b_value": 1.0, "combined_l2_error": 0.35, "y_l2_error": 0.3, "z_l2_error": 0.4},
            {"parameter_a_value": 6, "parameter_b_value": 0.5, "combined_l2_error": 0.16, "y_l2_error": 0.1, "z_l2_error": 0.2}
          ]
        }
        """
    )
    summary = build_summary(
        validation, benchmark, parity, time_convergence, control_sweep, control_grid
    )
    assert "## Validation" in summary["markdown"]
    assert "Potential residual" in summary["markdown"]
    assert "Potential iterations" in summary["markdown"]
    assert "Combined L2" in summary["markdown"]
    assert "## Benchmark" in summary["markdown"]
    assert "## External Reference Parity" in summary["markdown"]
    assert "Observable gate pass" in summary["markdown"]
    assert summary["parity"]["observable_gate_pass"] is False
    assert summary["parity"]["observable_offender_count"] == 2
    assert "## Time Convergence" in summary["markdown"]
    assert "## Control Sweep" in summary["markdown"]
    assert "## Control Grid" in summary["markdown"]
    assert "Best Y L2" in summary["markdown"]
    assert "Best Z L2" in summary["markdown"]
    assert "Best combined L2" in summary["markdown"]
    assert "Accepted levels" in summary["markdown"]
    assert "Slice Y L2" in summary["markdown"]
    out = tmp_path / "summary.json"
    md = tmp_path / "summary.md"
    md.write_text(
        render_markdown(
            summarize_validation_summary(validation),
            summarize_benchmark_report(benchmark),
            summarize_parity_report(parity),
            summarize_sweep_report(time_convergence, label="Time Convergence"),
            summarize_sweep_report(control_sweep, label="Control Sweep"),
            summarize_grid_report(control_grid, label="Control Grid"),
        )
    )
    out.write_text("placeholder")
    assert md.exists()


def test_build_summary_allows_benchmark_only(tmp_path: Path):
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        """
        {
          "case": "hartmann_ha20",
          "cold_seconds": 1.5,
          "warm_seconds": 0.8,
          "mean_seconds": 1.0,
          "repeats": 2.0
        }
        """
    )
    summary = build_summary(None, benchmark)
    assert summary["validation"] == []
    assert summary["benchmark"]["case"] == "hartmann_ha20"
    assert "## Benchmark" in summary["markdown"]


def test_summarize_parity_report(tmp_path: Path):
    path = tmp_path / "parity.json"
    path.write_text(
        """
        {
          "status": "skipped",
          "reason": "reference-case-unavailable",
          "case_dir": "",
          "sample_output": "",
          "parity_output": ""
        }
        """
    )
    summary = summarize_parity_report(path)
    assert summary.status == "skipped"
    assert summary.reason == "reference-case-unavailable"
    assert summary.observable_gate_pass is None


def test_summarize_sweep_report(tmp_path: Path):
    path = tmp_path / "sweep.json"
    path.write_text(
        """
        {
          "case": "hunt",
          "parameter": "outer_iterations",
          "levels": [
            {"parameter_value": 2, "combined_l2_error": 0.35, "y_l2_error": 0.3, "z_l2_error": 0.4, "accepted": 1.0},
            {"parameter_value": 6, "combined_l2_error": 0.16, "y_l2_error": 0.1, "z_l2_error": 0.2, "accepted": 0.0}
          ]
        }
        """
    )
    summary = summarize_sweep_report(path, label="Control Sweep")
    assert summary.case == "hunt"
    assert summary.parameter == "outer_iterations"
    assert summary.first_value == pytest.approx(2.0)
    assert summary.first_combined_l2_error == pytest.approx(0.35)
    assert summary.best_combined_value == pytest.approx(6.0)
    assert summary.best_combined_l2_error == pytest.approx(0.16)
    assert summary.best_y_value == pytest.approx(6.0)
    assert summary.best_y_l2_error == pytest.approx(0.1)
    assert summary.last_z_l2_error == pytest.approx(0.2)
    assert summary.best_z_value == pytest.approx(6.0)
    assert summary.best_z_l2_error == pytest.approx(0.2)
    assert summary.accepted_levels == 1
    assert summary.total_levels == 2
    assert summary.first_accepted is True
    assert summary.last_accepted is False


def test_summarize_grid_report(tmp_path: Path):
    path = tmp_path / "grid.json"
    path.write_text(
        """
        {
          "case": "hunt",
          "parameter_a": "outer_iterations",
          "parameter_b": "potential_relaxation",
          "levels": [
            {"parameter_a_value": 4, "parameter_b_value": 1.0, "combined_l2_error": 0.35, "y_l2_error": 0.3, "z_l2_error": 0.4},
            {"parameter_a_value": 6, "parameter_b_value": 0.5, "combined_l2_error": 0.16, "y_l2_error": 0.1, "z_l2_error": 0.2}
          ]
        }
        """
    )
    summary = summarize_grid_report(path, label="Control Grid")
    assert summary.case == "hunt"
    assert summary.parameter_a == "outer_iterations"
    assert summary.parameter_b == "potential_relaxation"
    assert summary.best_combined_a == pytest.approx(6.0)
    assert summary.best_combined_b == pytest.approx(0.5)
    assert summary.best_combined_l2_error == pytest.approx(0.16)
    assert summary.best_y_a == pytest.approx(6.0)
    assert summary.best_y_b == pytest.approx(0.5)
    assert summary.best_z_l2_error == pytest.approx(0.2)


def test_main_writes_json_and_markdown(tmp_path: Path):
    validation = tmp_path / "validation.json"
    validation.write_text(
        """
        {
          "hartmann": {"case": "hartmann", "residual": 1.0, "u_max": 2.0, "l2_error": 0.1}
        }
        """
    )
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        """
        {
          "case": "hartmann_ha20",
          "cold_seconds": 1.5,
          "warm_seconds": 0.8,
          "mean_seconds": 1.0,
          "repeats": 2.0
        }
        """
    )
    parity = tmp_path / "parity.json"
    parity.write_text(
        """
        {
          "status": "ok",
          "reason": "",
          "case_dir": "/tmp/case",
          "sample_output": "/tmp/sample.json",
          "parity_output": "/tmp/parity_report.json"
        }
        """
    )
    time_convergence = tmp_path / "time_convergence.json"
    time_convergence.write_text(
        """
        {
          "case": "hunt",
          "parameter": "dt",
          "levels": [
            {"parameter_value": 0.002, "combined_l2_error": 0.22, "y_l2_error": 0.1, "z_l2_error": 0.3, "accepted": 1.0},
            {"parameter_value": 0.001, "combined_l2_error": 0.23, "y_l2_error": 0.2, "z_l2_error": 0.25, "accepted": 0.0}
          ]
        }
        """
    )
    control_sweep = tmp_path / "control_sweep.json"
    control_sweep.write_text(
        """
        {
          "case": "hunt",
          "parameter": "outer_iterations",
          "levels": [
            {"parameter_value": 2, "combined_l2_error": 0.35, "y_l2_error": 0.3, "z_l2_error": 0.4},
            {"parameter_value": 6, "combined_l2_error": 0.16, "y_l2_error": 0.1, "z_l2_error": 0.2}
          ]
        }
        """
    )
    control_grid = tmp_path / "control_grid.json"
    control_grid.write_text(
        """
        {
          "case": "hunt",
          "parameter_a": "outer_iterations",
          "parameter_b": "potential_relaxation",
          "levels": [
            {"parameter_a_value": 4, "parameter_b_value": 1.0, "combined_l2_error": 0.35, "y_l2_error": 0.3, "z_l2_error": 0.4},
            {"parameter_a_value": 6, "parameter_b_value": 0.5, "combined_l2_error": 0.16, "y_l2_error": 0.1, "z_l2_error": 0.2}
          ]
        }
        """
    )
    out_json = tmp_path / "summary.json"
    out_md = tmp_path / "summary.md"

    exit_code = main(
        [
            "--validation-summary",
            str(validation),
            "--benchmark-report",
            str(benchmark),
            "--parity-summary",
            str(parity),
            "--time-convergence-summary",
            str(time_convergence),
            "--control-sweep-summary",
            str(control_sweep),
            "--control-grid-summary",
            str(control_grid),
            "--output-json",
            str(out_json),
            "--output-md",
            str(out_md),
        ]
    )

    assert exit_code == 0
    assert out_json.exists()
    assert out_md.exists()
    assert '"hartmann_ha20"' in out_json.read_text()
    assert "## External Reference Parity" in out_md.read_text()
    assert "## Time Convergence" in out_md.read_text()
    assert "## Control Grid" in out_md.read_text()
    assert "Best Y L2" in out_md.read_text()
