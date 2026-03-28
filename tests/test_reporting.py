from pathlib import Path

import pytest

from scripts.summarize_ci_artifacts import (
    build_summary,
    main,
    render_markdown,
    summarize_benchmark_report,
    summarize_parity_report,
    summarize_validation_summary,
)


pytestmark = pytest.mark.unit


def test_summarize_validation_summary(tmp_path: Path):
    path = tmp_path / "summary.json"
    path.write_text(
        """
        {
          "hartmann": {"case": "hartmann", "residual": 1.0, "u_max": 2.0, "l2_error": 0.1},
          "shercliff": {
            "case": "shercliff",
            "residual": 0.5,
            "u_max": 1.0,
            "y_l2_error": 0.2,
            "z_l2_error": 0.3,
            "slice_y_l2_error": 0.4,
            "slice_z_l2_error": 0.5
          }
        }
        """
    )
    summaries = summarize_validation_summary(path)
    assert [item.case for item in summaries] == ["hartmann", "shercliff"]
    assert summaries[0].l2_error == pytest.approx(0.1)
    assert summaries[1].y_l2_error == pytest.approx(0.2)
    assert summaries[1].slice_y_l2_error == pytest.approx(0.4)


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
          "parity_output": "/tmp/parity_report.json",
          "parity_report": {
            "metrics": {
              "u_max_abs_diff": 0.01,
              "freemhd_sample_y_l2_error": 0.02,
              "freemhd_sample_z_l2_error": 0.03
            }
          }
        }
        """
    )
    summary = build_summary(validation, benchmark, parity)
    assert "## Validation" in summary["markdown"]
    assert "## Benchmark" in summary["markdown"]
    assert "## FreeMHD Parity" in summary["markdown"]
    assert "Slice Y L2" in summary["markdown"]
    out = tmp_path / "summary.json"
    md = tmp_path / "summary.md"
    md.write_text(
        render_markdown(
            summarize_validation_summary(validation),
            summarize_benchmark_report(benchmark),
            summarize_parity_report(parity),
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
          "reason": "freemhd-case-unavailable",
          "case_dir": "",
          "sample_output": "",
          "parity_output": ""
        }
        """
    )
    summary = summarize_parity_report(path)
    assert summary.status == "skipped"
    assert summary.reason == "freemhd-case-unavailable"


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
    assert "## FreeMHD Parity" in out_md.read_text()
