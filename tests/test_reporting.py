from pathlib import Path

import pytest

from scripts.summarize_ci_artifacts import (
    build_summary,
    render_markdown,
    summarize_benchmark_report,
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
    summary = build_summary(validation, benchmark)
    assert "## Validation" in summary["markdown"]
    assert "## Benchmark" in summary["markdown"]
    assert "Slice Y L2" in summary["markdown"]
    out = tmp_path / "summary.json"
    md = tmp_path / "summary.md"
    md.write_text(render_markdown(summarize_validation_summary(validation), summarize_benchmark_report(benchmark)))
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
