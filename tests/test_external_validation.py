from __future__ import annotations

from pathlib import Path

import pytest

from lmx.external_validation import (
    compare_magnetic_obstacle_reference_observables,
    load_magnetic_obstacle_reference_observables,
    magnetic_obstacle_reference_template_rows,
    write_magnetic_obstacle_reference_comparison_plots,
    write_magnetic_obstacle_reference_comparison_table,
    write_magnetic_obstacle_reference_template,
)


pytestmark = pytest.mark.unit


def test_magnetic_obstacle_reference_observable_csv_round_trip(tmp_path: Path):
    path = tmp_path / "reference.csv"
    path.write_text(
        "observable,value,tolerance,relative_tolerance,units,source\n"
        "centerline_velocity_deficit_ratio,0.25,0.02,0.10,dimensionless,Votyakov digitized figure\n"
        "pressure_drop_proxy,1.20,0.05,,dimensionless,Cuevas digitized figure\n",
        encoding="utf-8",
    )

    reference = load_magnetic_obstacle_reference_observables(path)

    assert reference["centerline_velocity_deficit_ratio"]["value"] == pytest.approx(0.25)
    assert reference["centerline_velocity_deficit_ratio"]["relative_tolerance"] == pytest.approx(0.10)
    assert reference["pressure_drop_proxy"]["source"] == "Cuevas digitized figure"


def test_compare_magnetic_obstacle_reference_observables_writes_publication_table(tmp_path: Path):
    reference = {
        "centerline_velocity_deficit_ratio": {"value": 0.25, "tolerance": 0.02, "relative_tolerance": 0.10},
        "wake_recovery_ratio": {"value": 0.92, "tolerance": 0.02},
        "pressure_drop_proxy": {"value": 1.2, "tolerance": 0.05, "units": "dimensionless"},
    }
    lmx_observables = {
        "centerline_velocity_deficit_ratio": 0.26,
        "wake_recovery_ratio": 0.91,
        "pressure_drop_proxy": 1.31,
        "current_proxy_peak": 4.0,
    }

    comparison = compare_magnetic_obstacle_reference_observables(lmx_observables, reference)
    table_path = write_magnetic_obstacle_reference_comparison_table(comparison, tmp_path / "comparison.csv")

    assert comparison["compared_observable_count"] == 3
    assert comparison["passed_observable_count"] == 2
    assert comparison["validation_pass"] is False
    assert "current_proxy_peak" in comparison["extra_lmx_observables"]
    assert table_path.read_text(encoding="utf-8").startswith("observable,lmx_value,reference_value")


def test_write_magnetic_obstacle_reference_comparison_plots(tmp_path: Path):
    comparison = compare_magnetic_obstacle_reference_observables(
        {
            "centerline_velocity_deficit_ratio": 0.26,
            "wake_recovery_ratio": 0.91,
        },
        {
            "centerline_velocity_deficit_ratio": {"value": 0.25, "tolerance": 0.02},
            "wake_recovery_ratio": {"value": 0.95, "tolerance": 0.02},
        },
    )

    paths = write_magnetic_obstacle_reference_comparison_plots(comparison, tmp_path)

    assert [path.suffix for path in paths] == [".png", ".pdf"]
    assert all(path.exists() for path in paths)
    assert all(path.stat().st_size > 0 for path in paths)


def test_compare_magnetic_obstacle_reference_observables_reports_missing_lmx_observable():
    comparison = compare_magnetic_obstacle_reference_observables(
        {"centerline_velocity_deficit_ratio": 0.2},
        {
            "centerline_velocity_deficit_ratio": {"value": 0.2, "tolerance": 0.01},
            "wake_recovery_ratio": {"value": 0.9, "tolerance": 0.02},
        },
    )

    assert comparison["missing_lmx_observable_count"] == 1
    assert comparison["validation_pass"] is False
    assert comparison["rows"][1]["status"] == "missing_lmx_observable"


def test_magnetic_obstacle_reference_template_has_required_columns(tmp_path: Path):
    template_rows = magnetic_obstacle_reference_template_rows()
    template_path = write_magnetic_obstacle_reference_template(tmp_path / "template.csv")
    text = template_path.read_text(encoding="utf-8")

    assert {"observable", "value", "tolerance", "relative_tolerance"} <= set(template_rows[0])
    assert "centerline_velocity_deficit_ratio" in text
    assert "pressure_drop_proxy" in text


def test_magnetic_obstacle_reference_csv_rejects_missing_columns(tmp_path: Path):
    path = tmp_path / "bad.csv"
    path.write_text("observable,value\npressure_drop_proxy,1.0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        load_magnetic_obstacle_reference_observables(path)
