import json
from pathlib import Path

import pytest

from lmx.publication import (
    PUBLICATION_FIGURE_SPECS,
    publication_figure_campaign_summary,
    publication_figure_rows,
    write_publication_figure_manifest,
)


pytestmark = pytest.mark.unit


def _write_minimal_static(static_dir: Path) -> None:
    static_dir.mkdir(parents=True, exist_ok=True)
    for spec in PUBLICATION_FIGURE_SPECS:
        (static_dir / spec.artifact).write_bytes(b"plot")
        (static_dir / spec.summary).write_text(
            json.dumps({"validation": {"validation_pass": True, "value": 1.0}}) + "\n"
        )
    (static_dir / "straight_duct_profile_comparison_summary.json").write_text(
        json.dumps(
            {
                "hartmann": {"l2_error": 0.001},
                "shercliff": {"y_l2_error": 0.002, "z_l2_error": 0.003},
                "hunt": {"y_l2_error": 0.004, "z_l2_error": 0.005},
            }
        )
        + "\n"
    )
    (static_dir / "wham_blanket_transient_flow_summary.json").write_text(
        json.dumps(
            {
                "metrics": {
                    "final_mean_velocity_m_per_s": 0.2,
                    "final_pressure_drop_kpa": 26.0,
                    "final_bend_outboard_to_inboard_ratio": 1.05,
                    "steady_state_reached": True,
                }
            }
        )
        + "\n"
    )


def test_publication_figure_rows_collect_metrics(tmp_path: Path):
    static_dir = tmp_path / "generated"
    _write_minimal_static(static_dir)

    rows = publication_figure_rows(static_dir)
    closed = next(row for row in rows if row["family"] == "closed_duct_profiles")
    transient = next(row for row in rows if row["family"] == "wham_blanket_transient")

    assert len(rows) == len(PUBLICATION_FIGURE_SPECS)
    assert all(row["artifact_exists"] for row in rows)
    assert closed["selected_metrics"]["max_l2_error"] == pytest.approx(0.005)
    assert transient["selected_metrics"][
        "final_bend_outboard_to_inboard_ratio"
    ] == pytest.approx(1.05)


def test_write_publication_figure_manifest(tmp_path: Path):
    static_dir = tmp_path / "generated"
    out_dir = tmp_path / "out"
    _write_minimal_static(static_dir)

    outputs = write_publication_figure_manifest(out_dir, static_dir=static_dir)
    summary = publication_figure_campaign_summary(static_dir)

    assert (out_dir / "publication_figure_campaign_summary.json").exists()
    assert (out_dir / "publication_figure_campaign_table.csv").exists()
    assert all(path.exists() for path in outputs)
    assert summary["release_blocking"] is False
    assert summary["figure_count"] == len(PUBLICATION_FIGURE_SPECS)
