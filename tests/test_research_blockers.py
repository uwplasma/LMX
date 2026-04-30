from __future__ import annotations

import json
from pathlib import Path

import pytest

from lmx.research_blockers import (
    strict_blocker_closure_attempt_summary,
    write_strict_blocker_closure_attempt,
)


pytestmark = pytest.mark.unit


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_static_fixture(static_dir: Path) -> None:
    static_dir.mkdir(parents=True, exist_ok=True)
    for name in ("q2d_turbulence_decay_poster.png", "magnetic_obstacle_benchmark.png", "bent_pipe_overview.png"):
        (static_dir / name).write_bytes(b"plot")
    for name in (
        "q2dmhdfoam_external_reference_summary.json",
        "magnetic_obstacle_external_reference_template_summary.json",
        "dean_vortex_external_reference_template_summary.json",
    ):
        _write_json(static_dir / name, {"status": "template_only_no_external_reference_claim"})
    _write_json(
        static_dir / "q2d_turbulence_decay_summary.json",
        {
            "validation": {
                "validation_pass": True,
                "research_grade_turbulence_validation_pass": False,
            },
            "external_reference_comparison": {"status": "external_reference_csv_missing"},
        },
    )
    _write_json(
        static_dir / "magnetic_obstacle_benchmark_summary.json",
        {
            "validation": {
                "benchmark_pass": True,
                "conservation_pass": True,
                "research_grade_validation_pass": False,
            },
            "external_reference_comparison": {"status": "external_reference_csv_missing"},
        },
    )
    _write_json(
        static_dir / "bent_pipe_inductionless_summary.json",
        {
            "dean_number": 5.19e-7,
            "validation": {
                "validation_pass": True,
                "research_grade_charge_balance_pass": True,
                "research_grade_dean_validation_pass": False,
                "secondary_flow_peak_ratio": 0.0,
            },
            "external_reference_comparison": {"status": "external_reference_csv_missing"},
        },
    )


def test_strict_blocker_closure_attempt_keeps_strict_lanes_open(tmp_path: Path):
    static_dir = tmp_path / "generated"
    external_root = tmp_path / "external"
    _write_static_fixture(static_dir)
    (external_root / "Q2DmhdFoam/run/lidDriven").mkdir(parents=True)
    (external_root / "Q2DmhdFoam/run/lidDriven/IDM_output_U.txt").write_text("Weak turbulence:[]\n")

    summary = strict_blocker_closure_attempt_summary(static_dir=static_dir, external_codes_root=external_root)

    assert summary["release_decision"] == "do_not_tag_research_grade_release"
    assert summary["research_grade_ready"] is False
    assert summary["strict_closed_lane_count"] == 0
    assert set(summary["strict_open_lanes"]) == {
        "q2d_turbulence_external_parity",
        "magnetic_obstacle_external_validation",
        "dean_vortex_higher_inertia_validation",
    }
    magnetic = next(row for row in summary["lanes"] if row["lane"] == "magnetic_obstacle_external_validation")
    assert "0.997" in magnetic["key_result"]
    assert magnetic["release_decision"] == "block_research_grade_tag"


def test_write_strict_blocker_closure_attempt_artifacts(tmp_path: Path):
    static_dir = tmp_path / "generated"
    out_dir = tmp_path / "out"
    _write_static_fixture(static_dir)

    outputs = write_strict_blocker_closure_attempt(out_dir, static_dir=static_dir, external_codes_root=tmp_path / "ext")
    summary = json.loads((out_dir / "research_grade_strict_blocker_attempt.json").read_text(encoding="utf-8"))

    assert len(outputs) == 4
    assert all(path.exists() for path in outputs)
    assert (out_dir / "research_grade_strict_blocker_attempt.csv").read_text(encoding="utf-8").startswith("lane,status")
    assert summary["magnetic_obstacle_escalation"]["current_resolution_rerun"]["accepted"] is False
