import json
from pathlib import Path

import pytest

from lmx.research_closure import (
    RESEARCH_CLOSURE_LANES,
    research_grade_external_data_audit,
    research_grade_closure_rows,
    research_grade_closure_status,
    research_grade_final_disposition,
    write_research_grade_external_data_audit,
    write_research_grade_closure_status,
    write_research_grade_final_disposition,
)


pytestmark = pytest.mark.unit


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_open_static(static_dir: Path) -> None:
    static_dir.mkdir(parents=True, exist_ok=True)
    for spec in RESEARCH_CLOSURE_LANES:
        (static_dir / spec.primary_artifact).write_bytes(b"plot")
        _write_json(
            static_dir / spec.external_summary,
            {"status": "template_only_no_external_reference_claim"},
        )
    _write_json(
        static_dir / "q2d_turbulence_decay_summary.json",
        {
            "validation": {
                "validation_pass": True,
                "frame_count": 72,
                "turnover_count": 0.3,
                "max_courant": 0.05,
                "research_grade_turbulence_validation_pass": False,
            },
            "external_reference_comparison": {
                "status": "external_reference_csv_missing"
            },
        },
    )
    _write_json(
        static_dir / "magnetic_obstacle_benchmark_summary.json",
        {
            "validation": {
                "benchmark_pass": True,
                "research_grade_validation_pass": False,
                "max_charge_balance_residual": 1.0e-12,
            },
            "external_reference_comparison": {
                "status": "external_reference_csv_missing"
            },
        },
    )
    _write_json(
        static_dir / "bent_pipe_inductionless_summary.json",
        {
            "validation": {
                "validation_pass": True,
                "research_grade_charge_balance_pass": True,
                "research_grade_dean_validation_pass": False,
            },
            "external_reference_comparison": {
                "status": "external_reference_csv_missing"
            },
        },
    )


def test_research_grade_closure_status_tracks_open_lanes(tmp_path: Path):
    static_dir = tmp_path / "generated"
    _write_open_static(static_dir)

    rows = research_grade_closure_rows(static_dir)
    status = research_grade_closure_status(static_dir)

    assert len(rows) == len(RESEARCH_CLOSURE_LANES)
    assert status["research_grade_ready"] is False
    assert status["closed_lane_count"] == 0
    assert set(status["open_lanes"]) == {spec.lane for spec in RESEARCH_CLOSURE_LANES}
    dean = next(
        row for row in rows if row["lane"] == "dean_vortex_higher_inertia_validation"
    )
    assert dean["status"] == "resolved_secondary_flow_open"
    assert dean["physics_gate_pass"] is False


def test_write_research_grade_closure_status(tmp_path: Path):
    static_dir = tmp_path / "generated"
    out_dir = tmp_path / "out"
    _write_open_static(static_dir)

    outputs = write_research_grade_closure_status(out_dir, static_dir=static_dir)
    summary = json.loads(
        (out_dir / "research_grade_closure_status.json").read_text(encoding="utf-8")
    )

    assert all(path.exists() for path in outputs)
    assert (out_dir / "research_grade_closure_status.csv").exists()
    assert summary["lane_count"] == len(RESEARCH_CLOSURE_LANES)
    assert summary["strict_research_blocking"] is True


def test_research_grade_final_disposition_records_deferred_lanes(tmp_path: Path):
    static_dir = tmp_path / "generated"
    _write_open_static(static_dir)
    _write_json(
        static_dir / "q2d_lmx_q2dmhdfoam_lid_driven_parity_summary.json",
        {"strict_blocker_closed": True},
    )
    _write_json(
        static_dir / "magnetic_obstacle_votyakov_strict_attempt_summary.json",
        {
            "external_reference_comparison": {
                "validation_pass": False,
                "comparison": {
                    "rows": [
                        {
                            "observable": "minimum_centerline_velocity_ratio",
                            "lmx_value": 0.98,
                            "reference_value": -0.13,
                            "relative_error": 8.5,
                        }
                    ]
                },
            }
        },
    )
    _write_json(
        static_dir / "dean_vortex_bayat_rezai_strict_attempt_summary.json",
        {
            "current_lmx_dean_number": 1.0e-6,
            "reference_dean_number": 20.0,
            "external_reference_comparison": {
                "validation_pass": False,
                "comparison": {
                    "rows": [
                        {
                            "observable": "secondary_flow_rms_ratio",
                            "lmx_value": 0.0,
                            "reference_value": 0.04,
                        },
                        {
                            "observable": "secondary_flow_peak_ratio",
                            "lmx_value": 0.0,
                            "reference_value": 0.08,
                        },
                    ]
                },
            },
        },
    )

    summary = research_grade_final_disposition(static_dir)

    assert summary["final_push_complete"] is True
    assert summary["research_grade_ready"] is False
    assert summary["deferred_lane_count"] == 3
    assert any(
        row["final_decision"] == "defer_strict_turbulent_parity"
        for row in summary["rows"]
    )


def test_write_research_grade_final_disposition(tmp_path: Path):
    static_dir = tmp_path / "generated"
    out_dir = tmp_path / "out"
    _write_open_static(static_dir)

    outputs = write_research_grade_final_disposition(out_dir, static_dir=static_dir)
    summary = json.loads(
        (out_dir / "research_grade_final_disposition.json").read_text(encoding="utf-8")
    )

    assert [path.suffix for path in outputs] == [".json", ".csv", ".png"]
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)
    assert summary["case"] == "research_grade_final_disposition"


def test_research_grade_external_data_audit_reports_available_inputs(tmp_path: Path):
    static_dir = tmp_path / "generated"
    external_root = tmp_path / "external"
    freemhd_root = tmp_path / "freemhd"
    _write_open_static(static_dir)
    (external_root / "Q2DmhdFoam/run/lidDriven").mkdir(parents=True)
    (external_root / "Q2DmhdFoam/run/lidDriven/IDM_output_U.txt").write_text(
        "Weak turbulence:[]\n"
    )
    (external_root / "MHD_Solvers_OpenFOAM/solvers/mhdEpotFoam").mkdir(parents=True)
    (
        external_root / "MHD_Solvers_OpenFOAM/solvers/mhdEpotFoam/mhdEpotFoam.C"
    ).write_text("// solver\n")
    (freemhd_root / "FreeMHDPaperAllFigures/ClosedChannel").mkdir(parents=True)
    (freemhd_root / "freemhd_paper.pdf").write_bytes(b"%PDF")

    audit = research_grade_external_data_audit(
        static_dir=static_dir,
        external_codes_root=external_root,
        freemhd_cases_root=freemhd_root,
    )
    q2d = next(
        row for row in audit["rows"] if row["lane"] == "q2d_turbulence_external_parity"
    )
    magnetic = next(
        row
        for row in audit["rows"]
        if row["lane"] == "magnetic_obstacle_external_validation"
    )

    assert audit["source_count"] == 4
    assert audit["strict_closure"]["closed_lane_count"] == 0
    assert q2d["path_exists"] is True
    assert q2d["matched_reference_csv_exists"] is False
    assert magnetic["available_evidence_file_count"] >= 2


def test_write_research_grade_external_data_audit(tmp_path: Path):
    static_dir = tmp_path / "generated"
    external_root = tmp_path / "external"
    out_dir = tmp_path / "out"
    _write_open_static(static_dir)

    output = write_research_grade_external_data_audit(
        out_dir,
        static_dir=static_dir,
        external_codes_root=external_root,
        freemhd_cases_root=tmp_path / "freemhd",
    )
    summary = json.loads(output.read_text(encoding="utf-8"))

    assert output.exists()
    assert summary["case"] == "research_grade_external_data_audit"
    assert summary["matched_reference_csv_count"] == 0
