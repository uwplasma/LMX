from __future__ import annotations

import json
from pathlib import Path

import pytest

from lmx.research_figures import (
    VOTYAKOV_FIG7A_DIGITIZED,
    write_research_grade_closure_dashboard,
    write_research_grade_external_target_panel,
    write_research_grade_external_target_tables,
)


pytestmark = pytest.mark.unit


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_research_grade_external_target_tables_are_candidate_only(tmp_path: Path):
    outputs = write_research_grade_external_target_tables(tmp_path)

    assert len(outputs) == 5
    assert len(VOTYAKOV_FIG7A_DIGITIZED) >= 20
    magnetic_candidate = tmp_path / "magnetic_obstacle_reference_observables_candidate.csv"
    magnetic_text = magnetic_candidate.read_text(encoding="utf-8")
    assert "minimum_centerline_velocity_ratio" in magnetic_text
    assert "Candidate target only" in magnetic_text
    q2d_candidate = tmp_path / "q2d_turbulence_reference_observables_candidate.csv"
    dean_candidate = tmp_path / "dean_vortex_reference_observables_candidate.csv"
    assert "final_spectral_centroid" in q2d_candidate.read_text(encoding="utf-8")
    assert "secondary_flow_peak_ratio" in dean_candidate.read_text(encoding="utf-8")


def test_research_grade_external_target_panel_writes_media(tmp_path: Path):
    magnetic = tmp_path / "magnetic.json"
    q2d = tmp_path / "q2d.json"
    dean = tmp_path / "dean.json"
    idm = tmp_path / "IDM_output_U.txt"
    idm.write_text(
        "-1.0\n"
        "Weak turbulence:[[1.0, 0.2], [2.0, 0.1]]\n"
        "Strong turbulence:[[0.5, 0.15, 0.03]].\n",
        encoding="utf-8",
    )
    _write_json(
        magnetic,
        {
            "external_readiness": {
                "observables": {
                    "minimum_centerline_velocity_ratio": 0.98,
                    "centerline_velocity_deficit_ratio": 0.02,
                }
            }
        },
    )
    _write_json(q2d, {"turbulence_observables": {"source_path": str(idm)}})
    _write_json(
        dean,
        {
            "validation": {
                "dean_number": 1.0e-4,
                "secondary_flow_rms_ratio": 0.0,
                "secondary_flow_peak_ratio": 0.0,
                "normalized_velocity_centroid_shift": 0.0,
                "inner_outer_velocity_ratio": 1.0,
            }
        },
    )

    outputs = write_research_grade_external_target_panel(
        tmp_path,
        magnetic_summary_path=magnetic,
        q2dmhdfoam_summary_path=q2d,
        dean_summary_path=dean,
    )

    assert [path.suffix for path in outputs] == [".png", ".pdf"]
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)


def test_research_grade_closure_dashboard_writes_media_and_summary(tmp_path: Path):
    q2d = tmp_path / "q2d.json"
    magnetic = tmp_path / "magnetic.json"
    dean = tmp_path / "dean.json"
    closure = tmp_path / "closure.json"
    _write_json(
        q2d,
        {
            "strict_blocker_closed": True,
            "comparison": {
                "rows": [
                    {
                        "observable": "speed_mean",
                        "relative_error": 0.1,
                        "relative_tolerance": 0.2,
                        "validation_pass": True,
                    }
                ]
            },
        },
    )
    _write_json(
        magnetic,
        {
            "external_reference_comparison": {
                "comparison": {
                    "validation_pass": False,
                    "rows": [
                        {
                            "observable": "minimum_centerline_velocity_ratio",
                            "lmx_value": 0.98,
                            "reference_value": -0.13,
                            "effective_tolerance": 0.03,
                            "relative_error": 8.5,
                        }
                    ],
                }
            }
        },
    )
    _write_json(
        dean,
        {
            "current_lmx_dean_number": 1.0e-6,
            "reference_dean_number": 20.0,
            "external_reference_comparison": {
                "comparison": {
                    "validation_pass": False,
                    "rows": [
                        {
                            "observable": "secondary_flow_rms_ratio",
                            "lmx_value": 0.0,
                            "reference_value": 0.04,
                        }
                    ],
                }
            },
        },
    )
    _write_json(
        closure,
        {
            "closed_lane_count": 0,
            "lane_count": 3,
            "research_grade_ready": False,
            "open_lanes": ["q2d_turbulence_external_parity"],
            "rows": [
                {
                    "lane": "q2d_turbulence_external_parity",
                    "physics_gate_pass": False,
                    "external_gate_pass": False,
                    "closed": False,
                    "status": "external_adapter_ready_matched_parity_open",
                }
            ],
        },
    )

    outputs = write_research_grade_closure_dashboard(
        tmp_path,
        q2d_sidewall_summary_path=q2d,
        magnetic_strict_summary_path=magnetic,
        dean_strict_summary_path=dean,
        closure_status_path=closure,
    )

    assert [path.suffix for path in outputs] == [".png", ".pdf", ".json"]
    summary = json.loads((tmp_path / "research_grade_closure_dashboard_summary.json").read_text(encoding="utf-8"))
    assert summary["q2d_sidewall_gate_closed"] is True
    assert summary["magnetic_obstacle_external_validation_pass"] is False
    assert summary["research_grade_ready"] is False
    assert all(path.exists() and path.stat().st_size > 0 for path in outputs)
