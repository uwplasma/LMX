from __future__ import annotations

import json
from pathlib import Path

import pytest

from lmx.research_figures import (
    VOTYAKOV_FIG7A_DIGITIZED,
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
