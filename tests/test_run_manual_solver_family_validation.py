from pathlib import Path
import json

import pytest

from scripts import run_manual_solver_family_validation as manual_validation


pytestmark = pytest.mark.unit


def test_main_writes_manual_solver_family_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        manual_validation,
        "solve_steady",
        lambda case: type("Solution", (), {"mesh": object(), "case_name": case.name})(),
    )
    monkeypatch.setattr(
        manual_validation,
        "validation_summary",
        lambda solution, case_name, ha=None: {"case": case_name, "u_max": 1.0},
    )
    monkeypatch.setattr(manual_validation, "duct_layer_resolution_metrics", lambda case, mesh: {"hartmann_layer_cells": 3.0})
    monkeypatch.setattr(
        manual_validation,
        "hartmann_acceptance",
        lambda solution, ha, l2_threshold, linf_threshold: type(
            "Acceptance",
            (),
            {"passed": True, "l2_error": 0.01, "linf_error": 0.02},
        )(),
    )
    monkeypatch.setattr(
        manual_validation,
        "closed_channel_validation",
        lambda solution, case_kind, ha, reference_root=None: type(
            "Closed",
            (),
            {
                "y_profile": type("P", (), {"l2_error": 0.03, "linf_error": 0.04})(),
                "z_profile": type("P", (), {"l2_error": 0.05, "linf_error": 0.06})(),
            },
        )(),
    )

    output = tmp_path / "manual_summary.json"
    exit_code = manual_validation.main(
        [
            "--output",
            str(output),
            "--ha-values",
            "10",
            "--resolution",
            "8",
            "--reference-root",
            str(tmp_path / "refs"),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text())
    assert "hartmann_ha10" in payload
    assert "shercliff_ha10" in payload
    assert "hunt_ha10" in payload
