from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.freeze_freemhd_benchmark_a import compact_evidence, freeze_summary


def _record(case: str) -> dict[str, object]:
    return {
        "case_kind": case,
        "benchmark_spec": {
            "id": f"A2-{case}",
            "path": f"{case}.toml",
            "sha256": "a" * 64,
        },
        "settings": {"ny": 85, "nz": 63},
        "observables": {
            "velocity": {"y": {"l2_error": 0.004}},
            "lorentz": {"y": {"l2_error": 0.005}},
        },
        "integral_observables": {"pressure_gradient_relative_error": 0.006},
        "current_balance": {"acceptance_target": 0.001},
        "power_balance": {"mechanical_power_relative_error": 1.0e-5},
        "continuum_velocity_audit": {
            "reference_path": f"/nonportable/reference/{case}.txt",
            "axes": {"y": {"lmx_raw_analytical": {"l2_error": 0.007}}},
        },
    }


def _summary() -> dict[str, object]:
    levels = []
    for label in ("coarse", "medium", "fine", "confirmation"):
        levels.append(
            {"label": label, "records": [_record("shercliff"), _record("hunt")]}
        )
    return {
        "best_level_label": "confirmation",
        "implementation": {
            "runner_sha256": "1" * 64,
            "solver_core_sha256": "2" * 64,
            "lmx_version": "1.2.3",
            "solvax_version": "0.4.0",
        },
        "ladder": levels,
        "richardson": {
            "schema_version": 1,
            "levels": ["medium", "fine", "confirmation"],
            "cases": {},
            "research_grade_validation_pass": False,
        },
    }


def test_compact_evidence_is_portable_and_uses_confirmation() -> None:
    evidence = compact_evidence(_summary(), source_sha256="b" * 64)
    assert set(evidence) == {"richardson", "continuum", "power"}
    assert evidence["power"]["confirmation_level"] == "confirmation"
    assert evidence["power"]["implementation"]["solver_core_sha256"] == "2" * 64
    assert evidence["power"]["cases"]["hunt"]["primary_errors"]["lorentz_y_l2"] == 0.005
    continuum = evidence["continuum"]["cases"]["shercliff"]
    assert continuum["reference_file"] == "shercliff.txt"
    assert "reference_path" not in continuum


def test_freeze_summary_is_deterministic(tmp_path: Path) -> None:
    source = tmp_path / "summary.json"
    source.write_text(json.dumps(_summary()), encoding="utf-8")
    first = freeze_summary(source, tmp_path / "first")
    second = freeze_summary(source, tmp_path / "second")
    assert first.keys() == second.keys()
    for kind in first:
        assert first[kind].read_bytes() == second[kind].read_bytes()


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda summary: summary.update(ladder=summary["ladder"][:3]), "at least four"),
        (lambda summary: summary.update(best_level_label="fine"), "Best level"),
        (lambda summary: summary.pop("implementation"), "implementation fingerprint"),
        (
            lambda summary: summary["richardson"].update(
                levels=["coarse", "medium", "fine"]
            ),
            "final three",
        ),
        (
            lambda summary: summary["ladder"][-1].update(records=[_record("hunt")]),
            "one Shercliff",
        ),
    ],
)
def test_compact_evidence_rejects_inconsistent_campaign(mutation, message: str) -> None:
    summary = _summary()
    mutation(summary)
    with pytest.raises(ValueError, match=message):
        compact_evidence(summary, source_sha256="c" * 64)
