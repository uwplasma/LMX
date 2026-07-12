from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.freeze_samper_table_i import freeze_campaign


def _campaign(*, passed: bool = True) -> dict[str, object]:
    return {
        "schema_version": 1,
        "records": [
            {
                "case_kind": "shercliff",
                "hartmann_number": 5000,
                "finest_level_pass": passed,
            }
        ],
        "research_grade_validation_pass": passed,
    }


def test_freeze_campaign_writes_deterministic_compact_evidence(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    destination = tmp_path / "frozen.json"
    source.write_text(json.dumps(_campaign()), encoding="utf-8")

    payload = freeze_campaign(source, destination)

    assert payload["freeze"]["format"] == "compact-json"
    assert len(payload["freeze"]["source_sha256"]) == 64
    assert destination.read_text(encoding="utf-8").endswith("\n")
    assert not destination.with_suffix(".json.tmp").exists()


def test_freeze_campaign_rejects_incomplete_or_failing_evidence(tmp_path: Path) -> None:
    source = tmp_path / "source.json"
    destination = tmp_path / "frozen.json"
    incomplete = _campaign()
    incomplete["active_record"] = {}
    source.write_text(json.dumps(incomplete), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        freeze_campaign(source, destination)

    source.write_text(json.dumps(_campaign(passed=False)), encoding="utf-8")
    with pytest.raises(ValueError, match="failing"):
        freeze_campaign(source, destination)
