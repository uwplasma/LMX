from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from scripts.build_benchmark_a_acceptance import build_acceptance, write_acceptance


RESULTS = Path("benchmarks/results")


def _copy_evidence(destination: Path) -> None:
    destination.mkdir()
    for path in RESULTS.glob("benchmark-a-ha20-*.json"):
        shutil.copy2(path, destination / path.name)
    for path in RESULTS.glob("samper-table-i-*-ha*.json"):
        shutil.copy2(path, destination / path.name)


def test_build_acceptance_separates_claims_and_accepts_all_rows() -> None:
    payload = build_acceptance(RESULTS)
    assert payload["research_grade_validation_pass"] is True
    assert payload["finite_grid_freemhd"]["pass"] is True
    assert payload["analytical_continuum_audit"]["pass"] is True
    assert payload["conservation_and_power"]["pass"] is True
    assert payload["richardson_diagnostic"]["acceptance_gate"] is False
    assert payload["richardson_diagnostic"]["all_extrapolated_primary_pass"] is False
    assert len(payload["literature_table_i"]["rows"]) == 8


def test_write_acceptance_is_deterministic(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_acceptance(RESULTS, first)
    write_acceptance(RESULTS, second)
    assert first.read_bytes() == second.read_bytes()


@pytest.mark.parametrize("failure", ("fingerprint", "row"))
def test_build_acceptance_rejects_inconsistent_or_failing_evidence(
    tmp_path: Path, failure: str
) -> None:
    copied = tmp_path / "results"
    _copy_evidence(copied)
    path = copied / "samper-table-i-hunt-ha15000.json"
    payload = json.loads(path.read_text())
    if failure == "fingerprint":
        payload["implementation"]["solver_core_sha256"] = "0" * 64
    else:
        payload["research_grade_validation_pass"] = False
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="implementation|not a passing"):
        build_acceptance(copied)
