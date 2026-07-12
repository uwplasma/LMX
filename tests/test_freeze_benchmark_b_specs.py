from __future__ import annotations

import json
from pathlib import Path

from scripts.freeze_benchmark_b_specs import build_specification_index


def test_benchmark_b_specification_index_is_complete_and_deterministic():
    expected = build_specification_index()
    tracked = json.loads(
        Path("benchmarks/results/benchmark-b-specification.json").read_text(
            encoding="utf-8"
        )
    )
    assert tracked == expected
    assert expected["specification_freeze_pass"] is True
    assert expected["production_results_included"] is False
    assert {case["id"] for case in expected["cases"]} == {
        "B1-fringing-pipe",
        "B2-fringing-square",
    }
    assert all(
        case["mesh_levels"] == ["coarse", "medium", "fine"]
        for case in expected["cases"]
    )
    assert len(expected["production_blockers"]) == 2
    assert all(
        case["numerical_independence"]["tolerance_uncertainty_fraction_max"] == 0.25
        for case in expected["cases"]
    )
