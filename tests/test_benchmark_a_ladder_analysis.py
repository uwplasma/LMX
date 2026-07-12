import copy

import pytest

from scripts.analyze_freemhd_benchmark_a_ladder import analyze_ladder


pytestmark = pytest.mark.unit


def _record(case_kind: str, cells: int) -> dict:
    h = 1.0 / cells
    reference = [0.0, 1.0, 0.0]
    shape = [0.5, -1.0, 0.5]
    simulated = [value + h**2 * delta for value, delta in zip(reference, shape)]
    cut = {
        "coordinate": [-1.0, 0.0, 1.0],
        "reference": reference,
        "simulated": simulated,
        "l2_error": (0.5 * h**4) ** 0.5,
    }
    return {
        "case_kind": case_kind,
        "settings": {"ny": cells, "nz": cells},
        "benchmark_spec": {"id": f"A2-{case_kind}-ha20", "sha256": "a" * 64},
        "solver_diagnostics": {
            "potential_iterations_used": 20,
            "potential_residual": 1.0e-10,
        },
        "integral_observables": {
            "applied_pressure_gradient": 10.0 + 100.0 * h**2,
            "reference_pressure_gradient": 10.0,
        },
        "observables": {
            name: {axis: copy.deepcopy(cut) for axis in ("y", "z")}
            for name in ("velocity", "potential", "current", "lorentz")
        },
    }


def _levels() -> list[dict]:
    return [
        {
            "label": label,
            "records": [_record("shercliff", cells), _record("hunt", cells)],
        }
        for label, cells in (("coarse", 10), ("medium", 20), ("fine", 40))
    ]


def test_analyze_ladder_recovers_second_order_and_zero_extrapolated_error():
    result = analyze_ladder(_levels())

    for case in result["cases"].values():
        assert case["profiles"]["velocity_y"]["observed_order"] == pytest.approx(2.0)
        assert case["profiles"]["velocity_y"]["extrapolated_l2"] == pytest.approx(
            0.0, abs=1.0e-14
        )
        assert case["pressure_gradient"]["observed_order"] == pytest.approx(2.0)
        assert case["pressure_gradient"][
            "extrapolated_relative_error"
        ] == pytest.approx(0.0, abs=1.0e-14)
        assert case["extrapolated_primary_pass"] is True
    assert result["research_grade_validation_pass"] is True


def test_analyze_ladder_rejects_incomplete_or_inconsistent_inputs():
    with pytest.raises(ValueError, match="exactly three levels"):
        analyze_ladder(_levels()[:2])

    levels = _levels()
    levels[1]["records"][0]["benchmark_spec"]["sha256"] = "b" * 64
    with pytest.raises(ValueError, match="specification changed"):
        analyze_ladder(levels)


def test_analyze_ladder_marks_unavailable_order_without_claiming_extrapolation():
    levels = _levels()
    for level in levels:
        cut = level["records"][0]["observables"]["velocity"]["y"]
        cut["simulated"] = cut["reference"]
        cut["l2_error"] = 0.0

    profile = analyze_ladder(levels)["cases"]["shercliff"]["profiles"]["velocity_y"]
    assert profile["observed_order"] is None
    assert profile["extrapolation_status"] == "order_unavailable"
    assert profile["extrapolated_pass"] is False
