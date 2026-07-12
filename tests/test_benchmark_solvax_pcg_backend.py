from __future__ import annotations

from scripts.benchmark_solvax_pcg_backend import run_backend_comparison


def test_solvax_pcg_backend_benchmark_records_equivalence_and_resources():
    result = run_backend_comparison(grid=6, repeats=2, max_steps=32)
    assert result["implementation"]["solvax_version"] == "0.7.0"
    assert result["acceptance"]["forward_equivalent"] is True
    assert result["acceptance"]["gradient_verified"] is True
    assert result["acceptance"]["transpose_gradient_verified"] is True
    assert result["acceptance"]["transpose_residual_pass"] is True
    assert result["acceptance"]["end_to_end_hartmann_pass"] is True
    # Timing and compiler-memory acceptance is evaluated by the isolated CPU/GPU
    # evidence lanes. Parallel coverage workers intentionally share resources,
    # so this portable test verifies that the aggregate is computed correctly
    # without turning scheduler contention into a flaky numerical failure.
    component_values = [
        value
        for name, value in result["acceptance"].items()
        if name not in {"backend_promotion_pass", "cpu_promotion_pass"}
    ]
    assert result["acceptance"]["backend_promotion_pass"] is all(component_values)
    assert (
        result["acceptance"]["cpu_promotion_pass"]
        is result["acceptance"]["backend_promotion_pass"]
    )
    assert result["native"]["residual"] <= result["problem"]["tolerance"] + 1.0e-15
    assert result["solvax"]["residual"] <= result["problem"]["tolerance"] + 1.0e-15
    assert result["native"]["warm_median_seconds"] > 0.0
    assert result["solvax"]["warm_median_seconds"] > 0.0
    assert result["native"]["memory"]["temp_size_in_bytes"] is not None
    assert result["solvax"]["memory"]["temp_size_in_bytes"] is not None
    assert result["transpose_audit"]["residual"] <= result["problem"]["tolerance"]
    hartmann = result["end_to_end_hartmann"]
    assert hartmann["acceptance"]["pass"] is True
    assert hartmann["native"]["steps"] == hartmann["solvax"]["steps"]
