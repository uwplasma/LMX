from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path

import pytest

from scripts.benchmark_solvax_pcg_backend import run_backend_comparison
from scripts.freeze_solvax_pcg_acceptance import (
    EXPECTED_ROWS,
    build_acceptance,
    validate_ha20_evidence,
)


def _write(path: Path, payload: dict) -> Path:
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_solvax_backend_records_equivalence_and_resources() -> None:
    result = run_backend_comparison(grid=6, repeats=2, max_steps=32)
    assert result["implementation"]["solvax_version"] == version("solvax")
    for gate in (
        "forward_equivalent",
        "gradient_verified",
        "transpose_gradient_verified",
        "transpose_residual_pass",
        "end_to_end_hartmann_pass",
    ):
        assert result["acceptance"][gate] is True
    components = [
        value
        for name, value in result["acceptance"].items()
        if name not in {"backend_promotion_pass", "cpu_promotion_pass"}
    ]
    assert result["acceptance"]["backend_promotion_pass"] is all(components)
    assert result["acceptance"]["cpu_promotion_pass"] is result["acceptance"][
        "backend_promotion_pass"
    ]
    for backend in ("native", "solvax"):
        assert result[backend]["residual"] <= result["problem"]["tolerance"] + 1e-15
        assert result[backend]["warm_median_seconds"] > 0.0
        assert result[backend]["memory"]["temp_size_in_bytes"] is not None
    assert result["transpose_audit"]["residual"] <= result["problem"]["tolerance"]
    assert result["end_to_end_hartmann"]["acceptance"]["pass"] is True
    assert result["end_to_end_hartmann"]["native"]["steps"] == result[
        "end_to_end_hartmann"
    ]["solvax"]["steps"]


def _campaign(records: list[dict], *, solvax_version: str = "0.5.1") -> dict:
    return {
        "implementation": {
            "lmx_version": "1.1.3",
            "solvax_version": solvax_version,
            "solver_core_sha256": "a" * 64,
        },
        "controls": {"linear_solver": "solvax_pcg"},
        "records": records,
    }


def _table_row(case: str, ha: int, *, passed: bool) -> dict:
    return {
        "analytical_flow_rate": 1.0,
        "case_kind": case,
        "finest_level_pass": passed,
        "hartmann_number": ha,
        "hartmann_wall_conductance": 0.0,
        "levels": [
            {
                "analytical_relative_error": 0.005,
                "balances": {"current": {"pass": True}, "power": {"pass": True}},
                "mesh": [99, 99],
                "q_tilde": 1.0,
                "solver": {
                    "linear_iterations_used": 5,
                    "linear_residual": 1e-12,
                    "potential_iterations_used": 3,
                    "potential_residual": 1e-10,
                    "residual": 1e-10,
                },
            }
        ],
        "published_numerical_flow_rate": 1.0,
        "refinement": {"pass": True},
    }


def _ha20(directory: Path, implementation: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    source = "d" * 64
    common = {
        "implementation": implementation,
        "confirmation_level": "confirmation_85x63",
        "source_artifact_sha256": source,
    }
    _write(
        directory / "benchmark-a-ha20-power-balance.json",
        {
            **common,
            "cases": {
                case: {
                    "primary_errors": {"velocity_y_l2": 0.005},
                    "current_balance": {
                        "acceptance_target": 0.001,
                        "charge_balance_normalized": 1e-6,
                    },
                    "power_balance": {
                        "acceptance_target": 0.001,
                        "mechanical_power_relative_error": 1e-6,
                    },
                }
                for case in ("shercliff", "hunt")
            },
        },
    )
    _write(
        directory / "benchmark-a-ha20-continuum-reference.json",
        {
            **common,
            "cases": {
                case: {
                    "axes": {
                        axis: {
                            "lmx_raw_analytical": {"l2_error": 0.005},
                            "processed_freemhd_raw_analytical": {"l2_error": 0.01},
                        }
                        for axis in ("y", "z")
                    }
                }
                for case in ("shercliff", "hunt")
            },
        },
    )
    _write(
        directory / "benchmark-a-ha20-richardson.json",
        {
            **common,
            "cases": {
                case: {
                    "fine_primary_pass": True,
                    "extrapolated_primary_pass": case == "shercliff",
                }
                for case in ("shercliff", "hunt")
            },
        },
    )
    return directory


def test_acceptance_merges_confirmation_and_keeps_gpu_gate_open(tmp_path: Path):
    records = [
        _table_row(
            case,
            ha,
            passed=not (case == "shercliff" and ha == 15000),
        )
        for case, ha in sorted(EXPECTED_ROWS)
    ]
    confirmation = [
        _table_row("shercliff", 15000, passed=True)
    ]
    cpu = {
        "acceptance": {"backend_promotion_pass": True},
        "environment": {"backend": "cpu", "dtype": "float64"},
        "implementation": {"solvax_version": "0.5.1"},
    }
    result = build_acceptance(
        _write(tmp_path / "campaign.json", _campaign(records)),
        _write(tmp_path / "confirmation.json", _campaign(confirmation)),
        _write(tmp_path / "cpu.json", cpu),
    )
    assert result["literature_table_i"]["pass"] is True
    assert result["cpu_acceptance_pass"] is True
    assert result["gpu_equivalence"]["pass"] is False
    assert result["m3_promotion_pass"] is False


def test_acceptance_closes_with_matching_gpu_record(tmp_path: Path):
    records = [
        _table_row(case, ha, passed=True)
        for case, ha in sorted(EXPECTED_ROWS)
    ]
    implementation = {
        "benchmark_sha256": "b" * 64,
        "linear_sha256": "c" * 64,
        "lmx_version": "1.1.3",
        "solvax_version": "0.8.1",
        "solver_core_sha256": "a" * 64,
    }
    cpu = {
        "acceptance": {"backend_promotion_pass": True},
        "environment": {"backend": "cpu", "dtype": "float64"},
        "implementation": implementation,
        "problem": {"grid": [8, 8]},
    }
    gpu = {
        "acceptance": {"backend_promotion_pass": True},
        "environment": {"backend": "gpu", "dtype": "float64"},
        "implementation": implementation,
        "problem": {"grid": [8, 8]},
    }
    result = build_acceptance(
        _write(
            tmp_path / "campaign.json",
            _campaign(records, solvax_version="0.8.1"),
        ),
        _write(
            tmp_path / "confirmation.json",
            _campaign([], solvax_version="0.8.1"),
        ),
        _write(tmp_path / "cpu.json", cpu),
        _write(tmp_path / "gpu.json", gpu),
        _ha20(tmp_path / "ha20", implementation),
    )
    assert result["gpu_equivalence"]["status"] == "accepted"
    assert result["ha20_freemhd"]["status"] == "accepted"
    assert result["m3_promotion_pass"] is True
    assert result["promotion_blockers"] == []


def test_acceptance_rejects_wrong_solver_or_incomplete_rows(tmp_path: Path):
    bad = _campaign([])
    bad["controls"]["linear_solver"] = "cg"
    with pytest.raises(ValueError, match="linear_solver"):
        build_acceptance(
            _write(tmp_path / "bad.json", bad),
            _write(tmp_path / "confirm.json", bad),
            _write(tmp_path / "cpu.json", {}),
        )


def test_tracked_solvax_pcg_acceptance_is_complete_and_honest():
    payload = json.loads(
        Path("benchmarks/results/solvax-pcg-acceptance.json").read_text()
    )
    rows = payload["literature_table_i"]["records"]
    assert len(rows) == 8
    assert all(row["finest_level_pass"] for row in rows)
    assert payload["cpu_acceptance_pass"] is True
    assert payload["gpu_equivalence"]["environment"]["backend"] == "gpu"
    assert payload["gpu_equivalence"]["status"] == "accepted"
    assert payload["ha20_freemhd"]["status"] == "accepted"
    assert payload["m3_promotion_pass"] is True
    assert all(
        source["path"].startswith("https://github.com/uwplasma/LMX/releases/")
        for source in payload["sources"]
        if "equivalence" in source["path"]
    )


def test_current_solvax_equivalence_is_compact_and_cross_backend():
    path = Path("benchmarks/results/solvax-pcg-current-equivalence.json")
    payload = json.loads(path.read_text())

    assert path.stat().st_size < 4_096
    assert payload["implementation"]["solvax_version"] == "0.8.1"
    assert payload["problem"]["dtype"] == "float64"
    assert set(payload["backends"]) == {"cpu", "gpu"}
    assert payload["source_records"]["cpu_url"].startswith("https://github.com/")
    assert payload["source_records"]["gpu_url"].startswith("https://github.com/")
    assert all(payload["acceptance"].values())
    assert all(
        backend["promotion_pass"]
        and backend["native_residual"] <= payload["problem"]["tolerance"]
        and backend["solvax_residual"] <= payload["problem"]["tolerance"]
        for backend in payload["backends"].values()
    )
    assert all(
        len(source_hash) == 64
        for name, source_hash in payload["source_records"].items()
        if name.endswith("sha256")
    )


def test_ha20_validation_rejects_solver_fingerprint_mismatch(tmp_path: Path):
    implementation = {
        "solver_core_sha256": "a" * 64,
        "lmx_version": "1.1.3",
        "solvax_version": "0.5.1",
    }
    evidence = _ha20(tmp_path, implementation)
    with pytest.raises(ValueError, match="solver_core_sha256"):
        validate_ha20_evidence(
            evidence, {**implementation, "solver_core_sha256": "b" * 64}
        )
