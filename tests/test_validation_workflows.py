import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_manual_solver_family_validation as manual_validation


pytestmark = pytest.mark.unit


def _fringing_validation(**overrides):
    values = {
        "station_count": 5,
        "max_residual": 1.0e-4,
        "max_charge_balance_residual": 2.0e-6,
        "mean_velocity_span": 0.1,
        "volumetric_flow_rate_span": 2.0e-3,
        "axial_current_span": 0.01,
        "peak_velocity_span": 0.03,
        "pressure_span_range": 0.04,
        "max_wall_current_leakage": 1.0e-6,
        "net_boundary_current_residual": 2.0e-6,
        "field_mean_velocity_correlation": -0.8,
    }
    return SimpleNamespace(**(values | overrides))


@pytest.fixture
def stub_validation(monkeypatch: pytest.MonkeyPatch):
    state = {
        "summary": {
            "u_max": 1.0,
            "charge_balance_residual": 1.0e-7,
            "interface_current_residual": 1.0e-7,
        },
        "fringing": _fringing_validation(),
    }
    monkeypatch.setattr(
        manual_validation,
        "solve_steady",
        lambda case: SimpleNamespace(mesh=object(), case_name=case.name),
    )
    monkeypatch.setattr(
        manual_validation,
        "validation_summary",
        lambda solution, case_name, ha=None: {"case": case_name, **state["summary"]},
    )
    monkeypatch.setattr(
        manual_validation,
        "duct_layer_resolution_metrics",
        lambda case, mesh: {"hartmann_layer_cells": 3.0},
    )
    monkeypatch.setattr(
        manual_validation,
        "hartmann_acceptance",
        lambda solution, ha, l2_threshold, linf_threshold: SimpleNamespace(
            passed=True, l2_error=0.01, linf_error=0.02
        ),
    )
    monkeypatch.setattr(
        manual_validation,
        "closed_channel_validation",
        lambda solution, case_kind, ha, reference_root=None: SimpleNamespace(
            y_profile=SimpleNamespace(l2_error=0.03, linf_error=0.04),
            z_profile=SimpleNamespace(l2_error=0.05, linf_error=0.06),
        ),
    )
    monkeypatch.setattr(
        manual_validation,
        "solve_extruded_inductionless",
        lambda problem: SimpleNamespace(validation=state["fringing"]),
    )
    return state


def test_main_writes_manual_solver_family_summary(
    tmp_path: Path, stub_validation
):
    output = tmp_path / "manual_summary.json"
    exit_code = manual_validation.main([
        "--output", str(output), "--ha-values", "10", "--resolution", "8",
        "--reference-root", str(tmp_path / "refs"), "--include-fringing",
    ])

    assert exit_code == 0
    payload = json.loads(output.read_text())
    for case in (
        "hartmann_ha10", "shercliff_ha10", "hunt_ha10",
        "fringing_rect_duct_ha10", "fringing_layered_duct_ha10",
        "fringing_pipe_ogrid_ha10",
    ):
        assert case in payload
    assert payload["hartmann_ha10"]["conservation_pass"] == pytest.approx(1.0)
    fringing = payload["fringing_rect_duct_ha10"]
    assert fringing["conservation_pass"] == pytest.approx(1.0)
    assert fringing["physics_pass"] == pytest.approx(1.0)
    assert fringing["validation_pass"] == pytest.approx(1.0)


def test_main_can_fail_on_conservation_threshold(tmp_path: Path, stub_validation):
    stub_validation["summary"].update(
        charge_balance_residual=1.0e-2, interface_current_residual=1.0e-2
    )
    output = tmp_path / "manual_summary_fail.json"
    exit_code = manual_validation.main([
        "--output", str(output), "--ha-values", "10", "--resolution", "8",
        "--max-charge-balance", "1e-3", "--max-interface-current", "1e-3",
        "--fail-on-threshold",
    ])

    assert exit_code == 1


def test_main_writes_multi_resolution_csv_and_plot(tmp_path: Path, stub_validation):
    output = tmp_path / "manual_summary_multi.json"
    exit_code = manual_validation.main([
        "--output", str(output), "--ha-values", "10", "--resolutions", "8,10",
        "--include-fringing", "--write-csv", "--write-plot",
    ])

    assert exit_code == 0
    payload = json.loads(output.read_text())
    assert "hartmann_ha10_n8" in payload
    assert "fringing_rect_duct_ha10_n10" in payload
    assert output.with_suffix(".csv").exists()
    assert output.with_name("manual_summary_multi_fringing.png").exists()


def test_main_can_fail_on_fringing_physics_thresholds(
    tmp_path: Path, stub_validation
):
    stub_validation["fringing"] = _fringing_validation(
        volumetric_flow_rate_span=0.2,
        field_mean_velocity_correlation=-0.1,
    )
    output = tmp_path / "manual_summary_physics_fail.json"
    exit_code = manual_validation.main([
        "--output", str(output), "--ha-values", "10", "--resolution", "8",
        "--include-fringing", "--max-fringing-flow-span", "1e-2",
        "--max-field-velocity-correlation", "-0.5", "--fail-on-threshold",
    ])

    assert exit_code == 1
    payload = json.loads(output.read_text())["fringing_rect_duct_ha10"]
    assert payload["conservation_pass"] == pytest.approx(1.0)
    assert payload["physics_pass"] == pytest.approx(0.0)
    assert payload["validation_pass"] == pytest.approx(0.0)
