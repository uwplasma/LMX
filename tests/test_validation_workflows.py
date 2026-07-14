from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from scripts import run_validation_suite as suite
from scripts import run_manual_solver_family_validation as manual_validation


pytestmark = pytest.mark.unit


def test_run_validation_suite_writes_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    output = tmp_path / "artifacts"
    case = SimpleNamespace(
        name="hartmann_ha5", output=SimpleNamespace(directory=str(output / "hartmann"))
    )

    monkeypatch.setattr(
        suite.argparse.ArgumentParser,
        "parse_args",
        lambda self, argv=None: SimpleNamespace(
            output=output,
            ha=5.0,
            resolution=12,
            cases="hartmann,shercliff,hunt",
            reference_root=None,
            x_slice="1m",
            hartmann_l2_threshold=0.05,
            hartmann_linf_threshold=0.1,
            skip_paraview=False,
        ),
    )
    monkeypatch.setattr(suite, "make_hartmann_case", lambda **kwargs: case)
    monkeypatch.setattr(
        suite,
        "make_shercliff_case",
        lambda **kwargs: SimpleNamespace(
            name="shercliff_ha5",
            output=SimpleNamespace(directory=str(output / "shercliff")),
        ),
    )
    monkeypatch.setattr(
        suite,
        "make_hunt_case",
        lambda **kwargs: SimpleNamespace(
            name="hunt_ha5", output=SimpleNamespace(directory=str(output / "hunt"))
        ),
    )
    monkeypatch.setattr(
        suite,
        "solve_steady",
        lambda built_case: SimpleNamespace(
            state=SimpleNamespace(time=0.0, residual=0.1), mesh=SimpleNamespace()
        ),
    )
    monkeypatch.setattr(suite, "write_paraview", lambda *args, **kwargs: [])
    monkeypatch.setattr(suite, "write_profile_csv", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        suite, "extract_centerline", lambda solved: {"y": [0.0], "u": [1.0]}
    )
    monkeypatch.setattr(
        suite, "extract_midplane_profile", lambda solved, axis: {"z": [0.0], "u": [1.0]}
    )
    monkeypatch.setattr(
        suite,
        "validation_summary",
        lambda solved, case_name, ha: {
            "case": case_name,
            "residual": 0.1,
            "u_max": 1.0,
        },
    )
    monkeypatch.setattr(
        suite,
        "hartmann_validation",
        lambda solved, ha: SimpleNamespace(
            y_profile=SimpleNamespace(l2_error=0.0, linf_error=0.0)
        ),
    )
    monkeypatch.setattr(
        suite,
        "hartmann_acceptance",
        lambda solved, ha, l2_threshold, linf_threshold: SimpleNamespace(
            passed=True,
            l2_threshold=l2_threshold,
            linf_threshold=linf_threshold,
        ),
    )
    monkeypatch.setattr(
        suite, "write_analytic_comparison", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(suite, "write_acceptance_report", lambda *args, **kwargs: None)
    monkeypatch.setattr(suite, "write_metrics_json", lambda *args, **kwargs: None)

    exit_code = suite.main()

    summary = (output / "summary.json").read_text()
    assert exit_code == 0
    assert summary.count('"case"') == 3
    assert '"hartmann_ha5"' in summary
    assert '"shercliff_ha5"' in summary
    assert '"hunt_ha5"' in summary
    assert '"hartmann_ha5"' in capsys.readouterr().out


def test_run_validation_suite_handles_reference_branch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    output = tmp_path / "artifacts"
    case = SimpleNamespace(
        name="hunt_ha5", output=SimpleNamespace(directory=str(output / "hunt"))
    )

    monkeypatch.setattr(
        suite.argparse.ArgumentParser,
        "parse_args",
        lambda self, argv=None: SimpleNamespace(
            output=output,
            ha=5.0,
            resolution=12,
            cases="hartmann,shercliff,hunt",
            reference_root=tmp_path / "refs",
            x_slice="1m",
            hartmann_l2_threshold=0.05,
            hartmann_linf_threshold=0.1,
            skip_paraview=False,
        ),
    )
    monkeypatch.setattr(
        suite,
        "make_hartmann_case",
        lambda **kwargs: SimpleNamespace(
            name="hartmann_ha5",
            output=SimpleNamespace(directory=str(output / "hartmann")),
        ),
    )
    monkeypatch.setattr(
        suite,
        "make_shercliff_case",
        lambda **kwargs: SimpleNamespace(
            name="shercliff_ha5",
            output=SimpleNamespace(directory=str(output / "shercliff")),
        ),
    )
    monkeypatch.setattr(suite, "make_hunt_case", lambda **kwargs: case)
    monkeypatch.setattr(
        suite,
        "solve_steady",
        lambda built_case: SimpleNamespace(
            state=SimpleNamespace(time=0.0, residual=0.1), mesh=SimpleNamespace()
        ),
    )
    monkeypatch.setattr(suite, "write_paraview", lambda *args, **kwargs: [])
    monkeypatch.setattr(suite, "write_profile_csv", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        suite, "extract_centerline", lambda solved: {"y": [0.0], "u": [1.0]}
    )
    monkeypatch.setattr(
        suite, "extract_midplane_profile", lambda solved, axis: {"z": [0.0], "u": [1.0]}
    )
    monkeypatch.setattr(
        suite,
        "validation_summary",
        lambda solved, case_name, ha: {
            "case": case_name,
            "residual": 0.1,
            "u_max": 1.0,
        },
    )
    monkeypatch.setattr(
        suite,
        "hartmann_validation",
        lambda solved, ha: SimpleNamespace(
            y_profile=SimpleNamespace(l2_error=0.0, linf_error=0.0)
        ),
    )
    monkeypatch.setattr(
        suite,
        "hartmann_acceptance",
        lambda solved, ha, l2_threshold, linf_threshold: SimpleNamespace(
            passed=True,
            l2_threshold=l2_threshold,
            linf_threshold=linf_threshold,
        ),
    )
    monkeypatch.setattr(
        suite, "write_analytic_comparison", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(suite, "write_acceptance_report", lambda *args, **kwargs: None)
    monkeypatch.setattr(suite, "write_metrics_json", lambda *args, **kwargs: None)
    comparison = SimpleNamespace(
        y_profile=SimpleNamespace(l2_error=0.2, linf_error=0.3),
        z_profile=SimpleNamespace(l2_error=0.4, linf_error=0.5),
    )
    monkeypatch.setattr(
        suite, "closed_channel_validation", lambda *args, **kwargs: comparison
    )
    monkeypatch.setattr(
        suite, "processed_slice_validation", lambda *args, **kwargs: comparison
    )
    monkeypatch.setattr(
        suite, "write_closed_channel_validation", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(
        suite, "write_processed_slice_validation", lambda *args, **kwargs: None
    )

    exit_code = suite.main()

    summary = (output / "summary.json").read_text()
    assert exit_code == 0
    assert '"y_l2_error": 0.2' in summary
    assert '"combined_l2_error"' in summary
    assert '"slice_y_l2_error": 0.2' in summary
    assert '"slice_combined_l2_error"' in summary

def test_main_writes_manual_solver_family_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
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
    monkeypatch.setattr(
        manual_validation,
        "duct_layer_resolution_metrics",
        lambda case, mesh: {"hartmann_layer_cells": 3.0},
    )
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
    monkeypatch.setattr(
        manual_validation,
        "solve_extruded_inductionless",
        lambda problem: type(
            "ExtrudedSolution",
            (),
            {
                "validation": type(
                    "Validation",
                    (),
                    {
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
                    },
                )(),
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
            "--include-fringing",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text())
    assert "hartmann_ha10" in payload
    assert "shercliff_ha10" in payload
    assert "hunt_ha10" in payload
    assert "fringing_rect_duct_ha10" in payload
    assert "fringing_layered_duct_ha10" in payload
    assert "fringing_pipe_ogrid_ha10" in payload
    assert payload["hartmann_ha10"]["conservation_pass"] == pytest.approx(1.0)
    assert payload["fringing_rect_duct_ha10"]["conservation_pass"] == pytest.approx(1.0)
    assert payload["fringing_rect_duct_ha10"]["physics_pass"] == pytest.approx(1.0)
    assert payload["fringing_rect_duct_ha10"]["validation_pass"] == pytest.approx(1.0)


def test_main_can_fail_on_conservation_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        manual_validation,
        "solve_steady",
        lambda case: type("Solution", (), {"mesh": object(), "case_name": case.name})(),
    )
    monkeypatch.setattr(
        manual_validation,
        "validation_summary",
        lambda solution, case_name, ha=None: {
            "case": case_name,
            "u_max": 1.0,
            "charge_balance_residual": 1.0e-2,
            "interface_current_residual": 1.0e-2,
        },
    )
    monkeypatch.setattr(
        manual_validation, "duct_layer_resolution_metrics", lambda case, mesh: {}
    )
    monkeypatch.setattr(
        manual_validation,
        "hartmann_acceptance",
        lambda solution, ha, l2_threshold, linf_threshold: type(
            "Acceptance",
            (),
            {"passed": True, "l2_error": 0.01, "linf_error": 0.02},
        )(),
    )

    output = tmp_path / "manual_summary_fail.json"
    exit_code = manual_validation.main(
        [
            "--output",
            str(output),
            "--ha-values",
            "10",
            "--resolution",
            "8",
            "--max-charge-balance",
            "1e-3",
            "--max-interface-current",
            "1e-3",
            "--fail-on-threshold",
        ]
    )

    assert exit_code == 1


def test_main_writes_multi_resolution_csv_and_plot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        manual_validation,
        "solve_steady",
        lambda case: type("Solution", (), {"mesh": object(), "case_name": case.name})(),
    )
    monkeypatch.setattr(
        manual_validation,
        "validation_summary",
        lambda solution, case_name, ha=None: {
            "case": case_name,
            "u_max": 1.0,
            "charge_balance_residual": 1.0e-7,
            "interface_current_residual": 1.0e-7,
        },
    )
    monkeypatch.setattr(
        manual_validation, "duct_layer_resolution_metrics", lambda case, mesh: {}
    )
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
        "solve_extruded_inductionless",
        lambda problem: type(
            "ExtrudedSolution",
            (),
            {
                "validation": type(
                    "Validation",
                    (),
                    {
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
                    },
                )(),
            },
        )(),
    )

    output = tmp_path / "manual_summary_multi.json"
    exit_code = manual_validation.main(
        [
            "--output",
            str(output),
            "--ha-values",
            "10",
            "--resolutions",
            "8,10",
            "--include-fringing",
            "--write-csv",
            "--write-plot",
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text())
    assert "hartmann_ha10_n8" in payload
    assert "fringing_rect_duct_ha10_n10" in payload
    assert output.with_suffix(".csv").exists()
    assert output.with_name("manual_summary_multi_fringing.png").exists()


def test_main_can_fail_on_fringing_physics_thresholds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(
        manual_validation,
        "solve_steady",
        lambda case: type("Solution", (), {"mesh": object(), "case_name": case.name})(),
    )
    monkeypatch.setattr(
        manual_validation,
        "validation_summary",
        lambda solution, case_name, ha=None: {
            "case": case_name,
            "u_max": 1.0,
            "charge_balance_residual": 1.0e-7,
            "interface_current_residual": 1.0e-7,
        },
    )
    monkeypatch.setattr(
        manual_validation, "duct_layer_resolution_metrics", lambda case, mesh: {}
    )
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
        "solve_extruded_inductionless",
        lambda problem: type(
            "ExtrudedSolution",
            (),
            {
                "validation": type(
                    "Validation",
                    (),
                    {
                        "station_count": 5,
                        "max_residual": 1.0e-4,
                        "max_charge_balance_residual": 2.0e-6,
                        "mean_velocity_span": 0.1,
                        "volumetric_flow_rate_span": 0.2,
                        "axial_current_span": 0.01,
                        "peak_velocity_span": 0.03,
                        "pressure_span_range": 0.04,
                        "max_wall_current_leakage": 1.0e-6,
                        "net_boundary_current_residual": 2.0e-6,
                        "field_mean_velocity_correlation": -0.1,
                    },
                )(),
            },
        )(),
    )

    output = tmp_path / "manual_summary_physics_fail.json"
    exit_code = manual_validation.main(
        [
            "--output",
            str(output),
            "--ha-values",
            "10",
            "--resolution",
            "8",
            "--include-fringing",
            "--max-fringing-flow-span",
            "1e-2",
            "--max-field-velocity-correlation",
            "-0.5",
            "--fail-on-threshold",
        ]
    )

    assert exit_code == 1
    payload = json.loads(output.read_text())
    assert payload["fringing_rect_duct_ha10"]["conservation_pass"] == pytest.approx(1.0)
    assert payload["fringing_rect_duct_ha10"]["physics_pass"] == pytest.approx(0.0)
    assert payload["fringing_rect_duct_ha10"]["validation_pass"] == pytest.approx(0.0)
