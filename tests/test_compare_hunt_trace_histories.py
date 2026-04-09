from pathlib import Path
import json

import pytest

from scripts import compare_hunt_trace_histories as compare


pytestmark = pytest.mark.unit


def test_compare_trace_histories_aligns_pressure_and_epot_signals(tmp_path: Path):
    freemhd = {
        "records": [
            {
                "kind": "epot",
                "time": 0.1,
                "maxJ": 10.0,
                "maxJn": 7.0,
                "maxJnDensity": 7.0,
                "maxPsiub": 3.0,
                "maxPsiubDensity": 3.0,
                "maxCenteredJxB": 2.0,
                "maxJxB": 5.0,
            },
            {"kind": "pressure", "time": 0.1, "corr": 0, "maxU": 2.0, "pIterations": 1, "maxP": 9.0, "minP": 1.0, "pSpan": 8.0},
            {"kind": "pressure", "time": 0.1, "corr": 2, "maxU": 1.0, "pIterations": 2, "maxP": 9.0, "minP": 1.0, "pSpan": 8.0},
            {
                "kind": "epot",
                "time": 0.2,
                "maxJ": 8.0,
                "maxJn": 5.6,
                "maxJnDensity": 5.6,
                "maxPsiub": 2.4,
                "maxPsiubDensity": 2.4,
                "maxCenteredJxB": 1.6,
                "maxJxB": 4.0,
            },
            {"kind": "pressure", "time": 0.2, "corr": 2, "maxU": 0.5, "pIterations": 3, "maxP": 4.5, "minP": 0.5, "pSpan": 4.0},
        ]
    }
    lmx = {
                "lmx_solver": {
                    "trace": {
                        "time_history": [0.1, 0.2],
                        "u_max_history": [1.0, 0.5],
                        "mean_velocity_history": [1.0, 0.5],
                        "applied_forcing_history": [9.0, 4.5],
                        "pressure_proxy_history": [8.0, 4.0],
                        "current_scaled_pressure_proxy_history": [8.0, 4.0],
                        "current_max_history": [4.0, 3.2],
                "face_current_max_history": [7.0, 5.6],
                "emf_max_history": [3.0, 2.4],
                "lorentz_max_history": [2.0, 1.6],
                "face_lorentz_max_history": [5.0, 4.0],
                "residual_history": [],
                "potential_residual_history": [],
                "potential_iterations_history": [],
            }
        }
    }
    freemhd_path = tmp_path / "freemhd.json"
    lmx_path = tmp_path / "lmx.json"
    freemhd_path.write_text(json.dumps(freemhd))
    lmx_path.write_text(json.dumps(lmx))

    payload = compare.compare_trace_histories(freemhd_path, lmx_path)

    assert payload["u_max"]["l2_error"] == pytest.approx(0.0)
    assert payload["mean_velocity"]["l2_error"] == pytest.approx(0.0)
    assert payload["applied_forcing"]["l2_error"] == pytest.approx(0.0)
    assert payload["pressure_proxy"]["l2_error"] == pytest.approx(0.0)
    assert payload["current_scaled_pressure_proxy"]["l2_error"] == pytest.approx(0.0)
    assert payload["primary_pressure_metric"] == "pSpan"
    assert payload["primary_pressure_proxy_metric"] == "pressure_proxy"
    assert payload["primary_pressure_proxy"]["l2_error"] == pytest.approx(0.0)
    assert payload["current_max"]["l2_error"] == pytest.approx(0.0)
    assert payload["face_current_max"]["l2_error"] == pytest.approx(0.0)
    assert payload["face_current_density_max"]["l2_error"] == pytest.approx(0.0)
    assert payload["primary_current_metric"] == "face_current_density_max"
    assert payload["primary_current_max"]["l2_error"] == pytest.approx(0.0)
    assert payload["emf_max"]["l2_error"] == pytest.approx(0.0)
    assert payload["emf_density_max"]["l2_error"] == pytest.approx(0.0)
    assert payload["centered_lorentz_max"]["l2_error"] == pytest.approx(0.0)
    assert payload["lorentz_max"]["l2_error"] == pytest.approx(0.0)
    assert payload["face_lorentz_max"]["l2_error"] == pytest.approx(0.0)
    assert payload["primary_lorentz_metric"] == "centered_lorentz_max"
    assert payload["primary_lorentz_max"]["l2_error"] == pytest.approx(0.0)
    assert payload["current_max"]["mean_raw_relative_error"] == pytest.approx(0.6)
    assert payload["u_max"]["samples"][0]["freemhd_raw"] == pytest.approx(1.0)
    assert payload["u_max"]["samples"][1]["lmx_raw"] == pytest.approx(0.5)
    assert payload["freemhd_pressure_final_records"][0]["maxU"] == pytest.approx(1.0)
    assert payload["freemhd_epot_records"][1]["maxJnDensity"] == pytest.approx(5.6)


def test_main_writes_alignment_json(tmp_path: Path):
    freemhd_path = tmp_path / "freemhd.json"
    freemhd_path.write_text(json.dumps({"records": [{"kind": "epot", "time": 0.1, "maxJ": 1.0, "maxCenteredJxB": 2.0, "maxJxB": 2.0}, {"kind": "pressure", "time": 0.1, "corr": 0, "maxU": 3.0, "maxP": 4.0, "minP": 1.0, "pSpan": 3.0}]}))
    lmx_path = tmp_path / "lmx.json"
    lmx_path.write_text(
        json.dumps(
            {
                "lmx_solver": {
                    "trace": {
                        "time_history": [0.1],
                        "u_max_history": [3.0],
                        "mean_velocity_history": [3.0],
                        "applied_forcing_history": [4.0],
                        "pressure_proxy_history": [3.0],
                        "current_scaled_pressure_proxy_history": [3.0],
                        "current_max_history": [1.0],
                        "face_current_max_history": [0.5],
                        "emf_max_history": [0.25],
                        "lorentz_max_history": [2.0],
                        "face_lorentz_max_history": [2.0],
                        "residual_history": [],
                        "potential_residual_history": [],
                        "potential_iterations_history": [],
                    }
                }
            }
        )
    )
    output = tmp_path / "alignment.json"

    exit_code = compare.main(
        [
            "--freemhd-diag-json",
            str(freemhd_path),
            "--lmx-report-json",
            str(lmx_path),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text())
    assert payload["u_max"]["samples"][0]["abs_diff"] == pytest.approx(0.0)
    assert payload["pressure_proxy"]["samples"][0]["abs_diff"] == pytest.approx(0.0)
    assert payload["current_scaled_pressure_proxy"]["samples"][0]["abs_diff"] == pytest.approx(0.0)
    assert payload["primary_pressure_metric"] == "pSpan"
    assert payload["primary_pressure_proxy_metric"] == "pressure_proxy"
    assert payload["primary_current_metric"] == "current_max"
    assert payload["primary_current_max"]["samples"][0]["abs_diff"] == pytest.approx(0.0)
    assert payload["primary_lorentz_metric"] == "centered_lorentz_max"
    assert payload["primary_lorentz_max"]["samples"][0]["abs_diff"] == pytest.approx(0.0)
    assert payload["freemhd_pressure_final_records"][0]["time"] == pytest.approx(0.1)


def test_compare_trace_histories_tolerates_partial_live_logs(tmp_path: Path):
    freemhd_path = tmp_path / "freemhd.json"
    freemhd_path.write_text(
        json.dumps(
            {
                "records": [
                    {
                        "kind": "epot",
                        "time": 0.1,
                        "maxJ": 10.0,
                        "maxJn": 7.0,
                        "maxJnDensity": 7.0,
                        "maxPsiub": 3.0,
                        "maxPsiubDensity": 3.0,
                        "maxCenteredJxB": 5.0,
                        "maxJxB": 5.0,
                    }
                ]
            }
        )
    )
    lmx_path = tmp_path / "lmx.json"
    lmx_path.write_text(
        json.dumps(
            {
                "lmx_solver": {
                    "trace": {
                        "time_history": [0.1],
                        "u_max_history": [3.0],
                        "mean_velocity_history": [3.0],
                        "applied_forcing_history": [4.0],
                        "pressure_proxy_history": [4.0],
                        "current_scaled_pressure_proxy_history": [4.0],
                        "current_max_history": [10.0],
                        "face_current_max_history": [7.0],
                        "emf_max_history": [3.0],
                        "lorentz_max_history": [5.0],
                        "face_lorentz_max_history": [5.0],
                        "residual_history": [],
                        "potential_residual_history": [],
                        "potential_iterations_history": [],
                    }
                }
            }
        )
    )

    payload = compare.compare_trace_histories(freemhd_path, lmx_path)

    assert "u_max" not in payload
    assert payload["current_max"]["l2_error"] == pytest.approx(0.0)
    assert payload["face_current_max"]["l2_error"] == pytest.approx(0.0)
    assert payload["face_current_density_max"]["l2_error"] == pytest.approx(0.0)
    assert payload["primary_current_metric"] == "face_current_density_max"
    assert payload["emf_max"]["l2_error"] == pytest.approx(0.0)
    assert payload["emf_density_max"]["l2_error"] == pytest.approx(0.0)
    assert payload["face_lorentz_max"]["l2_error"] == pytest.approx(0.0)
    assert payload["centered_lorentz_max"]["l2_error"] == pytest.approx(0.0)
    assert payload["primary_lorentz_metric"] == "centered_lorentz_max"
    assert payload["face_current_max"]["max_raw_relative_error"] == pytest.approx(0.0)
    assert payload["freemhd_pressure_final_records"] == []
