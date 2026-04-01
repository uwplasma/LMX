from pathlib import Path
import json

import pytest

from scripts import compare_hunt_trace_histories as compare


pytestmark = pytest.mark.unit


def test_compare_trace_histories_aligns_pressure_and_epot_signals(tmp_path: Path):
    freemhd = {
        "records": [
            {"kind": "epot", "time": 0.1, "maxJ": 10.0, "maxJxB": 5.0},
            {"kind": "pressure", "time": 0.1, "corr": 0, "maxU": 2.0},
            {"kind": "pressure", "time": 0.1, "corr": 2, "maxU": 1.0},
            {"kind": "epot", "time": 0.2, "maxJ": 8.0, "maxJxB": 4.0},
            {"kind": "pressure", "time": 0.2, "corr": 2, "maxU": 0.5},
        ]
    }
    lmx = {
        "lmx_solver": {
            "trace": {
                "time_history": [0.1, 0.2],
                "u_max_history": [1.0, 0.5],
                "current_max_history": [4.0, 3.2],
                "lorentz_max_history": [2.0, 1.6],
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
    assert payload["current_max"]["l2_error"] == pytest.approx(0.0)
    assert payload["lorentz_max"]["l2_error"] == pytest.approx(0.0)
    assert payload["u_max"]["samples"][0]["freemhd_raw"] == pytest.approx(1.0)
    assert payload["u_max"]["samples"][1]["lmx_raw"] == pytest.approx(0.5)


def test_main_writes_alignment_json(tmp_path: Path):
    freemhd_path = tmp_path / "freemhd.json"
    freemhd_path.write_text(json.dumps({"records": [{"kind": "epot", "time": 0.1, "maxJ": 1.0, "maxJxB": 2.0}, {"kind": "pressure", "time": 0.1, "corr": 0, "maxU": 3.0}]}))
    lmx_path = tmp_path / "lmx.json"
    lmx_path.write_text(
        json.dumps(
            {
                "lmx_solver": {
                    "trace": {
                        "time_history": [0.1],
                        "u_max_history": [3.0],
                        "current_max_history": [1.0],
                        "lorentz_max_history": [2.0],
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
