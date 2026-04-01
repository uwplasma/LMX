from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_time_convergence_suite as suite


pytestmark = pytest.mark.unit


def test_parse_csv_floats():
    assert suite._parse_csv_floats("0.002, 0.001,0.0005") == [0.002, 0.001, 0.0005]


def test_observed_orders_reports_first_order_drop():
    levels = [
        {"dt": 2.0e-3, "y_l2_error": 4.0e-2},
        {"dt": 1.0e-3, "y_l2_error": 2.0e-2},
    ]
    orders = suite._observed_orders(levels)
    assert orders["y_l2_error"][0]["order"] == pytest.approx(1.0)


def test_run_time_convergence_suite_writes_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    output = tmp_path / "time_convergence"

    monkeypatch.setattr(
        suite.argparse.ArgumentParser,
        "parse_args",
        lambda self, argv=None: SimpleNamespace(
            output=output,
            cases="hartmann,hunt",
            ha=20.0,
            resolution=24,
            dts="0.002,0.001",
            reference_root=tmp_path / "refs",
            x_slice="1m",
            t_final=None,
            hartmann_l2_threshold=0.05,
            hartmann_linf_threshold=0.1,
        ),
    )
    monkeypatch.setattr(
        suite,
        "_build_case",
        lambda case_kind, ha, resolution, output_dir: SimpleNamespace(
            name=f"{case_kind}_ha{int(ha)}",
            time_stepper=SimpleNamespace(dt=0.01, t_final=1.0, max_steps=100),
        ),
    )
    monkeypatch.setattr(suite, "solve_steady", lambda case: SimpleNamespace(mesh=object()))
    monkeypatch.setattr(suite, "duct_layer_resolution_metrics", lambda case, mesh: {"hartmann_layer_cells": 6.0})
    monkeypatch.setattr(
        suite,
        "_collect_metrics",
        lambda solution, case_kind, ha, **kwargs: (
            {"l2_error": 0.04} if case_kind == "hartmann" else {"y_l2_error": 0.2, "z_l2_error": 0.1, "combined_l2_error": 0.158}
        ),
    )

    exit_code = suite.main([])

    assert exit_code == 0
    summary = (output / "summary.json").read_text()
    assert '"hartmann"' in summary
    assert '"hunt"' in summary
    assert '"hartmann_layer_cells": 6.0' in summary
    assert '"combined_l2_error": 0.158' in summary
    assert '"dt": 0.002' in summary
    assert '"cases"' in capsys.readouterr().out
