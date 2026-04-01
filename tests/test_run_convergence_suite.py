from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_convergence_suite as suite


pytestmark = pytest.mark.unit


def test_parse_csv_numbers():
    assert suite._parse_csv_numbers("16, 32,48") == [16, 32, 48]


def test_observed_orders_reports_second_order_drop():
    levels = [
        {"resolution": 16.0, "mesh_spacing": 0.25, "l2_error": 4.0e-2},
        {"resolution": 32.0, "mesh_spacing": 0.125, "l2_error": 1.0e-2},
    ]
    orders = suite._observed_orders(levels)
    assert orders["l2_error"][0]["order"] == pytest.approx(2.0)


def test_run_convergence_suite_writes_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    output = tmp_path / "convergence"

    monkeypatch.setattr(
        suite.argparse.ArgumentParser,
        "parse_args",
        lambda self, argv=None: SimpleNamespace(
            output=output,
            cases="hartmann,shercliff",
            ha=20.0,
            resolutions="16,32",
            reference_root=tmp_path / "refs",
            x_slice="1m",
            hartmann_l2_threshold=0.05,
            hartmann_linf_threshold=0.1,
        ),
    )
    monkeypatch.setattr(suite, "_build_case", lambda case_kind, ha, resolution, output_dir: SimpleNamespace(
        name=f"{case_kind}_ha{int(ha)}",
        geometry=SimpleNamespace(width=2.0, height=2.0, ny=resolution, nz=resolution),
        time_stepper=SimpleNamespace(dt=0.001, max_steps=100),
    ))
    monkeypatch.setattr(suite, "solve_steady", lambda case: SimpleNamespace())
    monkeypatch.setattr(
        suite,
        "_collect_metrics",
        lambda solution, case_kind, ha, **kwargs: (
            {"l2_error": 4.0e-2 if case_kind == "hartmann" and kwargs else 1.0e-2}
            if case_kind == "hartmann"
            else {"y_l2_error": 0.2, "z_l2_error": 0.1}
        ),
    )

    exit_code = suite.main([])

    assert exit_code == 0
    summary = (output / "summary.json").read_text()
    assert '"hartmann"' in summary
    assert '"shercliff"' in summary
    assert '"cases"' in capsys.readouterr().out
