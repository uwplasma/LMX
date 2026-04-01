from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_solver_control_sweep as suite


pytestmark = pytest.mark.unit


def test_parse_values_float_and_int():
    assert suite._parse_values("0.1,0.2", "float") == [0.1, 0.2]
    assert suite._parse_values("2,4,6", "int") == [2, 4, 6]


def test_run_solver_control_sweep_accepts_potential_tolerance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "control_sweep_tolerance"

    monkeypatch.setattr(
        suite.argparse.ArgumentParser,
        "parse_args",
        lambda self, argv=None: SimpleNamespace(
            output=output,
            case="hartmann",
            ha=20.0,
            resolution=24,
            wall_cells=None,
            parameter="potential_tolerance",
            values="0.01,0.001",
            value_type="float",
            reference_root=None,
            x_slice="1m",
        ),
    )
    monkeypatch.setattr(
        suite,
        "_build_case",
        lambda case_kind, ha, resolution, output_dir, wall_cells: SimpleNamespace(
            time_stepper=SimpleNamespace(
                dt=0.001,
                t_final=1.0,
                max_steps=400,
                outer_iterations=2,
                potential_iterations=200,
                potential_tolerance=None,
                relaxation=0.1,
            ),
        ),
    )
    monkeypatch.setattr(suite, "solve_steady", lambda case: SimpleNamespace(mesh=object()))
    monkeypatch.setattr(suite, "duct_layer_resolution_metrics", lambda case, mesh: {"hartmann_layer_cells": 6.0})
    monkeypatch.setattr(
        suite,
        "_collect_metrics",
        lambda solution, case_kind, ha, **kwargs: {"l2_error": 0.2, "u_max": 0.1, "residual": 1e-4},
    )

    exit_code = suite.main([])

    assert exit_code == 0
    summary = (output / "summary.json").read_text()
    assert '"parameter": "potential_tolerance"' in summary
    assert '"parameter_value": 0.01' in summary


def test_run_solver_control_sweep_writes_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    output = tmp_path / "control_sweep"

    monkeypatch.setattr(
        suite.argparse.ArgumentParser,
        "parse_args",
        lambda self, argv=None: SimpleNamespace(
            output=output,
            case="hunt",
            ha=20.0,
            resolution=24,
            wall_cells=4,
            parameter="outer_iterations",
            values="2,4",
            value_type="int",
            reference_root=tmp_path / "refs",
            x_slice="1m",
        ),
    )
    monkeypatch.setattr(
        suite,
        "_build_case",
        lambda case_kind, ha, resolution, output_dir, wall_cells: SimpleNamespace(
            time_stepper=SimpleNamespace(dt=0.002, t_final=1.0, max_steps=500, outer_iterations=2),
        ),
    )
    monkeypatch.setattr(suite, "solve_steady", lambda case: SimpleNamespace(mesh=object()))
    monkeypatch.setattr(suite, "duct_layer_resolution_metrics", lambda case, mesh: {"hartmann_layer_cells": 6.0})
    monkeypatch.setattr(suite, "_collect_metrics", lambda solution, case_kind, ha, **kwargs: {"y_l2_error": 0.2, "z_l2_error": 0.3})

    exit_code = suite.main([])

    assert exit_code == 0
    summary = (output / "summary.json").read_text()
    assert '"parameter": "outer_iterations"' in summary
    assert '"parameter_value": 2' in summary
    assert '"hartmann_layer_cells": 6.0' in summary
    assert '"levels"' in capsys.readouterr().out
