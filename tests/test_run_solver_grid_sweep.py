from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_solver_grid_sweep as suite


pytestmark = pytest.mark.unit


def test_run_solver_grid_sweep_writes_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    output = tmp_path / "control_grid"

    monkeypatch.setattr(
        suite.argparse.ArgumentParser,
        "parse_args",
        lambda self, argv=None: SimpleNamespace(
            output=output,
            case="hunt",
            ha=20.0,
            resolution=24,
            wall_cells=4,
            parameter_a="outer_iterations",
            values_a="4,6",
            type_a="int",
            parameter_b="potential_relaxation",
            values_b="1.0,0.5",
            type_b="float",
            reference_root=tmp_path / "refs",
            x_slice="1m",
        ),
    )
    monkeypatch.setattr(
        suite,
        "_build_case",
        lambda case_kind, ha, resolution, output_dir, wall_cells: SimpleNamespace(
            time_stepper=SimpleNamespace(
                dt=0.002,
                t_final=1.0,
                max_steps=500,
                outer_iterations=2,
                potential_relaxation=1.0,
            ),
        ),
    )
    monkeypatch.setattr(
        suite, "solve_steady", lambda case: SimpleNamespace(mesh=object())
    )
    monkeypatch.setattr(
        suite,
        "duct_layer_resolution_metrics",
        lambda case, mesh: {"hartmann_layer_cells": 6.0},
    )
    monkeypatch.setattr(
        suite,
        "_collect_metrics",
        lambda solution, case_kind, ha, **kwargs: {
            "combined_l2_error": 0.255,
            "y_l2_error": 0.2,
            "z_l2_error": 0.3,
        },
    )

    exit_code = suite.main([])

    assert exit_code == 0
    summary = (output / "summary.json").read_text()
    assert '"parameter_a": "outer_iterations"' in summary
    assert '"parameter_b": "potential_relaxation"' in summary
    assert '"parameter_a_value": 6' in summary
    assert '"parameter_b_value": 0.5' in summary
    assert '"levels"' in capsys.readouterr().out
