from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_solver_control_sweep as suite


pytestmark = pytest.mark.unit


def test_parse_values_float_and_int():
    assert suite._parse_values("0.1,0.2", "float") == [0.1, 0.2]
    assert suite._parse_values("2,4,6", "int") == [2, 4, 6]
    assert suite._parse_values("jacobi,cg", "str") == ["jacobi", "cg"]


def test_replace_like_rejects_unsupported_object():
    with pytest.raises(TypeError, match="Unsupported object replacement"):
        suite._replace_like(3.14, dt=0.1)


def test_replace_like_supports_plain_objects_with_dict():
    class Config:
        def __init__(self, dt: float, max_steps: int):
            self.dt = dt
            self.max_steps = max_steps

    updated = suite._replace_like(Config(0.01, 10), dt=0.02)
    assert updated.dt == pytest.approx(0.02)
    assert updated.max_steps == 10


def test_build_case_rejects_unknown_case(tmp_path: Path):
    with pytest.raises(ValueError, match="unknown"):
        suite._build_case("unknown", 20.0, 16, tmp_path, None)


def test_build_case_covers_hartmann_shercliff_and_hunt(tmp_path: Path):
    hartmann = suite._build_case("hartmann", 20.0, 12, tmp_path / "hartmann", None)
    shercliff = suite._build_case("shercliff", 20.0, 12, tmp_path / "shercliff", None)
    hunt = suite._build_case("hunt", 20.0, 18, tmp_path / "hunt", None)

    assert hartmann.name.startswith("hartmann")
    assert shercliff.name.startswith("shercliff")
    assert hunt.name.startswith("hunt")
    assert hunt.geometry.kind == "layered_duct"


def test_collect_metrics_covers_hartmann_and_slice_present_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    profile = SimpleNamespace(l2_error=0.2, linf_error=0.3)
    comparison = SimpleNamespace(y_profile=profile, z_profile=profile)
    slice_report = SimpleNamespace(y_profile=profile, z_profile=profile)

    monkeypatch.setattr(suite, "validation_summary", lambda *args, **kwargs: {"residual": 1e-4})
    monkeypatch.setattr(suite, "hartmann_validation", lambda *args, **kwargs: SimpleNamespace(l2_error=0.01, linf_error=0.02))
    monkeypatch.setattr(suite, "hartmann_acceptance", lambda *args, **kwargs: SimpleNamespace(passed=True))
    monkeypatch.setattr(suite, "closed_channel_validation", lambda *args, **kwargs: comparison)
    monkeypatch.setattr(suite, "processed_slice_validation", lambda *args, **kwargs: slice_report)

    hartmann_metrics = suite._collect_metrics(
        solution=SimpleNamespace(),
        case_kind="hartmann",
        ha=20.0,
        reference_root=None,
        x_slice="1m",
    )
    hunt_metrics = suite._collect_metrics(
        solution=SimpleNamespace(),
        case_kind="hunt",
        ha=20.0,
        reference_root=tmp_path / "refs",
        x_slice="1m",
    )

    assert hartmann_metrics["accepted"] == pytest.approx(1.0)
    assert hunt_metrics["slice_combined_l2_error"] == pytest.approx(0.2)


def test_collect_metrics_reference_branch_handles_missing_slice(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    profile = SimpleNamespace(l2_error=0.2, linf_error=0.3)
    comparison = SimpleNamespace(y_profile=profile, z_profile=profile)

    monkeypatch.setattr(suite, "validation_summary", lambda *args, **kwargs: {"residual": 1e-4})
    monkeypatch.setattr(suite, "closed_channel_validation", lambda *args, **kwargs: comparison)
    monkeypatch.setattr(suite, "processed_slice_validation", lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("missing")))

    metrics = suite._collect_metrics(
        solution=SimpleNamespace(),
        case_kind="hunt",
        ha=20.0,
        reference_root=tmp_path / "refs",
        x_slice="1m",
    )

    assert metrics["combined_l2_error"] == pytest.approx(0.2)
    assert "slice_y_l2_error" not in metrics


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


def test_run_solver_control_sweep_accepts_potential_relaxation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "control_sweep_relaxation"

    monkeypatch.setattr(
        suite.argparse.ArgumentParser,
        "parse_args",
        lambda self, argv=None: SimpleNamespace(
            output=output,
            case="hartmann",
            ha=20.0,
            resolution=24,
            wall_cells=None,
            parameter="potential_relaxation",
            values="1.0,0.5",
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
                potential_relaxation=1.0,
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
    assert '"parameter": "potential_relaxation"' in summary
    assert '"parameter_value": 0.5' in summary


def test_run_solver_control_sweep_accepts_potential_solver(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "control_sweep_solver"

    monkeypatch.setattr(
        suite.argparse.ArgumentParser,
        "parse_args",
        lambda self, argv=None: SimpleNamespace(
            output=output,
            case="hartmann",
            ha=20.0,
            resolution=24,
            wall_cells=None,
            parameter="potential_solver",
            values="jacobi,cg",
            value_type="str",
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
                potential_relaxation=1.0,
                potential_solver="jacobi",
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
    assert '"parameter": "potential_solver"' in summary
    assert '"parameter_value": "cg"' in summary


def test_run_solver_control_sweep_accepts_current_reconstruction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "control_sweep_current_reconstruction"

    monkeypatch.setattr(
        suite.argparse.ArgumentParser,
        "parse_args",
        lambda self, argv=None: SimpleNamespace(
            output=output,
            case="hunt",
            ha=20.0,
            resolution=24,
            wall_cells=4,
            parameter="current_reconstruction",
            values="cell_centered,face_averaged",
            value_type="str",
            reference_root=None,
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
                outer_iterations=6,
                current_reconstruction="cell_centered",
            ),
        ),
    )
    monkeypatch.setattr(suite, "solve_steady", lambda case: SimpleNamespace(mesh=object()))
    monkeypatch.setattr(suite, "duct_layer_resolution_metrics", lambda case, mesh: {"hartmann_layer_cells": 6.0})
    monkeypatch.setattr(
        suite,
        "_collect_metrics",
        lambda solution, case_kind, ha, **kwargs: {"y_l2_error": 0.2, "z_l2_error": 0.3, "combined_l2_error": 0.255},
    )

    exit_code = suite.main([])

    assert exit_code == 0
    summary = (output / "summary.json").read_text()
    assert '"parameter": "current_reconstruction"' in summary
    assert '"parameter_value": "face_averaged"' in summary


def test_run_solver_control_sweep_accepts_velocity_update_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "control_sweep_velocity_limit"

    monkeypatch.setattr(
        suite.argparse.ArgumentParser,
        "parse_args",
        lambda self, argv=None: SimpleNamespace(
            output=output,
            case="hunt",
            ha=20.0,
            resolution=24,
            wall_cells=4,
            parameter="velocity_update_limit",
            values="0.001,0.002",
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
                dt=0.002,
                t_final=1.0,
                max_steps=500,
                outer_iterations=2,
                velocity_update_limit=0.001,
            ),
        ),
    )
    monkeypatch.setattr(suite, "solve_steady", lambda case: SimpleNamespace(mesh=object()))
    monkeypatch.setattr(suite, "duct_layer_resolution_metrics", lambda case, mesh: {"hartmann_layer_cells": 6.0})
    monkeypatch.setattr(
        suite,
        "_collect_metrics",
        lambda solution, case_kind, ha, **kwargs: {"y_l2_error": 0.2, "z_l2_error": 0.3, "combined_l2_error": 0.255},
    )

    exit_code = suite.main([])

    assert exit_code == 0
    summary = (output / "summary.json").read_text()
    assert '"parameter": "velocity_update_limit"' in summary
    assert '"parameter_value": 0.002' in summary


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
    monkeypatch.setattr(
        suite,
        "_collect_metrics",
        lambda solution, case_kind, ha, **kwargs: {"y_l2_error": 0.2, "z_l2_error": 0.3, "combined_l2_error": 0.255},
    )

    exit_code = suite.main([])

    assert exit_code == 0
    summary = (output / "summary.json").read_text()
    assert '"parameter": "outer_iterations"' in summary
    assert '"parameter_value": 2' in summary
    assert '"hartmann_layer_cells": 6.0' in summary
    assert '"combined_l2_error": 0.255' in summary
    assert '"levels"' in capsys.readouterr().out


def test_run_solver_control_sweep_dt_updates_max_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "control_sweep_dt"

    monkeypatch.setattr(
        suite.argparse.ArgumentParser,
        "parse_args",
        lambda self, argv=None: SimpleNamespace(
            output=output,
            case="hartmann",
            ha=20.0,
            resolution=24,
            wall_cells=None,
            parameter="dt",
            values="0.01",
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
                t_final=0.1,
                max_steps=400,
                outer_iterations=2,
            ),
        ),
    )
    monkeypatch.setattr(suite, "solve_steady", lambda case: SimpleNamespace(mesh=object()))
    monkeypatch.setattr(suite, "duct_layer_resolution_metrics", lambda case, mesh: {})
    monkeypatch.setattr(suite, "_collect_metrics", lambda *args, **kwargs: {"l2_error": 0.2})

    assert suite.main([]) == 0
    summary = (output / "summary.json").read_text()
    assert '"dt": 0.01' in summary
    assert '"max_steps": 10.0' in summary


def test_run_solver_control_sweep_dt_does_not_round_up_max_steps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "control_sweep_dt_floor"

    monkeypatch.setattr(
        suite.argparse.ArgumentParser,
        "parse_args",
        lambda self, argv=None: SimpleNamespace(
            output=output,
            case="hartmann",
            ha=20.0,
            resolution=24,
            wall_cells=None,
            parameter="dt",
            values="0.06",
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
                t_final=0.1,
                max_steps=400,
                outer_iterations=2,
            ),
        ),
    )
    monkeypatch.setattr(suite, "solve_steady", lambda case: SimpleNamespace(mesh=object()))
    monkeypatch.setattr(suite, "duct_layer_resolution_metrics", lambda case, mesh: {})
    monkeypatch.setattr(suite, "_collect_metrics", lambda *args, **kwargs: {"l2_error": 0.2})

    assert suite.main([]) == 0
    summary = (output / "summary.json").read_text()
    assert '"dt": 0.06' in summary
    assert '"max_steps": 1.0' in summary
