from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_convergence_suite as suite


pytestmark = pytest.mark.unit


def test_parse_csv_numbers():
    assert suite._parse_csv_numbers("16, 32,48") == [16, 32, 48]


def test_hunt_wall_cells_and_mesh_spacing():
    case = SimpleNamespace(geometry=SimpleNamespace(width=2.0, height=1.0, ny=8, nz=4))

    assert suite._hunt_wall_cells(8) == 2
    assert suite._hunt_wall_cells(72) == 8
    assert suite._mesh_spacing(case) == pytest.approx(0.25)


def test_observed_orders_reports_second_order_drop():
    levels = [
        {"resolution": 16.0, "mesh_spacing": 0.25, "l2_error": 4.0e-2},
        {"resolution": 32.0, "mesh_spacing": 0.125, "l2_error": 1.0e-2},
    ]
    orders = suite._observed_orders(levels)
    assert orders["l2_error"][0]["order"] == pytest.approx(2.0)


def test_build_case_rejects_unknown_case(tmp_path: Path):
    with pytest.raises(ValueError, match="unknown"):
        suite._build_case("unknown", 20.0, 16, tmp_path)


def test_collect_metrics_for_hartmann(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(suite, "validation_summary", lambda *args, **kwargs: {"base": 1.0})
    monkeypatch.setattr(suite, "hartmann_validation", lambda *args, **kwargs: SimpleNamespace(l2_error=0.1, linf_error=0.2))
    monkeypatch.setattr(
        suite,
        "hartmann_acceptance",
        lambda *args, **kwargs: SimpleNamespace(passed=True, l2_threshold=0.05, linf_threshold=0.1),
    )

    metrics = suite._collect_metrics(
        solution=SimpleNamespace(),
        case_kind="hartmann",
        ha=20.0,
        reference_root=None,
        x_slice="1m",
        hartmann_l2_threshold=0.05,
        hartmann_linf_threshold=0.1,
    )

    assert metrics["l2_error"] == 0.1
    assert metrics["accepted"] == 1.0
    assert metrics["acceptance_l2_threshold"] == 0.05


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
        hartmann_l2_threshold=0.05,
        hartmann_linf_threshold=0.1,
    )

    assert metrics["combined_l2_error"] == pytest.approx(0.2)
    assert "slice_y_l2_error" not in metrics


def test_collect_metrics_reference_branch_includes_slice_metrics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    profile = SimpleNamespace(l2_error=0.2, linf_error=0.3)
    slice_profile = SimpleNamespace(l2_error=0.05, linf_error=0.07)
    comparison = SimpleNamespace(y_profile=profile, z_profile=profile)
    slice_report = SimpleNamespace(y_profile=slice_profile, z_profile=slice_profile)

    monkeypatch.setattr(suite, "validation_summary", lambda *args, **kwargs: {"residual": 1e-4})
    monkeypatch.setattr(suite, "closed_channel_validation", lambda *args, **kwargs: comparison)
    monkeypatch.setattr(suite, "processed_slice_validation", lambda *args, **kwargs: slice_report)

    metrics = suite._collect_metrics(
        solution=SimpleNamespace(),
        case_kind="hunt",
        ha=20.0,
        reference_root=tmp_path / "refs",
        x_slice="1m",
        hartmann_l2_threshold=0.05,
        hartmann_linf_threshold=0.1,
    )

    assert metrics["slice_y_l2_error"] == 0.05
    assert metrics["slice_combined_l2_error"] == pytest.approx(((0.05**2 + 0.05**2) / 2.0) ** 0.5)


def test_observed_orders_ignores_missing_or_none_orders():
    levels = [
        {"resolution": 16.0, "mesh_spacing": 0.25, "l2_error": 0.0},
        {"resolution": 32.0, "mesh_spacing": 0.125, "other_error": 1.0},
    ]

    assert suite._observed_orders(levels) == {}


def test_observed_orders_returns_empty_when_spacing_does_not_change():
    levels = [
        {"resolution": 16.0, "mesh_spacing": 0.25, "l2_error": 1.0},
        {"resolution": 32.0, "mesh_spacing": 0.25, "l2_error": 0.5},
    ]

    assert suite._observed_orders(levels) == {}


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
    monkeypatch.setattr(suite, "solve_steady", lambda case: SimpleNamespace(mesh=object()))
    monkeypatch.setattr(suite, "duct_layer_resolution_metrics", lambda case, mesh: {"hartmann_layer_cells": 8.0})
    monkeypatch.setattr(
        suite,
        "_collect_metrics",
        lambda solution, case_kind, ha, **kwargs: (
            {"l2_error": 4.0e-2 if case_kind == "hartmann" and kwargs else 1.0e-2}
            if case_kind == "hartmann"
            else {"y_l2_error": 0.2, "z_l2_error": 0.1, "combined_l2_error": 0.158}
        ),
    )

    exit_code = suite.main([])

    assert exit_code == 0
    summary = (output / "summary.json").read_text()
    assert '"hartmann"' in summary
    assert '"shercliff"' in summary
    assert '"hartmann_layer_cells": 8.0' in summary
    assert '"combined_l2_error": 0.158' in summary
    assert '"cases"' in capsys.readouterr().out
