from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_validation_suite as suite


pytestmark = pytest.mark.unit


def test_run_validation_suite_writes_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    output = tmp_path / "artifacts"
    case = SimpleNamespace(name="hartmann_ha5", output=SimpleNamespace(directory=str(output / "hartmann")))

    monkeypatch.setattr(
        suite.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            output=output,
            ha=5.0,
            reference_root=None,
            x_slice="1m",
            hartmann_l2_threshold=0.05,
            hartmann_linf_threshold=0.1,
        ),
    )
    monkeypatch.setattr(suite, "make_hartmann_case", lambda **kwargs: case)
    monkeypatch.setattr(suite, "make_shercliff_case", lambda **kwargs: SimpleNamespace(name="shercliff_ha5", output=SimpleNamespace(directory=str(output / "shercliff"))))
    monkeypatch.setattr(suite, "make_hunt_case", lambda **kwargs: SimpleNamespace(name="hunt_ha5", output=SimpleNamespace(directory=str(output / "hunt"))))
    monkeypatch.setattr(suite, "solve_steady", lambda built_case: SimpleNamespace(state=SimpleNamespace(time=0.0, residual=0.1), mesh=SimpleNamespace()))
    monkeypatch.setattr(suite, "write_paraview", lambda *args, **kwargs: [])
    monkeypatch.setattr(suite, "write_profile_csv", lambda *args, **kwargs: None)
    monkeypatch.setattr(suite, "extract_centerline", lambda solved: {"y": [0.0], "u": [1.0]})
    monkeypatch.setattr(suite, "extract_midplane_profile", lambda solved, axis: {"z": [0.0], "u": [1.0]})
    monkeypatch.setattr(suite, "validation_summary", lambda solved, case_name, ha: {"case": case_name, "residual": 0.1, "u_max": 1.0})
    monkeypatch.setattr(suite, "hartmann_validation", lambda solved, ha: SimpleNamespace(y_profile=SimpleNamespace(l2_error=0.0, linf_error=0.0)))
    monkeypatch.setattr(
        suite,
        "hartmann_acceptance",
        lambda solved, ha, l2_threshold, linf_threshold: SimpleNamespace(
            passed=True,
            l2_threshold=l2_threshold,
            linf_threshold=linf_threshold,
        ),
    )
    monkeypatch.setattr(suite, "write_analytic_comparison", lambda *args, **kwargs: None)
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


def test_run_validation_suite_handles_reference_branch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    output = tmp_path / "artifacts"
    case = SimpleNamespace(name="hunt_ha5", output=SimpleNamespace(directory=str(output / "hunt")))

    monkeypatch.setattr(
        suite.argparse.ArgumentParser,
        "parse_args",
        lambda self: SimpleNamespace(
            output=output,
            ha=5.0,
            reference_root=tmp_path / "refs",
            x_slice="1m",
            hartmann_l2_threshold=0.05,
            hartmann_linf_threshold=0.1,
        ),
    )
    monkeypatch.setattr(suite, "make_hartmann_case", lambda **kwargs: SimpleNamespace(name="hartmann_ha5", output=SimpleNamespace(directory=str(output / "hartmann"))))
    monkeypatch.setattr(suite, "make_shercliff_case", lambda **kwargs: SimpleNamespace(name="shercliff_ha5", output=SimpleNamespace(directory=str(output / "shercliff"))))
    monkeypatch.setattr(suite, "make_hunt_case", lambda **kwargs: case)
    monkeypatch.setattr(suite, "solve_steady", lambda built_case: SimpleNamespace(state=SimpleNamespace(time=0.0, residual=0.1), mesh=SimpleNamespace()))
    monkeypatch.setattr(suite, "write_paraview", lambda *args, **kwargs: [])
    monkeypatch.setattr(suite, "write_profile_csv", lambda *args, **kwargs: None)
    monkeypatch.setattr(suite, "extract_centerline", lambda solved: {"y": [0.0], "u": [1.0]})
    monkeypatch.setattr(suite, "extract_midplane_profile", lambda solved, axis: {"z": [0.0], "u": [1.0]})
    monkeypatch.setattr(suite, "validation_summary", lambda solved, case_name, ha: {"case": case_name, "residual": 0.1, "u_max": 1.0})
    monkeypatch.setattr(suite, "hartmann_validation", lambda solved, ha: SimpleNamespace(y_profile=SimpleNamespace(l2_error=0.0, linf_error=0.0)))
    monkeypatch.setattr(
        suite,
        "hartmann_acceptance",
        lambda solved, ha, l2_threshold, linf_threshold: SimpleNamespace(
            passed=True,
            l2_threshold=l2_threshold,
            linf_threshold=linf_threshold,
        ),
    )
    monkeypatch.setattr(suite, "write_analytic_comparison", lambda *args, **kwargs: None)
    monkeypatch.setattr(suite, "write_acceptance_report", lambda *args, **kwargs: None)
    monkeypatch.setattr(suite, "write_metrics_json", lambda *args, **kwargs: None)
    comparison = SimpleNamespace(y_profile=SimpleNamespace(l2_error=0.2, linf_error=0.3), z_profile=SimpleNamespace(l2_error=0.4, linf_error=0.5))
    monkeypatch.setattr(suite, "closed_channel_validation", lambda *args, **kwargs: comparison)
    monkeypatch.setattr(suite, "processed_slice_validation", lambda *args, **kwargs: comparison)
    monkeypatch.setattr(suite, "write_closed_channel_validation", lambda *args, **kwargs: None)
    monkeypatch.setattr(suite, "write_processed_slice_validation", lambda *args, **kwargs: None)

    exit_code = suite.main()

    summary = (output / "summary.json").read_text()
    assert exit_code == 0
    assert '"y_l2_error": 0.2' in summary
    assert '"combined_l2_error"' in summary
    assert '"slice_y_l2_error": 0.2' in summary
    assert '"slice_combined_l2_error"' in summary
