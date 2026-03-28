from pathlib import Path
from types import SimpleNamespace

import pytest

from lmx import cli


pytestmark = pytest.mark.unit


def test_cli_benchmark_branch_writes_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    report_path = tmp_path / "benchmark.json"
    recorded: dict[str, object] = {}

    monkeypatch.setattr(
        cli,
        "benchmark_solver",
        lambda repeats, ha, ny, nz: {
            "case": "hartmann_ha5",
            "cold_seconds": 1.0,
            "warm_seconds": 0.5,
            "mean_seconds": 0.75,
            "repeats": float(repeats),
            "backend": "cpu",
            "device_kind": "cpu",
            "jax_version": "0",
            "python_version": "3",
        },
    )
    monkeypatch.setattr(cli, "write_benchmark_report", lambda payload, path: recorded.update(payload=payload, path=path) or Path(path))

    exit_code = cli.main(["benchmark", "--repeats", "2", "--ha", "5", "--ny", "8", "--nz", "8", "--output", str(report_path)])

    assert exit_code == 0
    assert recorded["path"] == str(report_path)
    assert recorded["payload"]["case"] == "hartmann_ha5"
    assert '"case": "hartmann_ha5"' in capsys.readouterr().out


def test_cli_benchmark_branch_skips_writer_when_output_empty(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setattr(
        cli,
        "benchmark_solver",
        lambda repeats, ha, ny, nz: {"case": "hartmann_ha5", "cold_seconds": 1.0, "warm_seconds": 0.5, "mean_seconds": 0.75},
    )
    monkeypatch.setattr(cli, "write_benchmark_report", lambda payload, path: (_ for _ in ()).throw(AssertionError("unexpected write")))

    exit_code = cli.main(["benchmark"])

    assert exit_code == 0
    assert '"case": "hartmann_ha5"' in capsys.readouterr().out


def test_cli_run_branch_uses_case_builder_and_solver(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    output_dir = tmp_path / "run"
    case = SimpleNamespace(name="demo_case", output=SimpleNamespace(directory=str(output_dir)))
    solution = SimpleNamespace(state=SimpleNamespace(time=1.25, residual=0.01), mesh=SimpleNamespace())
    recorded: list[tuple[str, object]] = []

    monkeypatch.setattr(cli, "_build_case", lambda args: case)
    monkeypatch.setattr(cli, "solve_steady", lambda built_case: recorded.append(("solve", built_case)) or solution)
    monkeypatch.setattr(cli, "write_paraview", lambda solved, out_dir: recorded.append(("paraview", out_dir)) or [])
    monkeypatch.setattr(cli, "write_profile_csv", lambda path, profile: recorded.append(("csv", path)) or path)
    monkeypatch.setattr(cli, "extract_centerline", lambda solved: {"y": [0.0], "u": [1.0]})

    exit_code = cli.main(["run", "hartmann", "--output", str(output_dir)])

    assert exit_code == 0
    assert recorded[0][0] == "solve"
    assert recorded[1][0] == "paraview"
    assert recorded[2][0] == "csv"
    assert '"case": "demo_case"' in capsys.readouterr().out


def test_cli_validate_branches_into_reference_comparison(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    output_dir = tmp_path / "validate"
    case = SimpleNamespace(name="shercliff_ha5", output=SimpleNamespace(directory=str(output_dir)))
    solution = SimpleNamespace(state=SimpleNamespace(time=1.25, residual=0.01), mesh=SimpleNamespace())
    comparison = SimpleNamespace(
        y_profile=SimpleNamespace(l2_error=0.2, linf_error=0.3),
        z_profile=SimpleNamespace(l2_error=0.4, linf_error=0.5),
    )
    slice_report = SimpleNamespace(
        y_profile=SimpleNamespace(l2_error=0.6, linf_error=0.7),
        z_profile=SimpleNamespace(l2_error=0.8, linf_error=0.9),
    )
    recorded: dict[str, object] = {}

    monkeypatch.setattr(cli, "_build_case", lambda args: case)
    monkeypatch.setattr(cli, "solve_steady", lambda built_case: solution)
    monkeypatch.setattr(cli, "write_paraview", lambda solved, out_dir: [])
    monkeypatch.setattr(cli, "write_profile_csv", lambda path, profile: path)
    monkeypatch.setattr(cli, "extract_centerline", lambda solved: {"y": [0.0], "u": [1.0]})
    monkeypatch.setattr(cli, "extract_midplane_profile", lambda solved, axis: {"z": [0.0], "u": [1.0]})
    monkeypatch.setattr(cli, "validation_summary", lambda solved, case_name, ha: {"case": case_name, "residual": 0.01, "u_max": 1.0})
    monkeypatch.setattr(cli, "closed_channel_validation", lambda solved, case_name, ha, reference_root: comparison)
    monkeypatch.setattr(cli, "write_closed_channel_validation", lambda report, path: recorded.update(closed=path) or path)
    monkeypatch.setattr(cli, "processed_slice_validation", lambda solved, case_name, ha, x_slice, reference_root: slice_report)
    monkeypatch.setattr(cli, "write_processed_slice_validation", lambda report, path: recorded.update(slice=path) or path)
    monkeypatch.setattr(cli, "write_metrics_json", lambda payload, path: recorded.update(metrics=payload, metrics_path=path) or path)

    exit_code = cli.main(
        [
            "validate",
            "shercliff",
            "--ha",
            "5",
            "--output",
            str(output_dir),
            "--reference-root",
            str(tmp_path / "references"),
        ]
    )

    assert exit_code == 0
    assert "y_l2_error" in recorded["metrics"]
    assert recorded["closed"] == output_dir / "shercliff_ha5_analytic.json"
    assert recorded["slice"] == output_dir / "shercliff_ha5_slice.json"
    assert '"y_l2_error": 0.2' in capsys.readouterr().out


def test_cli_validate_hartmann_branch_writes_analytic_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    output_dir = tmp_path / "validate"
    case = SimpleNamespace(name="hartmann_ha5", output=SimpleNamespace(directory=str(output_dir)))
    solution = SimpleNamespace(state=SimpleNamespace(time=1.25, residual=0.01), mesh=SimpleNamespace())
    recorded: dict[str, object] = {}

    monkeypatch.setattr(cli, "_build_case", lambda args: case)
    monkeypatch.setattr(cli, "solve_steady", lambda built_case: solution)
    monkeypatch.setattr(cli, "write_paraview", lambda solved, out_dir: [])
    monkeypatch.setattr(cli, "write_profile_csv", lambda path, profile: path)
    monkeypatch.setattr(cli, "extract_centerline", lambda solved: {"y": [0.0], "u": [1.0]})
    monkeypatch.setattr(cli, "extract_midplane_profile", lambda solved, axis: {"z": [0.0], "u": [1.0]})
    monkeypatch.setattr(cli, "validation_summary", lambda solved, case_name, ha: {"case": case_name, "residual": 0.01, "u_max": 1.0})
    monkeypatch.setattr(
        cli,
        "hartmann_validation",
        lambda solved, ha: SimpleNamespace(y_profile=SimpleNamespace(l2_error=0.2, linf_error=0.3)),
    )
    monkeypatch.setattr(cli, "write_analytic_comparison", lambda report, path, axis_name: recorded.update(analytic=path, axis=axis_name) or path)
    monkeypatch.setattr(cli, "write_metrics_json", lambda payload, path: recorded.update(metrics=payload, metrics_path=path) or path)

    exit_code = cli.main(["validate", "hartmann", "--ha", "5", "--output", str(output_dir)])

    assert exit_code == 0
    assert recorded["axis"] == "y"
    assert recorded["analytic"] == output_dir / "hartmann_ha5_analytic.json"
    assert '"y_l2_error": 0.2' not in capsys.readouterr().out


def test_cli_validate_reference_branch_handles_missing_slice_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    output_dir = tmp_path / "validate"
    case = SimpleNamespace(name="hunt_ha5", output=SimpleNamespace(directory=str(output_dir)))
    solution = SimpleNamespace(state=SimpleNamespace(time=1.25, residual=0.01), mesh=SimpleNamespace())
    comparison = SimpleNamespace(
        y_profile=SimpleNamespace(l2_error=0.2, linf_error=0.3),
        z_profile=SimpleNamespace(l2_error=0.4, linf_error=0.5),
    )
    recorded: dict[str, object] = {}

    monkeypatch.setattr(cli, "_build_case", lambda args: case)
    monkeypatch.setattr(cli, "solve_steady", lambda built_case: solution)
    monkeypatch.setattr(cli, "write_paraview", lambda solved, out_dir: [])
    monkeypatch.setattr(cli, "write_profile_csv", lambda path, profile: path)
    monkeypatch.setattr(cli, "extract_centerline", lambda solved: {"y": [0.0], "u": [1.0]})
    monkeypatch.setattr(cli, "extract_midplane_profile", lambda solved, axis: {"z": [0.0], "u": [1.0]})
    monkeypatch.setattr(cli, "validation_summary", lambda solved, case_name, ha: {"case": case_name, "residual": 0.01, "u_max": 1.0})
    monkeypatch.setattr(cli, "closed_channel_validation", lambda solved, case_name, ha, reference_root: comparison)
    monkeypatch.setattr(cli, "processed_slice_validation", lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("missing")))
    monkeypatch.setattr(cli, "write_closed_channel_validation", lambda report, path: recorded.update(closed=path) or path)
    monkeypatch.setattr(cli, "write_metrics_json", lambda payload, path: recorded.update(metrics=payload) or path)

    exit_code = cli.main(["validate", "hunt", "--output", str(output_dir), "--reference-root", str(tmp_path / "refs")])

    assert exit_code == 0
    assert recorded["closed"] == output_dir / "hunt_ha5_analytic.json"
    assert "slice_y_l2_error" not in recorded["metrics"]
    assert '"y_l2_error": 0.2' in capsys.readouterr().out


def test_build_case_rejects_unknown_case():
    with pytest.raises(ValueError, match="mystery"):
        cli._build_case(SimpleNamespace(case="mystery", ha=1.0, output="./out"))
