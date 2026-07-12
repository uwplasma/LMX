from pathlib import Path
from types import SimpleNamespace
import runpy

import pytest

from lmx import cli
from lmx.config import FringingSpec, LoggingSpec, RestartSpec, RunConfig


pytestmark = pytest.mark.unit


def test_cli_benchmark_branch_writes_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
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
    monkeypatch.setattr(
        cli,
        "write_benchmark_report",
        lambda payload, path: recorded.update(payload=payload, path=path) or Path(path),
    )

    exit_code = cli.main(
        [
            "benchmark",
            "--repeats",
            "2",
            "--ha",
            "5",
            "--ny",
            "8",
            "--nz",
            "8",
            "--output",
            str(report_path),
        ]
    )

    assert exit_code == 0
    assert recorded["path"] == str(report_path)
    assert recorded["payload"]["case"] == "hartmann_ha5"
    assert '"case": "hartmann_ha5"' in capsys.readouterr().out


def test_cli_benchmark_branch_skips_writer_when_output_empty(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    monkeypatch.setattr(
        cli,
        "benchmark_solver",
        lambda repeats, ha, ny, nz: {
            "case": "hartmann_ha5",
            "cold_seconds": 1.0,
            "warm_seconds": 0.5,
            "mean_seconds": 0.75,
        },
    )
    monkeypatch.setattr(
        cli,
        "write_benchmark_report",
        lambda payload, path: (_ for _ in ()).throw(AssertionError("unexpected write")),
    )

    exit_code = cli.main(["benchmark"])

    assert exit_code == 0
    assert '"case": "hartmann_ha5"' in capsys.readouterr().out


def test_cli_run_branch_uses_case_builder_and_solver(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    output_dir = tmp_path / "run"
    case = SimpleNamespace(
        name="demo_case", output=SimpleNamespace(directory=str(output_dir))
    )
    solution = SimpleNamespace(
        state=SimpleNamespace(time=1.25, residual=0.01), mesh=SimpleNamespace()
    )
    recorded: list[tuple[str, object]] = []

    monkeypatch.setattr(cli, "_build_case", lambda args: case)
    monkeypatch.setattr(
        cli,
        "solve_steady",
        lambda built_case: recorded.append(("solve", built_case)) or solution,
    )
    monkeypatch.setattr(
        cli,
        "write_solution_outputs",
        lambda solved, built_case, out_dir, write_npz, write_plots: (
            recorded.append(("outputs", out_dir))
            or {"paraview": [], "csv": [], "npz": [], "plots": []}
        ),
    )

    exit_code = cli.main(["run", "hartmann", "--output", str(output_dir)])

    assert exit_code == 0
    assert recorded[0][0] == "solve"
    assert recorded[1][0] == "outputs"
    assert '"case": "demo_case"' in capsys.readouterr().out


def test_cli_run_branch_dispatches_extruded_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    output_dir = tmp_path / "fringing"
    case = SimpleNamespace(
        name="fringing_rect_demo",
        forcing=1.0,
        geometry=SimpleNamespace(kind="rect_duct"),
        solver=SimpleNamespace(kind="extruded_inductionless", mode="steady"),
        output=SimpleNamespace(
            directory=str(output_dir),
            write_npz=True,
            write_json_summary=True,
            write_plots=False,
        ),
    )
    problem = SimpleNamespace(case=case, profile=SimpleNamespace())
    solution = SimpleNamespace(
        bundle=SimpleNamespace(
            x=cli.jnp.asarray([0.0, 1.0]), u=cli.jnp.asarray([[[1.0]], [[0.5]]])
        ),
        validation=SimpleNamespace(
            max_residual=1.0e-4,
            max_charge_balance_residual=1.0e-6,
            max_wall_current_leakage=2.0e-6,
            net_boundary_current_residual=3.0e-6,
            field_mean_velocity_correlation=-0.8,
        ),
        station_history=(),
    )
    recorded: dict[str, object] = {}

    monkeypatch.setattr(cli, "_build_extruded_problem", lambda args: problem)
    monkeypatch.setattr(
        cli, "solve_extruded_inductionless", lambda built_problem: solution
    )
    monkeypatch.setattr(
        cli,
        "write_extruded_solution_outputs",
        lambda solved, built_case, out_dir, write_npz, write_plots=False: (
            recorded.update(
                out_dir=Path(out_dir), write_npz=write_npz, write_plots=write_plots
            )
            or {
                "csv": [Path(out_dir) / "stations.csv"],
                "npz": [Path(out_dir) / "bundle.npz"],
                "plots": [],
            }
        ),
    )

    exit_code = cli.main(
        [
            "run",
            "fringing_rect",
            "--output",
            str(output_dir),
            "--nx-stations",
            "5",
            "--plots",
        ]
    )

    assert exit_code == 0
    assert recorded["out_dir"] == output_dir
    assert recorded["write_npz"] is True
    assert recorded["write_plots"] is True
    assert '"solver_kind": "extruded_inductionless"' in capsys.readouterr().out


def test_cli_dispatches_direct_toml_run(monkeypatch: pytest.MonkeyPatch):
    recorded: dict[str, object] = {}
    monkeypatch.setattr(
        cli,
        "run_from_toml",
        lambda path: recorded.update(path=path) or {"case": "demo"},
    )

    exit_code = cli.main(["/tmp/demo_case.toml"])

    assert exit_code == 0
    assert recorded["path"] == "/tmp/demo_case.toml"


def test_cli_case_builders_reject_unknown_case():
    with pytest.raises(ValueError):
        cli._build_case(SimpleNamespace(case="unknown", ha=5.0, output="out"))
    with pytest.raises(ValueError):
        cli._build_extruded_problem(
            SimpleNamespace(
                case="unknown",
                ha=5.0,
                width=2.0,
                height=2.0,
                ny=4,
                nz=4,
                length=6.0,
                nx_stations=5,
                entry_center=1.0,
                exit_center=4.0,
                transition_width=0.5,
                wall_cells=1,
                insulator_cells=1,
                radius=0.5,
                nr=4,
                ntheta=8,
            )
        )


def test_python_module_entrypoint_delegates_to_cli_main(
    monkeypatch: pytest.MonkeyPatch,
):
    recorded: dict[str, object] = {}

    monkeypatch.setattr(
        "lmx.cli.main", lambda argv=None: recorded.update(argv=argv) or 0
    )
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("lmx", run_name="__main__")

    assert excinfo.value.code == 0
    assert recorded["argv"] is None


def test_run_config_uses_restart_bundle_and_writes_restart_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    output_dir = tmp_path / "run"
    case = cli._build_case(
        SimpleNamespace(case="hartmann", ha=5.0, output=str(output_dir))
    )
    case = case.__class__(
        **{
            **case.__dict__,
            "output": case.output.__class__(
                **{**case.output.__dict__, "directory": str(output_dir)}
            ),
        }
    )
    config = RunConfig(
        case=case,
        solve_mode="steady",
        logging=LoggingSpec(enabled=False),
        restart=RestartSpec(
            enabled=True,
            path=tmp_path / "restart.npz",
            reset_histories=False,
            write_restart=True,
            restart_filename="resume_state.npz",
        ),
    )
    bundle = SimpleNamespace(
        path=(tmp_path / "restart.npz").resolve(),
        state=SimpleNamespace(time=0.2),
        diagnostics=SimpleNamespace(
            time_history=[],
            u_max_history=[],
            mean_velocity_history=[],
            applied_forcing_history=[],
            pressure_proxy_history=[],
            residual_history=[],
            courant_like=[],
            ohmic_power=[],
            current_max_history=[],
            face_current_max_history=[],
            emf_max_history=[],
            lorentz_max_history=[],
            potential_residual_history=[],
            potential_iterations_history=[],
        ),
    )
    solution = SimpleNamespace(
        state=SimpleNamespace(time=0.4, residual=1e-3, u=cli.jnp.asarray([[1.0]])),
        diagnostics=SimpleNamespace(
            potential_residual_history=cli.jnp.asarray([1e-4]),
            potential_iterations_history=cli.jnp.asarray([12.0]),
        ),
        mesh=SimpleNamespace(),
        case_name=case.name,
    )

    monkeypatch.setattr(cli, "load_restart_bundle", lambda path: bundle)
    monkeypatch.setattr(
        cli,
        "validate_restart_bundle",
        lambda bundle, mesh, geometry_kind, case_name: None,
    )
    monkeypatch.setattr(cli, "_build_mesh", lambda built_case: SimpleNamespace())
    perf_values = iter([10.0, 13.5])
    monkeypatch.setattr(cli.time, "perf_counter", lambda: next(perf_values))
    monkeypatch.setattr(
        cli,
        "_solve_case_with_optional_logger",
        lambda built_case, **kwargs: solution,
    )
    monkeypatch.setattr(
        cli,
        "write_solution_outputs",
        lambda solved, built_case, out_dir, write_npz, write_plots: {
            "paraview": [],
            "csv": [],
            "npz": [],
            "plots": [],
        },
    )
    monkeypatch.setattr(
        cli, "write_restart_npz", lambda solved, built_case, path: Path(path)
    )

    summary = cli._run_config(config)

    assert summary["restart"]["enabled"] is True
    assert summary["restart"]["start_time"] == pytest.approx(0.2)
    assert summary["restart"]["output"] == "resume_state.npz"
    assert summary["execution_seconds"] == pytest.approx(3.5)
    assert '"restart"' in capsys.readouterr().out


def test_run_config_dispatches_extruded_solver_kind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    case = cli._build_case(
        SimpleNamespace(case="hartmann", ha=5.0, output=str(tmp_path / "run"))
    )
    case = case.__class__(
        **{
            **case.__dict__,
            "name": "fringing_rect_demo",
            "geometry": case.geometry.__class__(
                **{**case.geometry.__dict__, "length": 6.0, "nx": 5}
            ),
            "solver": case.solver.__class__(
                **{**case.solver.__dict__, "kind": "extruded_inductionless"}
            ),
            "output": case.output.__class__(
                **{**case.output.__dict__, "directory": str(tmp_path / "run")}
            ),
        }
    )
    config = RunConfig(
        case=case,
        solve_mode="steady",
        logging=LoggingSpec(enabled=False),
        fringing=FringingSpec(
            enabled=True,
            entry_center=1.0,
            exit_center=4.0,
            transition_width=0.5,
            axis="z",
        ),
    )
    recorded: dict[str, object] = {}
    solution = SimpleNamespace(
        bundle=SimpleNamespace(
            x=cli.jnp.asarray([0.0, 1.0]),
            u=cli.jnp.asarray([[[1.0]], [[0.5]]]),
        ),
        validation=SimpleNamespace(
            max_residual=1.0e-4,
            max_charge_balance_residual=1.0e-6,
            max_wall_current_leakage=2.0e-6,
            net_boundary_current_residual=3.0e-6,
            field_mean_velocity_correlation=-0.8,
        ),
        station_history=(),
    )

    monkeypatch.setattr(
        cli,
        "build_extruded_problem_from_case",
        lambda built_case, **kwargs: (
            recorded.update(problem_kwargs=kwargs) or SimpleNamespace(case=built_case)
        ),
    )
    perf_values = iter([20.0, 26.0])
    monkeypatch.setattr(cli.time, "perf_counter", lambda: next(perf_values))
    monkeypatch.setattr(
        cli,
        "solve_extruded_inductionless",
        lambda problem, initial_bundle=None: solution,
    )
    monkeypatch.setattr(
        cli,
        "write_extruded_solution_outputs",
        lambda solved, built_case, out_dir, write_npz, write_plots=False: (
            recorded.update(
                out_dir=Path(out_dir), write_npz=write_npz, write_plots=write_plots
            )
            or {
                "csv": [Path(out_dir) / "stations.csv"],
                "npz": [Path(out_dir) / "bundle.npz"],
                "plots": [],
            }
        ),
    )

    summary = cli._run_config(config)

    assert recorded["problem_kwargs"]["entry_center"] == pytest.approx(1.0)
    assert recorded["problem_kwargs"]["exit_center"] == pytest.approx(4.0)
    assert summary["solver_kind"] == "extruded_inductionless"
    assert summary["station_count"] == 2
    assert summary["execution_seconds"] == pytest.approx(6.0)
    assert '"solver_kind": "extruded_inductionless"' in capsys.readouterr().out


def test_run_config_extruded_requires_restart_path_when_restart_enabled(tmp_path: Path):
    case = cli._build_case(
        SimpleNamespace(case="hartmann", ha=5.0, output=str(tmp_path / "run"))
    )
    case = case.__class__(
        **{
            **case.__dict__,
            "name": "fringing_rect_demo",
            "geometry": case.geometry.__class__(
                **{**case.geometry.__dict__, "length": 6.0, "nx": 5}
            ),
            "solver": case.solver.__class__(
                **{**case.solver.__dict__, "kind": "extruded_inductionless"}
            ),
        }
    )
    config = RunConfig(
        case=case,
        solve_mode="steady",
        logging=LoggingSpec(enabled=False),
        fringing=FringingSpec(
            enabled=True,
            entry_center=1.0,
            exit_center=4.0,
            transition_width=0.5,
            axis="z",
        ),
        restart=RestartSpec(enabled=True),
    )
    with pytest.raises(ValueError, match="restart.path"):
        cli._run_config(config)


def test_run_config_extruded_requires_fringing_block(tmp_path: Path):
    case = cli._build_case(
        SimpleNamespace(case="hartmann", ha=5.0, output=str(tmp_path / "run"))
    )
    case = case.__class__(
        **{
            **case.__dict__,
            "name": "fringing_rect_demo",
            "geometry": case.geometry.__class__(
                **{**case.geometry.__dict__, "length": 6.0, "nx": 5}
            ),
            "solver": case.solver.__class__(
                **{**case.solver.__dict__, "kind": "extruded_inductionless"}
            ),
        }
    )
    config = RunConfig(
        case=case, solve_mode="steady", logging=LoggingSpec(enabled=False)
    )
    with pytest.raises(ValueError, match="\\[fringing\\] block"):
        cli._run_config(config)


def test_run_config_supports_extruded_restart_and_structured_output_layout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
):
    output_dir = tmp_path / "fringing_run"
    input_path = tmp_path / "fringing_case.toml"
    input_path.write_text("[case]\nname='demo'\n", encoding="utf-8")
    case = cli._build_case(
        SimpleNamespace(case="hartmann", ha=5.0, output=str(output_dir))
    )
    case = case.__class__(
        **{
            **case.__dict__,
            "name": "fringing_layered_demo",
            "geometry": case.geometry.__class__(
                **{
                    **case.geometry.__dict__,
                    "kind": "layered_duct",
                    "length": 6.0,
                    "nx": 5,
                }
            ),
            "solver": case.solver.__class__(
                **{**case.solver.__dict__, "kind": "extruded_inductionless"}
            ),
            "output": case.output.__class__(
                **{
                    **case.output.__dict__,
                    "directory": str(output_dir),
                    "write_npz": True,
                    "copy_input_file": True,
                }
            ),
        }
    )
    config = RunConfig(
        case=case,
        solve_mode="steady",
        logging=LoggingSpec(enabled=True),
        restart=RestartSpec(
            enabled=True,
            path=tmp_path / "extruded_restart.npz",
            reset_histories=False,
            write_restart=True,
            restart_filename="fringing_resume.npz",
        ),
        fringing=FringingSpec(
            enabled=True,
            entry_center=1.0,
            exit_center=4.0,
            transition_width=0.5,
            axis="z",
        ),
        input_path=input_path,
    )
    restart_bundle = SimpleNamespace(
        path=(tmp_path / "extruded_restart.npz").resolve(),
        bundle=SimpleNamespace(
            x=cli.jnp.asarray([0.0, 1.0]),
            y=cli.jnp.asarray([0.0]),
            z=cli.jnp.asarray([0.0]),
        ),
    )
    solution = SimpleNamespace(
        bundle=SimpleNamespace(
            x=cli.jnp.asarray([0.0, 1.0]), u=cli.jnp.asarray([[[1.0]], [[0.5]]])
        ),
        validation=SimpleNamespace(
            max_residual=1.0e-4,
            max_charge_balance_residual=1.0e-6,
            max_wall_current_leakage=2.0e-6,
            net_boundary_current_residual=3.0e-6,
            field_mean_velocity_correlation=-0.8,
        ),
        station_history=(),
    )
    recorded: dict[str, object] = {}

    monkeypatch.setattr(
        cli, "load_extruded_restart_bundle", lambda path: restart_bundle
    )
    monkeypatch.setattr(
        cli,
        "validate_extruded_restart_bundle",
        lambda bundle, case: recorded.update(validated_case=case.name),
    )
    monkeypatch.setattr(
        cli,
        "build_extruded_problem_from_case",
        lambda built_case, **kwargs: (
            recorded.update(problem_kwargs=kwargs) or SimpleNamespace(case=built_case)
        ),
    )
    monkeypatch.setattr(
        cli,
        "solve_extruded_inductionless",
        lambda problem, initial_bundle=None: (
            recorded.update(initial_bundle=initial_bundle) or solution
        ),
    )
    monkeypatch.setattr(
        cli,
        "write_extruded_solution_outputs",
        lambda solved, built_case, out_dir, write_npz, write_plots=False: {
            "csv": [Path(out_dir) / "postProcessing" / "stations.csv"],
            "npz": [Path(out_dir) / "fields" / "bundle.npz"],
            "plots": [],
        },
    )
    monkeypatch.setattr(
        cli,
        "write_extruded_restart_npz",
        lambda solved, built_case, path: Path(path),
    )

    summary = cli._run_config(config)

    assert recorded["validated_case"] == case.name
    assert recorded["initial_bundle"] is restart_bundle.bundle
    assert summary["restart"]["enabled"] is True
    assert summary["restart"]["output"] == "fringing_resume.npz"
    assert (output_dir / "system" / input_path.name).exists()
    assert '"restart"' in capsys.readouterr().out


def test_cli_validate_branches_into_reference_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    output_dir = tmp_path / "validate"
    case = SimpleNamespace(
        name="shercliff_ha5", output=SimpleNamespace(directory=str(output_dir))
    )
    solution = SimpleNamespace(
        state=SimpleNamespace(time=1.25, residual=0.01), mesh=SimpleNamespace()
    )
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
    monkeypatch.setattr(
        cli, "extract_centerline", lambda solved: {"y": [0.0], "u": [1.0]}
    )
    monkeypatch.setattr(
        cli, "extract_midplane_profile", lambda solved, axis: {"z": [0.0], "u": [1.0]}
    )
    monkeypatch.setattr(
        cli,
        "validation_summary",
        lambda solved, case_name, ha: {
            "case": case_name,
            "residual": 0.01,
            "u_max": 1.0,
        },
    )
    monkeypatch.setattr(
        cli,
        "closed_channel_validation",
        lambda solved, case_name, ha, reference_root: comparison,
    )
    monkeypatch.setattr(
        cli,
        "write_closed_channel_validation",
        lambda report, path: recorded.update(closed=path) or path,
    )
    monkeypatch.setattr(
        cli,
        "processed_slice_validation",
        lambda solved, case_name, ha, x_slice, reference_root: slice_report,
    )
    monkeypatch.setattr(
        cli,
        "write_processed_slice_validation",
        lambda report, path: recorded.update(slice=path) or path,
    )
    monkeypatch.setattr(
        cli,
        "write_metrics_json",
        lambda payload, path: (
            recorded.update(metrics=payload, metrics_path=path) or path
        ),
    )

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
    assert "combined_l2_error" in recorded["metrics"]
    assert recorded["closed"] == output_dir / "shercliff_ha5_analytic.json"
    assert recorded["slice"] == output_dir / "shercliff_ha5_slice.json"
    assert recorded["metrics"]["combined_l2_error"] == pytest.approx(
        ((0.2**2 + 0.4**2) / 2.0) ** 0.5
    )
    assert '"y_l2_error": 0.2' in capsys.readouterr().out


def test_cli_validate_hartmann_branch_writes_analytic_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    output_dir = tmp_path / "validate"
    case = SimpleNamespace(
        name="hartmann_ha5", output=SimpleNamespace(directory=str(output_dir))
    )
    solution = SimpleNamespace(
        state=SimpleNamespace(time=1.25, residual=0.01), mesh=SimpleNamespace()
    )
    recorded: dict[str, object] = {}

    monkeypatch.setattr(cli, "_build_case", lambda args: case)
    monkeypatch.setattr(cli, "solve_steady", lambda built_case: solution)
    monkeypatch.setattr(cli, "write_paraview", lambda solved, out_dir: [])
    monkeypatch.setattr(cli, "write_profile_csv", lambda path, profile: path)
    monkeypatch.setattr(
        cli, "extract_centerline", lambda solved: {"y": [0.0], "u": [1.0]}
    )
    monkeypatch.setattr(
        cli, "extract_midplane_profile", lambda solved, axis: {"z": [0.0], "u": [1.0]}
    )
    monkeypatch.setattr(
        cli,
        "validation_summary",
        lambda solved, case_name, ha: {
            "case": case_name,
            "residual": 0.01,
            "u_max": 1.0,
        },
    )
    monkeypatch.setattr(
        cli,
        "hartmann_validation",
        lambda solved, ha: SimpleNamespace(
            y_profile=SimpleNamespace(l2_error=0.2, linf_error=0.3)
        ),
    )
    monkeypatch.setattr(
        cli,
        "hartmann_acceptance",
        lambda solved, ha, l2_threshold, linf_threshold: SimpleNamespace(
            passed=True,
            l2_threshold=l2_threshold,
            linf_threshold=linf_threshold,
        ),
    )
    monkeypatch.setattr(
        cli,
        "write_analytic_comparison",
        lambda report, path, axis_name: (
            recorded.update(analytic=path, axis=axis_name) or path
        ),
    )
    monkeypatch.setattr(
        cli,
        "write_acceptance_report",
        lambda report, path: recorded.update(acceptance=path) or path,
    )
    monkeypatch.setattr(
        cli,
        "write_metrics_json",
        lambda payload, path: (
            recorded.update(metrics=payload, metrics_path=path) or path
        ),
    )

    exit_code = cli.main(
        ["validate", "hartmann", "--ha", "5", "--output", str(output_dir)]
    )

    assert exit_code == 0
    assert recorded["axis"] == "y"
    assert recorded["analytic"] == output_dir / "hartmann_ha5_analytic.json"
    assert recorded["acceptance"] == output_dir / "hartmann_ha5_acceptance.json"
    assert recorded["metrics"]["accepted"] == pytest.approx(1.0)
    assert '"y_l2_error": 0.2' not in capsys.readouterr().out


def test_cli_validate_reference_branch_handles_missing_slice_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    output_dir = tmp_path / "validate"
    case = SimpleNamespace(
        name="hunt_ha5", output=SimpleNamespace(directory=str(output_dir))
    )
    solution = SimpleNamespace(
        state=SimpleNamespace(time=1.25, residual=0.01), mesh=SimpleNamespace()
    )
    comparison = SimpleNamespace(
        y_profile=SimpleNamespace(l2_error=0.2, linf_error=0.3),
        z_profile=SimpleNamespace(l2_error=0.4, linf_error=0.5),
    )
    recorded: dict[str, object] = {}

    monkeypatch.setattr(cli, "_build_case", lambda args: case)
    monkeypatch.setattr(cli, "solve_steady", lambda built_case: solution)
    monkeypatch.setattr(cli, "write_paraview", lambda solved, out_dir: [])
    monkeypatch.setattr(cli, "write_profile_csv", lambda path, profile: path)
    monkeypatch.setattr(
        cli, "extract_centerline", lambda solved: {"y": [0.0], "u": [1.0]}
    )
    monkeypatch.setattr(
        cli, "extract_midplane_profile", lambda solved, axis: {"z": [0.0], "u": [1.0]}
    )
    monkeypatch.setattr(
        cli,
        "validation_summary",
        lambda solved, case_name, ha: {
            "case": case_name,
            "residual": 0.01,
            "u_max": 1.0,
        },
    )
    monkeypatch.setattr(
        cli,
        "closed_channel_validation",
        lambda solved, case_name, ha, reference_root: comparison,
    )
    monkeypatch.setattr(
        cli,
        "processed_slice_validation",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("missing")),
    )
    monkeypatch.setattr(
        cli,
        "write_closed_channel_validation",
        lambda report, path: recorded.update(closed=path) or path,
    )
    monkeypatch.setattr(
        cli,
        "write_metrics_json",
        lambda payload, path: recorded.update(metrics=payload) or path,
    )

    exit_code = cli.main(
        [
            "validate",
            "hunt",
            "--output",
            str(output_dir),
            "--reference-root",
            str(tmp_path / "refs"),
        ]
    )

    assert exit_code == 0
    assert recorded["closed"] == output_dir / "hunt_ha5_analytic.json"
    assert "slice_y_l2_error" not in recorded["metrics"]
    assert "combined_l2_error" in recorded["metrics"]
    assert '"y_l2_error": 0.2' in capsys.readouterr().out


def test_build_case_rejects_unknown_case():
    with pytest.raises(ValueError, match="mystery"):
        cli._build_case(SimpleNamespace(case="mystery", ha=1.0, output="./out"))


def test_solve_case_with_optional_logger_falls_back_on_typeerror(
    monkeypatch: pytest.MonkeyPatch,
):
    case = SimpleNamespace(name="demo")
    calls: list[tuple[str, object]] = []

    def fake_transient(case, **kwargs):
        if kwargs:
            raise TypeError("old signature")
        calls.append(("transient", case))
        return "transient-ok"

    def fake_steady(case, **kwargs):
        if kwargs:
            raise TypeError("old signature")
        calls.append(("steady", case))
        return "steady-ok"

    monkeypatch.setattr(cli, "solve_transient", fake_transient)
    monkeypatch.setattr(cli, "solve_steady", fake_steady)

    assert (
        cli._solve_case_with_optional_logger(
            case, solve_mode="transient", logger=object()
        )
        == "transient-ok"
    )
    assert (
        cli._solve_case_with_optional_logger(case, solve_mode="steady", logger=object())
        == "steady-ok"
    )
    assert calls == [("transient", case), ("steady", case)]


def test_write_run_summary_respects_disabled_json_summary(tmp_path: Path):
    case = SimpleNamespace(
        name="demo", output=SimpleNamespace(write_json_summary=False)
    )
    assert cli._write_run_summary({"case": "demo"}, case, tmp_path) is None


def test_run_config_requires_restart_path(tmp_path: Path):
    case = cli._build_case(
        SimpleNamespace(case="hartmann", ha=5.0, output=str(tmp_path))
    )
    config = RunConfig(
        case=case,
        solve_mode="steady",
        logging=LoggingSpec(enabled=False),
        restart=RestartSpec(enabled=True, path=None),
    )

    with pytest.raises(ValueError, match="restart.path"):
        cli._run_config(config)


def test_run_branch_quiet_disables_logging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    case = cli._build_case(
        SimpleNamespace(case="hartmann", ha=5.0, output=str(tmp_path))
    )
    recorded: dict[str, object] = {}

    monkeypatch.setattr(cli, "_build_case", lambda args: case)
    monkeypatch.setattr(
        cli,
        "_run_config",
        lambda config: (
            recorded.update(enabled=config.logging.enabled) or {"case": case.name}
        ),
    )

    exit_code = cli.main(["run", "hartmann", "--output", str(tmp_path), "--quiet"])

    assert exit_code == 0
    assert recorded["enabled"] is False


def test_run_branch_verbose_enables_debug_logging(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    case = cli._build_case(
        SimpleNamespace(case="hartmann", ha=5.0, output=str(tmp_path))
    )
    recorded: dict[str, object] = {}

    monkeypatch.setattr(cli, "_build_case", lambda args: case)
    monkeypatch.setattr(
        cli,
        "_run_config",
        lambda config: (
            recorded.update(
                enabled=config.logging.enabled, verbosity=config.logging.verbosity
            )
            or {"case": case.name}
        ),
    )

    exit_code = cli.main(["run", "hartmann", "--output", str(tmp_path), "--verbose"])

    assert exit_code == 0
    assert recorded["enabled"] is True
    assert recorded["verbosity"] == "debug"


def test_run_branch_explicit_verbosity_overrides_default(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    case = cli._build_case(
        SimpleNamespace(case="hartmann", ha=5.0, output=str(tmp_path))
    )
    recorded: dict[str, object] = {}

    monkeypatch.setattr(cli, "_build_case", lambda args: case)
    monkeypatch.setattr(
        cli,
        "_run_config",
        lambda config: (
            recorded.update(
                enabled=config.logging.enabled, verbosity=config.logging.verbosity
            )
            or {"case": case.name}
        ),
    )

    exit_code = cli.main(
        ["run", "hartmann", "--output", str(tmp_path), "--verbosity", "normal"]
    )

    assert exit_code == 0
    assert recorded["enabled"] is True
    assert recorded["verbosity"] == "normal"
