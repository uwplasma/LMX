from pathlib import Path
import importlib.util
from types import SimpleNamespace
import json

import numpy as np
import pytest

import lmx.example_runner as example_runner
from lmx.example_runner import run_case_example
from lmx.reference_data import default_fringing_pipe_reference_root


pytestmark = pytest.mark.unit


def _fringing_pipe_root_or_skip() -> Path:
    root = default_fringing_pipe_reference_root()
    if not root.exists():
        pytest.skip("optional FreeMHD fringing-pipe reference data are not available")
    return root


def _stub_case_example_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, case_name: str
) -> Path:
    solution = SimpleNamespace(
        state=SimpleNamespace(
            u=np.array([[1.0]]), phi=np.array([[0.0]]), time=0.0, residual=0.0
        ),
        mesh=SimpleNamespace(),
        case_name=case_name,
    )
    reference_root = tmp_path / "refs"
    reference_root.mkdir()
    monkeypatch.setattr(example_runner, "solve_steady", lambda case: solution)
    monkeypatch.setattr(example_runner, "write_paraview", lambda solution, out_dir: [])
    monkeypatch.setattr(example_runner, "write_profile_csv", lambda path, profile: path)
    monkeypatch.setattr(
        example_runner, "extract_centerline", lambda solution: {"y": [0.0], "u": [1.0]}
    )
    monkeypatch.setattr(
        example_runner,
        "extract_midplane_profile",
        lambda solution, axis, fluid_only=True: {"coord": [0.0], "u": [1.0]},
    )
    monkeypatch.setattr(
        example_runner,
        "validation_summary",
        lambda solution, name, ha: {"u_max": 1.0},
    )
    monkeypatch.setattr(
        example_runner,
        "write_case_overview_plots",
        lambda solution, out_dir, **kwargs: [out_dir / "overview.png"],
    )
    monkeypatch.setattr(
        example_runner, "write_metrics_json", lambda payload, path: path.write_text("{}")
    )
    return reference_root


def test_run_case_example_writes_hartmann_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fake_solution = SimpleNamespace(
        state=SimpleNamespace(
            u=np.array([[1.0]]), phi=np.array([[0.0]]), time=0.0, residual=0.0
        ),
        mesh=SimpleNamespace(),
        case_name="hartmann_ha5",
    )

    monkeypatch.setattr(example_runner, "solve_steady", lambda case: fake_solution)
    monkeypatch.setattr(
        example_runner,
        "write_paraview",
        lambda solution, out_dir: (out_dir / "hartmann_ha5.vtr").write_text("vtk"),
    )
    monkeypatch.setattr(
        example_runner,
        "write_profile_csv",
        lambda path, profile: path.write_text("coord,u\n0,1\n"),
    )
    monkeypatch.setattr(
        example_runner, "extract_centerline", lambda solution: {"y": [0.0], "u": [1.0]}
    )
    monkeypatch.setattr(
        example_runner,
        "extract_midplane_profile",
        lambda solution, axis, fluid_only=True: {
            "y" if axis == "y" else "z": [0.0],
            "u": [1.0],
        },
    )
    monkeypatch.setattr(
        example_runner,
        "validation_summary",
        lambda solution, case_name, ha: {"u_max": 1.0},
    )
    monkeypatch.setattr(
        example_runner,
        "hartmann_validation",
        lambda solution, ha: SimpleNamespace(
            coordinate=np.array([0.0]),
            reference=np.array([1.0]),
            l2_error=0.0,
            linf_error=0.0,
        ),
    )

    def fake_write_case_overview_plots(solution, out_dir: Path, **kwargs):
        outputs = [
            out_dir / "overview.png",
            out_dir / "overview.pdf",
            out_dir / "diagnostics.png",
            out_dir / "diagnostics.pdf",
        ]
        for path in outputs:
            path.write_bytes(b"plot")
        return outputs

    monkeypatch.setattr(
        example_runner, "write_case_overview_plots", fake_write_case_overview_plots
    )
    monkeypatch.setattr(
        example_runner,
        "write_metrics_json",
        lambda payload, path: path.write_text("{}"),
    )

    report = run_case_example(
        case_kind="hartmann",
        ha=5.0,
        ny=8,
        nz=8,
        out_dir=tmp_path,
        reference_root=None,
    )

    assert report["case"] == "hartmann_ha5"
    assert (tmp_path / "overview.png").exists()
    assert (tmp_path / "overview.pdf").exists()
    assert (tmp_path / "diagnostics.png").exists()
    assert (tmp_path / "diagnostics.pdf").exists()
    assert (tmp_path / "example_report.json").exists()
    assert (tmp_path / "hartmann_ha5.vtr").exists()
    assert (tmp_path / "hartmann_ha5_centerline.csv").exists()


def _load_example_module(filename: str):
    root = Path(__file__).resolve().parents[1]
    candidates = [root / "examples" / filename]
    existing = [path for path in candidates if path.is_file()]
    assert len(existing) == 1, (
        f"Expected one workflow named {filename}, found {existing}"
    )
    module_path = existing[0]
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fringing_benchmark_demo_writes_extruded_bundle_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    module = _load_example_module("fringing_benchmark_demo.py")

    monkeypatch.setattr(
        module,
        "build_square_duct_extruded_problem",
        lambda **kwargs: SimpleNamespace(
            case=SimpleNamespace(
                name="fringing_case",
                solver=SimpleNamespace(kind="extruded_inductionless"),
            ),
            profile=SimpleNamespace(
                x=np.array([0.0, 1.0]), field_scale=np.array([0.0, 1.0]), axis="z"
            ),
        ),
    )
    monkeypatch.setattr(
        module,
        "build_layered_duct_extruded_problem",
        lambda **kwargs: SimpleNamespace(
            case=SimpleNamespace(
                name="fringing_case_layered",
                solver=SimpleNamespace(kind="extruded_inductionless"),
            ),
            profile=SimpleNamespace(
                x=np.array([0.0, 1.0]), field_scale=np.array([0.0, 1.0]), axis="z"
            ),
        ),
    )
    monkeypatch.setattr(
        module,
        "build_pipe_ogrid_extruded_problem",
        lambda **kwargs: SimpleNamespace(
            case=SimpleNamespace(
                name="fringing_case_pipe",
                solver=SimpleNamespace(kind="extruded_inductionless"),
            ),
            profile=SimpleNamespace(
                x=np.array([0.0, 1.0]), field_scale=np.array([0.0, 1.0]), axis="z"
            ),
        ),
    )
    monkeypatch.setattr(
        module,
        "solve_extruded_inductionless",
        lambda *args, **kwargs: SimpleNamespace(
            station_history=(
                {
                    "x": 0.0,
                    "field_scale": 0.0,
                    "mean_velocity": 1.0,
                    "u_max": 1.1,
                    "pressure_span": 0.3,
                    "axial_current": 0.01,
                    "current_scaled_pressure_proxy": 0.2,
                },
                {
                    "x": 1.0,
                    "field_scale": 1.0,
                    "mean_velocity": 0.8,
                    "u_max": 0.9,
                    "pressure_span": 0.2,
                    "axial_current": 0.015,
                    "current_scaled_pressure_proxy": 0.25,
                },
            ),
            bundle=SimpleNamespace(
                x=np.array([0.0, 1.0]),
                y=np.array([-1.0, 1.0]),
                z=np.array([-1.0, 1.0]),
                field_scale=np.array([0.0, 1.0]),
                u=np.ones((2, 2, 2)),
                charge_balance_residual=np.array([1.0e-7, 2.0e-7]),
                wall_current_leakage=np.array([1.0e-8, 2.0e-8]),
                axial_current=np.array([0.01, 0.015]),
            ),
            validation=SimpleNamespace(
                station_count=2,
                max_residual=1.0e-6,
                max_charge_balance_residual=2.0e-7,
                mean_velocity_span=0.2,
                volumetric_flow_rate_span=0.1,
                axial_current_span=0.01,
                peak_velocity_span=0.03,
                pressure_span_range=0.04,
                max_wall_current_leakage=2.0e-8,
                net_boundary_current_residual=3.0e-8,
                field_mean_velocity_correlation=0.9,
            ),
        ),
    )

    summary = module.run_fringing_benchmark_demo(
        out_dir=tmp_path, nx_stations=2, ny=4, nz=4
    )
    assert summary["case"] == "fringing_case"
    assert "extruded_bundle" in summary
    assert "validation" in summary
    assert "pressure_span" in summary["extruded_bundle"]
    assert "u_peak" in summary["extruded_bundle"]
    assert (tmp_path / "fringing_benchmark.png").exists()
    assert (tmp_path / "fringing_benchmark_summary.json").exists()

    summary_layered = module.run_fringing_benchmark_demo(
        out_dir=tmp_path / "layered",
        geometry_kind="layered_duct",
        nx_stations=2,
        ny=4,
        nz=4,
    )
    assert summary_layered["case"] == "fringing_case_layered"
    assert summary_layered["geometry_kind"] == "layered_duct"

    summary_pipe = module.run_fringing_benchmark_demo(
        out_dir=tmp_path / "pipe",
        geometry_kind="pipe_ogrid",
        nx_stations=2,
        ny=4,
        nz=4,
    )
    assert summary_pipe["case"] == "fringing_case_pipe"
    assert summary_pipe["geometry_kind"] == "pipe_ogrid"


def test_variable_field_extruded_demo_writes_summary(tmp_path: Path):
    module = _load_example_module("variable_field_extruded_demo.py")
    module.OUTPUT_DIR = tmp_path
    module.NY = 16
    module.NZ = 16
    module.NX_STATIONS = 9
    summary = module.run_variable_field_extruded_demo()
    validation = summary["validation"]
    assert summary["case"] == "variable_field_extruded"
    assert summary["geometry_kind"] == "rect_duct"
    assert validation["finite_velocity"] is True
    assert validation["mean_velocity_change"] > 0.0
    assert validation["current_proxy_change"] > 0.0
    assert isinstance(validation["validation_pass"], bool)
    assert (tmp_path / "extruded_overview.png").exists()
    assert (tmp_path / "variable_field_extruded_summary.json").exists()


def test_operator_verification_demo_writes_summary(tmp_path: Path):
    module = _load_example_module("operator_verification_demo.py")
    module.OUTPUT_DIR = tmp_path
    module.RESOLUTIONS = (12, 24, 48)
    summary = module.run_operator_verification_demo()
    assert summary["case"] == "operator_verification"
    assert summary["observed_order"]["gradient_y"] > 1.8
    assert summary["observed_order"]["gradient_z"] > 1.8
    assert (tmp_path / "operator_verification.png").exists()
    assert (tmp_path / "operator_verification_summary.json").exists()


def test_extruded_restart_demo_is_state_exact(tmp_path: Path):
    module = _load_example_module("extruded_restart_demo.py")
    summary = module.run_extruded_restart_demo(out_dir=tmp_path)
    assert summary["max_state_difference"] == 0.0
    assert summary["max_mean_velocity_difference"] == 0.0
    assert summary["max_charge_balance_difference"] == 0.0
    assert (tmp_path / "extruded_restart_demo.png").is_file()


def test_freemhd_closed_channel_observable_parity_writes_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    module = _load_example_module("freemhd_closed_channel_observable_parity.py")
    module.OUTPUT_DIR = tmp_path
    monkeypatch.setattr(
        module,
        "_observable_record",
        lambda case_kind, **_kwargs: {
            "case_kind": case_kind,
            "observables": {
                name: {
                    "y": {
                        "coordinate": [-1.0, 0.0, 1.0],
                        "reference": [0.0, 1.0, 0.0],
                        "simulated": [0.0, 0.98, 0.0],
                        "l2_error": 2.0e-2,
                        "linf_error": 2.0e-2,
                        "reference_peak_abs": 1.0e-8 if name == "potential" else 1.0,
                    },
                    "z": {
                        "coordinate": [-1.0, 0.0, 1.0],
                        "reference": [0.0, 1.0, 0.0],
                        "simulated": [0.0, 0.97, 0.0],
                        "l2_error": 3.0e-2,
                        "linf_error": 3.0e-2,
                    },
                    "peak_ratio": 0.99,
                }
                for name in (
                    ("velocity", "potential", "current")
                    if case_kind == "shercliff"
                    else ("velocity", "potential", "current", "lorentz")
                )
            },
        },
    )
    monkeypatch.setattr(
        module,
        "write_freemhd_observable_parity_plots",
        lambda records, out_dir, case_title: [
            Path(out_dir) / "freemhd_closed_channel_observable_parity.png",
            Path(out_dir) / "freemhd_closed_channel_observable_parity.pdf",
        ],
    )
    (tmp_path / "freemhd_closed_channel_observable_parity.png").write_bytes(b"img")
    (tmp_path / "freemhd_closed_channel_observable_parity.pdf").write_bytes(b"pdf")
    summary = module.run_freemhd_closed_channel_observable_parity()
    assert summary["case"] == "freemhd_closed_channel_observable_parity"
    assert len(summary["records"]) == 2
    jets = module._compare_side_jets([-1, -0.7, 0, 0.7, 1], [0, 1.2, 1, 1.3, 0], [-1, -0.7, 0, 0.7, 1], [0, 1.4, 1, 1.4, 0])
    assert jets["normalized_location_error"] == 0.0
    assert summary["observable_gate"]["low_signal_count"] == 2
    assert summary["observable_gate"]["missing_observable_count"] == 1
    assert summary["observable_gate"]["research_grade_validation_pass"] is False
    assert (tmp_path / "freemhd_closed_channel_observable_parity_summary.json").exists()
    cli_output = tmp_path / "cli"
    assert module.main([
        "--reference-root", str(tmp_path / "references"),
        "--output", str(cli_output),
    ]) == 0
    assert (cli_output / "freemhd_closed_channel_observable_parity_summary.json").is_file()
    with pytest.raises(SystemExit, match="0"):
        module.main(["--help"])
    assert "--reference-root" in capsys.readouterr().out


def test_freemhd_continuum_velocity_audit_keeps_shared_scale_and_endpoint_disclosure(
    monkeypatch: pytest.MonkeyPatch,
):
    module = _load_example_module("freemhd_closed_channel_observable_parity.py")
    analytical = SimpleNamespace(
        coordinate=np.asarray([-1.0, 0.0, 1.0]),
        midplane_y=np.asarray([-0.1, 2.0, -0.1]),
        midplane_z=np.asarray([0.0, 1.0, 0.0]),
        path="analytical.txt",
    )
    monkeypatch.setattr(
        module, "load_closed_channel_analytical", lambda *args, **kwargs: analytical
    )
    observables = {
        "velocity": {
            "y": {
                "coordinate": [-1.0, 0.0, 1.0],
                "simulated": [0.0, 1.0, 0.0],
                "reference": [0.0, 0.9, 0.0],
            },
            "z": {
                "coordinate": [-1.0, 0.0, 1.0],
                "simulated": [0.0, 0.5, 0.0],
                "reference": [0.0, 0.45, 0.0],
            },
        }
    }

    audit = module._continuum_velocity_audit(
        "shercliff",
        observables,
        ha=20,
        length_scale=1.0,
        velocity_scale=2.0,
        reference_root=Path("references"),
    )

    assert audit["reference_path"] == "analytical.txt"
    assert audit["axes"]["y"]["analytical_endpoint_values"] == pytest.approx(
        [-0.05, -0.05]
    )
    assert audit["axes"]["y"]["lmx_no_slip_endpoint_corrected_analytical"][
        "l2_error"
    ] == pytest.approx(0.0)
    assert audit["axes"]["y"]["processed_freemhd_raw_analytical"]["l2_error"] > 0.0


@pytest.mark.external
def test_pipe_reference_comparison_demo_writes_summary(tmp_path: Path):
    _fringing_pipe_root_or_skip()
    module = _load_example_module("pipe_reference_comparison_demo.py")
    summary = module.run_pipe_reference_comparison_demo(
        out_dir=tmp_path,
        nr=4,
        ntheta=16,
        nx_stations=4,
        max_steps=4,
        coupling_iterations=4,
        potential_iterations=16,
    )
    assert summary["geometry_kind"] == "pipe_ogrid"
    assert summary["normalization"]["center"] == "shared_peak_axial_velocity"
    assert (tmp_path / "pipe_reference_comparison.png").exists()
    assert (tmp_path / "pipe_reference_comparison_summary.json").exists()


def test_build_case_rejects_unknown_kind():
    with pytest.raises(ValueError, match="Unsupported case kind"):
        example_runner._build_case("bad", 5.0, 8, 8)


def test_portable_path_prefers_relative_and_falls_back_to_name(tmp_path: Path):
    nested = tmp_path / "nested" / "file.json"
    nested.parent.mkdir()
    nested.write_text("{}")

    assert (
        example_runner._portable_path(nested, relative_to=tmp_path)
        == "nested/file.json"
    )
    assert (
        example_runner._portable_path("/outside/path/file.json", relative_to=tmp_path)
        == "file.json"
    )


def test_solve_case_snapshots_records_fully_developed_frames(
    monkeypatch: pytest.MonkeyPatch,
):
    case = example_runner._build_case("hartmann", 5.0, 6, 6)

    def fake_step(**kwargs):
        u_prev = kwargs["u_previous"]
        updated = np.asarray(u_prev) + 0.1
        zeros = np.zeros_like(updated)
        return (
            updated,
            zeros,
            zeros,
            zeros,
            zeros,
            1.0e-6,
            1.0e-6,
            2,
            1.0e-6,
            3,
            0.2,
            0.1,
            0.05,
            float(np.mean(updated)),
            0.3,
            1.0e-4,
            1.0e-4,
        )

    monkeypatch.setattr(example_runner.solvers, "_fully_developed_case_step", fake_step)
    frames = example_runner.solve_case_snapshots(case, frame_count=3)
    assert len(frames) >= 3
    assert frames[-1]["time"] > frames[0]["time"]
    assert "pressure_proxy" in frames[0]
    assert "face_lorentz_max" in frames[0]


def test_run_case_example_cli_prints_report(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], tmp_path: Path
):
    monkeypatch.setattr(
        example_runner,
        "run_case_example",
        lambda **kwargs: {
            "case": "hartmann_ha5",
            "plots": [],
            "metrics": {"u_max": 1.0},
            "output_dir": str(tmp_path),
        },
    )
    exit_code = example_runner.run_case_example_cli(
        case_kind="hartmann",
        ha=5.0,
        ny=8,
        nz=8,
        out_dir=tmp_path,
        reference_root=None,
    )
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["case"] == "hartmann_ha5"


def test_run_case_example_uses_reference_data_when_available(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    reference_root = _stub_case_example_runtime(
        tmp_path, monkeypatch, case_name="shercliff_ha5"
    )
    monkeypatch.setattr(
        example_runner,
        "closed_channel_validation",
        lambda solution, case_name, ha, reference_root: SimpleNamespace(
            y_profile=SimpleNamespace(
                coordinate=np.array([0.0]), reference=np.array([1.0]), l2_error=0.1
            ),
            z_profile=SimpleNamespace(
                coordinate=np.array([0.0]), reference=np.array([0.9]), l2_error=0.2
            ),
            reference_path=Path("analytical.csv"),
        ),
    )
    report = run_case_example(
        case_kind="shercliff",
        ha=5.0,
        ny=8,
        nz=8,
        out_dir=tmp_path,
        reference_root=reference_root,
    )

    assert report["reference"]["available"] is True
    assert report["reference"]["kind"] == "closed_channel_analytical"


def test_run_case_example_handles_missing_reference_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    reference_root = _stub_case_example_runtime(
        tmp_path, monkeypatch, case_name="hunt_ha5"
    )
    monkeypatch.setattr(
        example_runner,
        "closed_channel_validation",
        lambda *args, **kwargs: (_ for _ in ()).throw(FileNotFoundError("missing")),
    )
    report = run_case_example(
        case_kind="hunt",
        ha=5.0,
        ny=8,
        nz=8,
        out_dir=tmp_path,
        reference_root=reference_root,
    )

    assert report["reference"]["available"] is False
