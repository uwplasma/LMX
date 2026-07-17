import json
from pathlib import Path
import importlib.util
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from lmx.reference_data import default_fringing_pipe_reference_root


pytestmark = pytest.mark.unit


def _fringing_pipe_root_or_skip() -> Path:
    root = default_fringing_pipe_reference_root()
    if not root.exists():
        pytest.skip("optional FreeMHD fringing-pipe reference data are not available")
    return root


def _load_example_module(filename: str):
    module_path = Path(__file__).resolve().parents[1] / "examples" / filename
    assert module_path.is_file(), f"Missing example {filename}"
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fringing_benchmark_demo_runs_real_bounded_diagnostic(tmp_path: Path):
    script = Path(__file__).resolve().parents[1] / "examples/fringing_benchmark_demo.py"
    subprocess.run([sys.executable, script], cwd=tmp_path, timeout=30, check=True)
    summary_path = next((tmp_path / "artifacts").rglob("fringing_benchmark_summary.json"))
    summary = json.loads(summary_path.read_text())
    assert summary["status"] == "research-stage internal diagnostic"
    assert summary["geometry_kind"] == "rect_duct"
    assert summary["shape"] == [7, 8, 8]
    assert summary["validation"]["station_count"] == 7
    assert summary["validation"]["max_charge_balance_residual"] < 1.0e-10
    assert len(summary["mean_velocity"]) == len(summary["field_scale"]) == 7
    assert all((summary_path.parent / name).is_file() for name in summary["plots"])


def test_variable_field_extruded_demo_writes_summary(tmp_path: Path):
    script = Path(__file__).resolve().parents[1] / "examples/variable_field_extruded_demo.py"
    subprocess.run([sys.executable, script], cwd=tmp_path, timeout=60, check=True)
    summary_path = next((tmp_path / "artifacts").rglob("variable_field_extruded_summary.json"))
    summary = json.loads(summary_path.read_text())
    validation = summary["validation"]
    assert summary["case"].startswith("variable_field_duct")
    assert summary["geometry_kind"] == "rect_duct"
    assert validation["finite_velocity"] is True
    assert validation["mean_velocity_change"] > 0.0
    assert validation["current_proxy_change"] > 0.0
    assert isinstance(validation["validation_pass"], bool)
    assert (summary_path.parent / "extruded_overview.png").exists()


def test_extruded_restart_demo_is_state_exact(tmp_path: Path):
    script = Path(__file__).resolve().parents[1] / "examples/extruded_restart_demo.py"
    subprocess.run([sys.executable, script], cwd=tmp_path, timeout=60, check=True)
    summary_path = next((tmp_path / "artifacts").rglob("extruded_restart_summary.json"))
    summary = json.loads(summary_path.read_text())
    assert summary["max_state_difference"] == 0.0
    assert summary["max_mean_velocity_difference"] == 0.0
    assert summary["max_charge_balance_difference"] == 0.0
    assert (summary_path.parent / "extruded_restart_demo.png").is_file()


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
