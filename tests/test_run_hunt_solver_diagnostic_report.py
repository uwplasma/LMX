from pathlib import Path
from types import SimpleNamespace
import json

import jax.numpy as jnp
import pytest

from lmx.core import Diagnostics, MHDState, Solution
from lmx.mesh import generate_rect_duct_mesh
from scripts import run_hunt_solver_diagnostic_report as huntdiag


pytestmark = pytest.mark.unit


def _fake_solution() -> Solution:
    mesh = generate_rect_duct_mesh(width=2.0, height=2.0, ny=3, nz=3)
    u = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    zeros = jnp.zeros_like(u)
    state = MHDState(u=u, phi=zeros, jy=zeros, jz=zeros, lorentz_x=zeros, time=0.1, residual=1e-4)
    diagnostics = Diagnostics(
        residual_history=jnp.asarray([1e-4]),
        courant_like=jnp.asarray([0.1]),
        ohmic_power=jnp.asarray([0.2]),
        mean_velocity_history=jnp.asarray([0.9]),
        applied_forcing_history=jnp.asarray([0.3]),
        pressure_proxy_history=jnp.asarray([0.4]),
        current_max_history=jnp.asarray([3.0]),
        face_current_max_history=jnp.asarray([5.0]),
        emf_max_history=jnp.asarray([2.0]),
        lorentz_max_history=jnp.asarray([4.0]),
        face_lorentz_max_history=jnp.asarray([4.5]),
        potential_residual_history=jnp.asarray([1e-3]),
        potential_iterations_history=jnp.asarray([123.0]),
    )
    return Solution(mesh=mesh, state=state, diagnostics=diagnostics, case_name="hunt_ha20")


def test_hunt_solver_diagnostic_report_writes_solver_first_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run_dir = tmp_path / "run"
    (run_dir / "0" / "liquid").mkdir(parents=True)
    (run_dir / "0" / "liquid" / "U").write_text("internalField   uniform ( 0.1175 0 0 );\n")
    (run_dir / "system").mkdir()
    (run_dir / "constant").mkdir()
    (run_dir / "postProcessing" / "liquid" / "minMax" / "0").mkdir(parents=True)
    (run_dir / "postProcessing" / "sampleDict" / "liquid" / "0.0001").mkdir(parents=True)
    (run_dir / "postProcessing" / "liquid" / "minMax" / "0" / "fieldMinMax.dat").write_text(
        "# header\n0.0001 mag(U) 0.0 (0 0 0) 0 0.25 (0 0 0) 0\n"
    )
    (run_dir / "system" / "controlDict").write_text("BtStartTime 1e-5;\nBtDuration 2e-4;\n")
    sample_lines = "0.0 0.0 0.0 0.0 0.0\n1.0 0.0 1.0 0.0 0.0\n2.0 0.0 0.0 0.0 0.0\n"
    (run_dir / "postProcessing" / "sampleDict" / "liquid" / "0.0001" / "centerlineY_potE_U.xy").write_text(sample_lines)
    (run_dir / "postProcessing" / "sampleDict" / "liquid" / "0.0001" / "centerlineZ_potE_U.xy").write_text(sample_lines)

    monkeypatch.setattr(huntdiag, "solve_steady", lambda case: _fake_solution())

    output = tmp_path / "diagnostics.json"
    exit_code = huntdiag.main(
        [
            "--freemhd-run-dir",
            str(run_dir),
            "--ha",
            "20",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text())
    assert payload["case_kind"] == "hunt"
    assert "lmx_solver" in payload
    assert "freemhd_run" in payload
    assert "comparison" in payload
    assert payload["lmx_solver"]["diagnostics"]["potential_residual"] == pytest.approx(0.001)
    assert payload["lmx_solver"]["magnetic_field"]["ramp_duration"] == pytest.approx(2e-4)
    assert payload["lmx_solver"]["trace"]["time_history"] == []
    assert payload["lmx_solver"]["trace"]["u_max_history"] == []
    assert payload["lmx_solver"]["trace"]["mean_velocity_history"] == pytest.approx([0.9])
    assert payload["lmx_solver"]["trace"]["applied_forcing_history"] == pytest.approx([0.3])
    assert payload["lmx_solver"]["trace"]["pressure_proxy_history"] == pytest.approx([0.4])
    assert payload["lmx_solver"]["trace"]["current_scaled_pressure_proxy_history"] == pytest.approx([0.4])
    assert payload["lmx_solver"]["trace"]["current_max_history"] == [3.0]
    assert payload["lmx_solver"]["trace"]["face_current_max_history"] == [5.0]
    assert payload["lmx_solver"]["trace"]["emf_max_history"] == [2.0]
    assert payload["lmx_solver"]["trace"]["lorentz_max_history"] == [4.0]
    assert payload["lmx_solver"]["trace"]["face_lorentz_max_history"] == [4.5]
    assert payload["comparison"]["u_max_abs_diff"] == pytest.approx(0.75)
    assert payload["comparison"]["sample_combined_l2_error"] < 1e-6


def test_hunt_solver_diagnostic_report_accepts_cg_volume_backend(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run_dir = tmp_path / "run"
    (run_dir / "0" / "liquid").mkdir(parents=True)
    (run_dir / "0" / "liquid" / "U").write_text("internalField   uniform ( 0.1175 0 0 );\n")
    (run_dir / "system").mkdir()
    (run_dir / "constant").mkdir()
    (run_dir / "postProcessing" / "liquid" / "minMax" / "0").mkdir(parents=True)
    (run_dir / "postProcessing" / "liquid" / "minMax" / "0" / "fieldMinMax.dat").write_text(
        "# header\n0.0001 mag(U) 0.0 (0 0 0) 0 0.25 (0 0 0) 0\n"
    )
    (run_dir / "system" / "controlDict").write_text("BtStartTime 1e-5;\nBtDuration 2e-4;\n")

    captured = {}

    def fake_solve(case):
        captured["potential_solver"] = case.time_stepper.potential_solver
        return _fake_solution()

    monkeypatch.setattr(huntdiag, "solve_steady", fake_solve)

    output = tmp_path / "diagnostics.json"
    exit_code = huntdiag.main(
        [
            "--freemhd-run-dir",
            str(run_dir),
            "--ha",
            "20",
            "--potential-solver",
            "cg_volume",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert captured["potential_solver"] == "cg_volume"


def test_hunt_solver_diagnostic_report_accepts_current_reconstruction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run_dir = tmp_path / "run"
    (run_dir / "0" / "liquid").mkdir(parents=True)
    (run_dir / "0" / "liquid" / "U").write_text("internalField   uniform ( 0.1175 0 0 );\n")
    (run_dir / "system").mkdir()
    (run_dir / "constant").mkdir()
    (run_dir / "postProcessing" / "liquid" / "minMax" / "0").mkdir(parents=True)
    (run_dir / "postProcessing" / "liquid" / "minMax" / "0" / "fieldMinMax.dat").write_text(
        "# header\n0.0001 mag(U) 0.0 (0 0 0) 0 0.25 (0 0 0) 0\n"
    )
    (run_dir / "system" / "controlDict").write_text("BtStartTime 1e-5;\nBtDuration 2e-4;\n")

    captured = {}

    def fake_solve(case):
        captured["current_reconstruction"] = case.time_stepper.current_reconstruction
        return _fake_solution()

    monkeypatch.setattr(huntdiag, "solve_steady", fake_solve)

    output = tmp_path / "diagnostics.json"
    exit_code = huntdiag.main(
        [
            "--freemhd-run-dir",
            str(run_dir),
            "--ha",
            "20",
            "--current-reconstruction",
            "face_averaged",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert captured["current_reconstruction"] == "face_averaged"


def test_hunt_solver_diagnostic_report_accepts_hybrid_face_lorentz_reconstruction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run_dir = tmp_path / "run"
    (run_dir / "0" / "liquid").mkdir(parents=True)
    (run_dir / "0" / "liquid" / "U").write_text("internalField   uniform ( 0.1175 0 0 );\n")
    (run_dir / "system").mkdir()
    (run_dir / "constant").mkdir()
    (run_dir / "postProcessing" / "liquid" / "minMax" / "0").mkdir(parents=True)
    (run_dir / "postProcessing" / "liquid" / "minMax" / "0" / "fieldMinMax.dat").write_text(
        "# header\n0.0001 mag(U) 0.0 (0 0 0) 0 0.25 (0 0 0) 0\n"
    )
    (run_dir / "system" / "controlDict").write_text("BtStartTime 1e-5;\nBtDuration 2e-4;\n")

    captured = {}

    def fake_solve(case):
        captured["current_reconstruction"] = case.time_stepper.current_reconstruction
        return _fake_solution()

    monkeypatch.setattr(huntdiag, "solve_steady", fake_solve)

    output = tmp_path / "diagnostics.json"
    exit_code = huntdiag.main(
        [
            "--freemhd-run-dir",
            str(run_dir),
            "--ha",
            "20",
            "--current-reconstruction",
            "hybrid_face_lorentz",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert captured["current_reconstruction"] == "hybrid_face_lorentz"


def test_hunt_solver_diagnostic_report_adds_inlet_velocity_boundary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run_dir = tmp_path / "run"
    (run_dir / "0" / "liquid").mkdir(parents=True)
    (run_dir / "0" / "liquid" / "U").write_text("internalField   uniform ( 0.1175 0 0 );\n")
    (run_dir / "system").mkdir()
    (run_dir / "constant").mkdir()
    (run_dir / "postProcessing" / "liquid" / "minMax" / "0").mkdir(parents=True)
    (run_dir / "postProcessing" / "liquid" / "minMax" / "0" / "fieldMinMax.dat").write_text(
        "# header\n0.0001 mag(U) 0.0 (0 0 0) 0 0.25 (0 0 0) 0\n"
    )
    (run_dir / "system" / "controlDict").write_text("BtStartTime 1e-5;\nBtDuration 2e-4;\n")

    captured = {}

    def fake_solve(case):
        captured["boundary_kinds"] = [bc.kind for bc in case.boundary_conditions]
        return _fake_solution()

    monkeypatch.setattr(huntdiag, "solve_steady", fake_solve)

    output = tmp_path / "diagnostics.json"
    exit_code = huntdiag.main(
        [
            "--freemhd-run-dir",
            str(run_dir),
            "--ha",
            "20",
            "--drive-mode",
            "inlet_velocity",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert "inlet_velocity" in captured["boundary_kinds"]


def test_hunt_solver_diagnostic_report_auto_infers_flow_rate_inlet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run_dir = tmp_path / "run"
    (run_dir / "0" / "liquid").mkdir(parents=True)
    (run_dir / "0" / "liquid" / "U").write_text(
        """internalField   uniform ( 0.1175 0 0 );

boundaryField
{
    inlet
    {
        type            flowRateInletVelocity;
        value           uniform ( 0.1175 0 0 );
        volumetricFlowRate 0.0047;
    }
}
"""
    )
    (run_dir / "system").mkdir()
    (run_dir / "constant").mkdir()
    (run_dir / "postProcessing" / "liquid" / "minMax" / "0").mkdir(parents=True)
    (run_dir / "postProcessing" / "liquid" / "minMax" / "0" / "fieldMinMax.dat").write_text(
        "# header\n0.0001 mag(U) 0.0 (0 0 0) 0 0.25 (0 0 0) 0\n"
    )
    (run_dir / "system" / "controlDict").write_text("BtStartTime 1e-5;\nBtDuration 2e-4;\n")

    captured = {}

    def fake_solve(case):
        captured["boundary_kinds"] = [bc.kind for bc in case.boundary_conditions]
        captured["boundary_values"] = [bc.value for bc in case.boundary_conditions if bc.kind == "inlet_flow_rate"]
        return _fake_solution()

    monkeypatch.setattr(huntdiag, "solve_steady", fake_solve)

    output = tmp_path / "diagnostics.json"
    exit_code = huntdiag.main(
        [
            "--freemhd-run-dir",
            str(run_dir),
            "--ha",
            "20",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text())
    assert payload["drive_mode"] == "inlet_flow_rate"
    assert payload["recovered_inlet_flow_rate"] == pytest.approx(0.0047)
    assert "inlet_flow_rate" in captured["boundary_kinds"]
    assert captured["boundary_values"][0] == pytest.approx(0.0047)


def test_hunt_solver_diagnostic_report_supports_inlet_flow_rate_drive_mode(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run_dir = tmp_path / "run"
    (run_dir / "0" / "liquid").mkdir(parents=True)
    (run_dir / "0" / "liquid" / "U").write_text("internalField   uniform ( 0.1175 0 0 );\n")
    (run_dir / "system").mkdir()
    (run_dir / "constant").mkdir()
    (run_dir / "postProcessing" / "liquid" / "minMax" / "0").mkdir(parents=True)
    (run_dir / "postProcessing" / "liquid" / "minMax" / "0" / "fieldMinMax.dat").write_text(
        "# header\n0.0001 mag(U) 0.0 (0 0 0) 0 0.25 (0 0 0) 0\n"
    )
    (run_dir / "system" / "controlDict").write_text("BtStartTime 1e-5;\nBtDuration 2e-4;\n")

    captured = {}

    def fake_solve(case):
        captured["boundary_kinds"] = [bc.kind for bc in case.boundary_conditions]
        captured["boundary_values"] = [bc.value for bc in case.boundary_conditions if bc.kind == "inlet_flow_rate"]
        return _fake_solution()

    monkeypatch.setattr(huntdiag, "solve_steady", fake_solve)

    output = tmp_path / "diagnostics.json"
    exit_code = huntdiag.main(
        [
            "--freemhd-run-dir",
            str(run_dir),
            "--ha",
            "20",
            "--drive-mode",
            "inlet_flow_rate",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert "inlet_flow_rate" in captured["boundary_kinds"]
    assert captured["boundary_values"][0] == pytest.approx(0.1175 * 4.0)


def test_hunt_solver_diagnostic_report_supports_restart_npz(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    run_dir = tmp_path / "run"
    (run_dir / "0" / "liquid").mkdir(parents=True)
    (run_dir / "0" / "liquid" / "U").write_text("internalField   uniform ( 0.1175 0 0 );\n")
    (run_dir / "system").mkdir()
    (run_dir / "constant").mkdir()
    (run_dir / "postProcessing" / "liquid" / "minMax" / "0").mkdir(parents=True)
    (run_dir / "postProcessing" / "liquid" / "minMax" / "0" / "fieldMinMax.dat").write_text(
        "# header\n0.0001 mag(U) 0.0 (0 0 0) 0 0.25 (0 0 0) 0\n"
    )
    (run_dir / "system" / "controlDict").write_text("BtStartTime 1e-5;\nBtDuration 2e-4;\n")

    fake_restart = SimpleNamespace(
        path=(tmp_path / "restart_in.npz").resolve(),
        state=SimpleNamespace(time=2e-5),
        diagnostics=SimpleNamespace(),
    )
    captured = {}

    def fake_solve(case, **kwargs):
        captured["initial_state"] = kwargs.get("initial_state")
        captured["initial_diagnostics"] = kwargs.get("initial_diagnostics")
        captured["append_diagnostics"] = kwargs.get("append_diagnostics")
        return _fake_solution()

    monkeypatch.setattr(huntdiag, "load_restart_bundle", lambda path: fake_restart)
    monkeypatch.setattr(huntdiag, "validate_restart_bundle", lambda bundle, mesh, geometry_kind, case_name: None)
    monkeypatch.setattr(huntdiag, "_build_mesh", lambda case: SimpleNamespace())
    monkeypatch.setattr(huntdiag, "solve_steady", fake_solve)
    monkeypatch.setattr(huntdiag, "write_restart_npz", lambda solution, case, path: Path(path))

    output = tmp_path / "diagnostics.json"
    restart_out = tmp_path / "restart_out.npz"
    exit_code = huntdiag.main(
        [
            "--freemhd-run-dir",
            str(run_dir),
            "--ha",
            "20",
            "--restart-npz",
            str(fake_restart.path),
            "--append-histories",
            "--write-restart-npz",
            str(restart_out),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text())
    assert captured["initial_state"] is fake_restart.state
    assert captured["initial_diagnostics"] is fake_restart.diagnostics
    assert captured["append_diagnostics"] is True
    assert payload["restart"]["input"] == str(fake_restart.path)
    assert payload["restart"]["start_time"] == pytest.approx(2e-5)
    assert payload["restart"]["append_histories"] is True
    assert payload["restart"]["output"] == str(restart_out.resolve())
