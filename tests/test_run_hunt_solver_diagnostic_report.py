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
    assert payload["lmx_solver"]["trace"]["time_history"] == []
    assert payload["lmx_solver"]["trace"]["u_max_history"] == []
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
