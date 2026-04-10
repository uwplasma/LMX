from pathlib import Path

import pytest

from scripts import run_reference_parity_report as parity


pytestmark = pytest.mark.unit


def test_infer_initial_velocity_x_reads_uniform_liquid_u(tmp_path: Path):
    path = tmp_path / "0" / "liquid"
    path.mkdir(parents=True)
    (path / "U").write_text("internalField   uniform ( 0.9725 0 0 );\n")
    assert parity.infer_initial_velocity_x(tmp_path) == pytest.approx(0.9725)


def test_infer_inlet_drive_mode_and_flow_rate_reads_hunt_inlet_block(tmp_path: Path):
    path = tmp_path / "0" / "liquid"
    path.mkdir(parents=True)
    (path / "U").write_text(
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
    assert parity.infer_inlet_drive_mode(tmp_path) == "inlet_flow_rate"
    assert parity.infer_inlet_flow_rate(tmp_path) == pytest.approx(0.0047)
    assert parity.infer_reduced_inlet_flow_rate(tmp_path, reduced_area=4.0, initial_velocity=0.1175) == pytest.approx(0.47)


def test_infer_magnetic_ramp_reads_control_dict(tmp_path: Path):
    system = tmp_path / "system"
    system.mkdir()
    (system / "controlDict").write_text("BtStartTime 1e-5;\nBtDuration 2e-4;\n")
    assert parity.infer_magnetic_ramp(tmp_path) == pytest.approx((1e-5, 2e-4))


def test_main_writes_parity_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    run_dir = tmp_path / "run"
    (run_dir / "0" / "liquid").mkdir(parents=True)
    (run_dir / "0" / "liquid" / "U").write_text("internalField   uniform ( 0.9725 0 0 );\n")

    # Minimal structure that keeps compare_with_reference_outputs happy.
    (run_dir / "system").mkdir()
    (run_dir / "constant").mkdir()
    (run_dir / "postProcessing" / "liquid" / "minMax" / "0").mkdir(parents=True)
    (run_dir / "postProcessing" / "liquid" / "minMax" / "0" / "fieldMinMax.dat").write_text(
        "# header\n0.1 mag(U) 0.0 (0 0 0) 0 0.25 (0 0 0) 0\n"
    )
    (run_dir / "system" / "controlDict").write_text("application epotMultiRegionFoam;\nBtStartTime 1e-5;\nBtDuration 2e-4;\n")
    (run_dir / "0" / "liquid" / "potE").write_text("internalField uniform 0;\n")

    output = tmp_path / "report.json"
    exit_code = parity.main(
        [
            "--case-kind",
            "hartmann",
            "--ha",
            "5",
            "--reference-run-dir",
            str(run_dir),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = output.read_text()
    assert '"case_name": "hartmann_ha5"' in payload
    stdout = capsys.readouterr().out
    assert '"initial_velocity": 0.9725' in stdout
    assert '"magnetic_ramp_duration": 0.0002' in stdout


def test_main_uses_hunt_inlet_drive_when_forcing_unspecified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
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

    recorded = {}

    def fake_compare(case, reference_run_dir):
        recorded["forcing"] = case.forcing
        recorded["boundary_conditions"] = case.boundary_conditions
        return parity.ValidationReport(case_name=case.name, metrics={"u_max_abs_diff": 0.1}, artifacts={})

    monkeypatch.setattr(parity, "compare_with_reference_outputs", fake_compare)

    output = tmp_path / "report.json"
    exit_code = parity.main(
        [
            "--case-kind",
            "hunt",
            "--ha",
            "20",
            "--reference-run-dir",
            str(run_dir),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert recorded["forcing"] == pytest.approx(0.0)
    flow_boundaries = [boundary for boundary in recorded["boundary_conditions"] if boundary.kind == "inlet_flow_rate"]
    assert flow_boundaries
    assert flow_boundaries[0].value == pytest.approx(0.47)
    stdout = capsys.readouterr().out
    assert '"forcing": 0.0' in stdout
    assert '"drive_mode": "inlet_flow_rate"' in stdout
    assert '"recovered_inlet_flow_rate": 0.0047' in stdout
    assert '"reduced_inlet_flow_rate": 0.47' in stdout
