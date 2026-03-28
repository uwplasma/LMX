from pathlib import Path

import pytest

from scripts import run_freemhd_parity_report as parity


pytestmark = pytest.mark.unit


def test_infer_initial_velocity_x_reads_uniform_liquid_u(tmp_path: Path):
    path = tmp_path / "0" / "liquid"
    path.mkdir(parents=True)
    (path / "U").write_text("internalField   uniform ( 0.9725 0 0 );\n")
    assert parity.infer_initial_velocity_x(tmp_path) == pytest.approx(0.9725)


def test_infer_parity_forcing_uses_lorentz_balance_for_hunt():
    assert parity.infer_parity_forcing("hartmann", 20.0, 0.1175) == pytest.approx(0.0)
    assert parity.infer_parity_forcing("hunt", 20.0, 0.1175) == pytest.approx(47.0)
    assert parity.infer_parity_forcing("hunt", 100.0, 0.1175) == pytest.approx(1175.0)


def test_main_writes_parity_report(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    run_dir = tmp_path / "run"
    (run_dir / "0" / "liquid").mkdir(parents=True)
    (run_dir / "0" / "liquid" / "U").write_text("internalField   uniform ( 0.9725 0 0 );\n")

    # Minimal structure that keeps compare_with_freemhd happy.
    (run_dir / "system").mkdir()
    (run_dir / "constant").mkdir()
    (run_dir / "postProcessing" / "liquid" / "minMax" / "0").mkdir(parents=True)
    (run_dir / "postProcessing" / "liquid" / "minMax" / "0" / "fieldMinMax.dat").write_text(
        "# header\n0.1 mag(U) 0.0 (0 0 0) 0 0.25 (0 0 0) 0\n"
    )
    (run_dir / "system" / "controlDict").write_text("application epotMultiRegionFoam;")
    (run_dir / "0" / "liquid" / "potE").write_text("internalField uniform 0;\n")

    output = tmp_path / "report.json"
    exit_code = parity.main(
        [
            "--case-kind",
            "hartmann",
            "--ha",
            "5",
            "--freemhd-run-dir",
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


def test_main_infers_hunt_forcing_when_unspecified(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    run_dir = tmp_path / "run"
    (run_dir / "0" / "liquid").mkdir(parents=True)
    (run_dir / "0" / "liquid" / "U").write_text("internalField   uniform ( 0.1175 0 0 );\n")

    recorded = {}

    def fake_compare(case, freemhd_run_dir):
        recorded["forcing"] = case.forcing
        return parity.ValidationReport(case_name=case.name, metrics={"u_max_abs_diff": 0.1}, artifacts={})

    monkeypatch.setattr(parity, "compare_with_freemhd", fake_compare)

    output = tmp_path / "report.json"
    exit_code = parity.main(
        [
            "--case-kind",
            "hunt",
            "--ha",
            "20",
            "--freemhd-run-dir",
            str(run_dir),
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    assert recorded["forcing"] == pytest.approx(47.0)
    stdout = capsys.readouterr().out
    assert '"forcing": 47.0' in stdout
