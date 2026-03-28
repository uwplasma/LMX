from pathlib import Path

import pytest

from scripts import sample_freemhd_profiles as sampler


pytestmark = pytest.mark.unit


def test_write_sample_dict_creates_expected_content(tmp_path: Path):
    case_dir = tmp_path / "case"
    (case_dir / "system").mkdir(parents=True)
    path = sampler.write_sample_dict(
        case_dir=case_dir,
        dict_name="lmxSampleDict",
        x_position=0.015,
        y_min=-0.1,
        y_max=0.1,
        z_min=-0.1,
        z_max=0.1,
        n_points=201,
    )
    text = path.read_text()
    assert path.name == "lmxSampleDict"
    assert "class       dictionary;" in text
    assert "object      lmxSampleDict;" in text
    assert "start (0.015 -0.1 0.0);" in text
    assert "end (0.015 0.1 0.0);" in text
    assert "fields          (U potE);" in text


def test_run_postprocess_sampling_uses_expected_docker_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    recorded = {}

    def fake_run(command, text, capture_output, check):
        recorded["command"] = command
        return sampler.subprocess.CompletedProcess(args=command, returncode=0, stdout="done", stderr="")

    monkeypatch.setattr(sampler.subprocess, "run", fake_run)
    sampler.run_postprocess_sampling(
        image="microfluidica/openfoam:2206",
        case_dir=case_dir,
        region="liquid",
        time="0.0001",
        dict_name="lmxSampleDict",
        platform="linux/amd64",
    )

    command = recorded["command"]
    assert command[:5] == ["docker", "run", "--rm", "--platform", "linux/amd64"]
    assert "microfluidica/openfoam:2206" in command
    assert "postProcess -region liquid -func lmxSampleDict -time 0.0001" in command[-1]


def test_main_reports_sample_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    case_dir = tmp_path / "case"
    (case_dir / "system").mkdir(parents=True)
    sample_root = case_dir / "postProcessing" / "lmxSampleDict" / "liquid" / "0.0001"
    sample_root.mkdir(parents=True)
    sample_rows = "0.0 0.0 0.0 0.0 0.0\n1.0 0.0 1.0 0.0 0.0\n"
    (sample_root / "centerlineY_potE_U.xy").write_text(sample_rows)
    (sample_root / "centerlineZ_potE_U.xy").write_text(sample_rows)

    def fake_run_postprocess(**kwargs):
        return sampler.subprocess.CompletedProcess(args=["docker"], returncode=0, stdout="done", stderr="")

    monkeypatch.setattr(sampler, "run_postprocess_sampling", fake_run_postprocess)

    output = tmp_path / "sample.json"
    exit_code = sampler.main(
        [
            "--case-dir",
            str(case_dir),
            "--output",
            str(output),
        ]
    )

    payload = output.read_text()
    assert exit_code == 0
    assert '"status": "ok"' in capsys.readouterr().out
    assert "centerlineY_potE_U.xy" in payload
