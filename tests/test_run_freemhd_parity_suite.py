from pathlib import Path

import pytest

from scripts import run_freemhd_parity_suite as suite


pytestmark = pytest.mark.unit


def test_parity_suite_writes_skipped_summary_when_case_missing(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    output = tmp_path / "artifacts"
    exit_code = suite.main(["--output", str(output)])
    summary = (output / "summary.json").read_text()
    assert exit_code == 0
    assert '"status": "skipped"' in summary
    assert '"reason": "freemhd-case-unavailable"' in capsys.readouterr().out


def test_parity_suite_runs_sample_and_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    output = tmp_path / "artifacts"

    def fake_sample_main(argv):
        sample_output = output / "sample_profiles.json"
        sample_output.parent.mkdir(parents=True, exist_ok=True)
        sample_output.write_text('{"status":"ok"}')
        return 0

    def fake_parity_main(argv):
        parity_output = output / "parity_report.json"
        parity_output.write_text('{"metrics":{"u_max_abs_diff":0.1,"freemhd_sample_y_l2_error":0.2,"freemhd_sample_z_l2_error":0.3}}')
        return 0

    monkeypatch.setattr(suite.sample_profiles, "main", fake_sample_main)
    monkeypatch.setattr(suite.parity_report, "main", fake_parity_main)

    exit_code = suite.main(
        [
            "--output",
            str(output),
            "--case-dir",
            str(case_dir),
        ]
    )

    summary = (output / "summary.json").read_text()
    assert exit_code == 0
    assert '"status": "ok"' in summary
    assert '"u_max_abs_diff": 0.1' in summary
    assert '"parity_output"' in capsys.readouterr().out


def test_parity_suite_can_run_case_before_sampling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    output = tmp_path / "artifacts"

    monkeypatch.setattr(suite, "case_needs_run", lambda case_dir, time_name: True)

    def fake_run_main(argv):
        run_output = output / "run_case.json"
        run_output.parent.mkdir(parents=True, exist_ok=True)
        run_output.write_text('{"status":"ok"}')
        return 0

    def fake_sample_main(argv):
        sample_output = output / "sample_profiles.json"
        sample_output.write_text('{"status":"ok"}')
        return 0

    def fake_parity_main(argv):
        parity_output = output / "parity_report.json"
        parity_output.write_text('{"metrics":{"u_max_abs_diff":0.4}}')
        return 0

    monkeypatch.setattr(suite.freemhd_case, "main", fake_run_main)
    monkeypatch.setattr(suite.sample_profiles, "main", fake_sample_main)
    monkeypatch.setattr(suite.parity_report, "main", fake_parity_main)

    exit_code = suite.main(
        [
            "--output",
            str(output),
            "--case-dir",
            str(case_dir),
            "--run-case-if-needed",
            "--run-cores",
            "8",
        ]
    )

    summary = (output / "summary.json").read_text()
    assert exit_code == 0
    assert '"status": "ok"' in summary
    assert '"run_output"' in summary
    assert '"u_max_abs_diff": 0.4' in summary
    assert '"parity_output"' in capsys.readouterr().out


def test_parity_suite_reports_run_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    case_dir = tmp_path / "case"
    case_dir.mkdir()
    output = tmp_path / "artifacts"

    monkeypatch.setattr(suite, "case_needs_run", lambda case_dir, time_name: True)
    monkeypatch.setattr(suite.freemhd_case, "main", lambda argv: 1)

    exit_code = suite.main(
        [
            "--output",
            str(output),
            "--case-dir",
            str(case_dir),
            "--run-case-if-needed",
        ]
    )

    summary = (output / "summary.json").read_text()
    assert exit_code == 1
    assert '"status": "failed"' in summary
    assert '"reason": "run-failed"' in summary
    assert '"run_output"' in capsys.readouterr().out
