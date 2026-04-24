from pathlib import Path
import json

import pytest

from scripts import run_freemhd_parity_suite


pytestmark = pytest.mark.unit


def test_run_freemhd_parity_suite_skips_without_external_references(tmp_path: Path):
    output = tmp_path / "parity"

    rc = run_freemhd_parity_suite.main(
        [
            "--output",
            str(output),
            "--freemhd-install-dir",
            str(tmp_path / "missing_freemhd"),
            "--processed-root",
            str(tmp_path / "missing_processed"),
        ]
    )

    assert rc == 0
    payload = json.loads((output / "summary.json").read_text())
    assert payload["status"] == "skipped"
    assert "FreeMHD reference outputs" in payload["reason"]
    assert (output / "summary.md").exists()


def test_observable_max_l2_reads_nested_axis_payloads():
    records = [
        {
            "observables": {
                "velocity": {
                    "y": {"l2_error": 0.01},
                    "z": {"l2_error": 0.02},
                },
                "current": {
                    "y": {"l2_error": 0.03},
                },
            }
        }
    ]

    assert run_freemhd_parity_suite._observable_max_l2(records, axis="y") == 0.03
    assert run_freemhd_parity_suite._observable_max_l2(records, axis="z") == 0.02


def test_freemhd_parity_markdown_includes_observable_gate(tmp_path: Path):
    path = tmp_path / "summary.md"
    run_freemhd_parity_suite._write_markdown(
        path,
        {
            "status": "completed",
            "reason": "",
            "case_dir": "/tmp/reference",
            "sample_output": "/tmp/sample",
            "parity_output": "/tmp/parity",
            "parity_report": {
                "metrics": {"reference_sample_y_l2_error": 0.02},
                "observable_gate": {
                    "research_grade_validation_pass": False,
                    "observable_offender_count": 3,
                    "missing_observable_count": 0,
                    "low_signal_count": 1,
                },
            },
        },
    )

    text = path.read_text()
    assert "## Observable Gate" in text
    assert "Research-grade pass: `False`" in text
    assert "Offenders: `3`" in text
