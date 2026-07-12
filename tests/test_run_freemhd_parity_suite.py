from pathlib import Path
import json
from types import SimpleNamespace
import sys

import pytest

import examples
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


def test_run_freemhd_parity_suite_uses_available_reference_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    freemhd_root = tmp_path / "freemhd"
    (freemhd_root / "freemhd_output" / "shercliff").mkdir(parents=True)
    (freemhd_root / "freemhd_output" / "hunt").mkdir(parents=True)
    processed_root = tmp_path / "processed"
    processed_root.mkdir()

    transient = SimpleNamespace(
        OUTPUT_DIR=tmp_path / "unset", FREEMHD_INSTALL_DIR=tmp_path / "unset"
    )
    transient.run_freemhd_closed_channel_parity = lambda: {
        "records": [
            {"y_l2_error": 0.01, "z_l2_error": 0.02, "u_max_abs_diff": 0.03},
            {"y_l2_error": 0.04, "z_l2_error": 0.01, "u_max_abs_diff": 0.02},
        ]
    }
    observable = SimpleNamespace(
        OUTPUT_DIR=tmp_path / "unset", REFERENCE_ROOT=tmp_path / "unset"
    )
    observable.run_freemhd_closed_channel_observable_parity = lambda: {
        "records": [
            {
                "observables": {
                    "velocity": {
                        "y": {"l2_error": 0.05},
                        "z": {"l2_error": 0.06},
                    }
                }
            }
        ],
        "observable_gate": {
            "research_grade_validation_pass": False,
            "observable_offender_count": 2,
            "missing_observable_count": 1,
            "low_signal_count": 0,
        },
    }
    monkeypatch.setitem(
        sys.modules, "campaigns.freemhd.freemhd_closed_channel_parity", transient
    )
    monkeypatch.setitem(
        sys.modules, "examples.freemhd_closed_channel_observable_parity", observable
    )
    monkeypatch.setattr(
        examples, "freemhd_closed_channel_parity", transient, raising=False
    )
    monkeypatch.setattr(
        examples, "freemhd_closed_channel_observable_parity", observable, raising=False
    )
    monkeypatch.setattr(
        run_freemhd_parity_suite,
        "audit_freemhd_case_against_spec",
        lambda *_args, case_kind, **_kwargs: {"case_kind": case_kind, "matched": True},
    )

    summary = run_freemhd_parity_suite.run_suite(
        output=tmp_path / "out",
        freemhd_install_dir=freemhd_root,
        processed_root=processed_root,
    )

    assert summary["status"] == "completed"
    assert summary["sample_output"].endswith("closed_channel_parity")
    assert summary["parity_output"].endswith(
        "freemhd_closed_channel_observable_parity_summary.json"
    )
    assert summary["parity_report"]["metrics"]["reference_sample_y_l2_error"] == 0.05
    assert summary["parity_report"]["metrics"]["reference_sample_z_l2_error"] == 0.06
    assert summary["parity_report"]["metrics"]["u_max_abs_diff"] == 0.03
    assert summary["parity_report"]["observable_gate"]["observable_offender_count"] == 2
    assert summary["matched_case_gate"] is True


def test_freemhd_parity_suite_rejects_mismatched_case_before_profile_comparison(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    freemhd_root = tmp_path / "freemhd"
    for case_kind in ("shercliff", "hunt"):
        (freemhd_root / "freemhd_output" / case_kind).mkdir(parents=True)
    monkeypatch.setattr(
        run_freemhd_parity_suite,
        "audit_freemhd_case_against_spec",
        lambda *_args, case_kind, **_kwargs: {
            "case_kind": case_kind,
            "matched": False,
            "failed_check_count": 1,
            "checks": [{"name": "physics.hartmann", "pass": False}],
        },
    )

    summary = run_freemhd_parity_suite.run_suite(
        output=tmp_path / "out",
        freemhd_install_dir=freemhd_root,
        processed_root=tmp_path / "missing",
    )

    assert summary["status"] == "invalid_reference"
    assert summary["matched_case_gate"] is False
    assert "closed_channel_parity" not in summary["runs"]
    assert summary["parity_report"]["metrics"]["reference_sample_y_l2_error"] is None
