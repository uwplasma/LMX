#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


DEFAULT_FREEMHD_INSTALL_DIR = Path("/Users/rogerio/local/tests/freemhd_install")
DEFAULT_PROCESSED_ROOT = Path("/Users/rogerio/local/tests/freemhd_test_cases/FreeMHDPaperAllFigures/ClosedChannel")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def _write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# FreeMHD Parity Suite",
        "",
        f"- Status: `{payload['status']}`",
        f"- Reason: `{payload['reason']}`",
        f"- Case directory: `{payload['case_dir'] or '-'}`",
        f"- Sample output: `{payload['sample_output'] or '-'}`",
        f"- Parity output: `{payload['parity_output'] or '-'}`",
    ]
    metrics = payload.get("parity_report", {}).get("metrics", {})
    observable_gate = payload.get("parity_report", {}).get("observable_gate", {})
    if metrics:
        lines.extend(
            [
                "",
                "## Metrics",
                "",
                f"- Max velocity-profile L2: `{metrics.get('reference_sample_y_l2_error', '-')}`",
                f"- Max secondary-profile L2: `{metrics.get('reference_sample_z_l2_error', '-')}`",
                f"- Max U abs diff: `{metrics.get('u_max_abs_diff', '-')}`",
            ]
        )
    if observable_gate:
        lines.extend(
            [
                "",
                "## Observable Gate",
                "",
                f"- Research-grade pass: `{observable_gate.get('research_grade_validation_pass', '-')}`",
                f"- Offenders: `{observable_gate.get('observable_offender_count', '-')}`",
                f"- Missing observables: `{observable_gate.get('missing_observable_count', '-')}`",
                f"- Low-signal cuts: `{observable_gate.get('low_signal_count', '-')}`",
            ]
        )
    path.write_text("\n".join(lines) + "\n")


def _skip_payload(output: Path, reason: str) -> dict[str, Any]:
    return {
        "status": "skipped",
        "reason": reason,
        "case_dir": "",
        "sample_output": "",
        "parity_output": "",
        "parity_report": {"metrics": {}},
        "output": str(output),
    }


def _max_record_metric(records: list[dict[str, Any]], key: str) -> float | None:
    values = [float(record[key]) for record in records if key in record]
    return max(values) if values else None


def _observable_max_l2(records: list[dict[str, Any]], *, axis: str) -> float | None:
    values: list[float] = []
    for record in records:
        observables = record.get("observables", {})
        if not isinstance(observables, dict):
            continue
        for observable in observables.values():
            if not isinstance(observable, dict):
                continue
            cut = observable.get(axis)
            if isinstance(cut, dict) and "l2_error" in cut:
                values.append(float(cut["l2_error"]))
    return max(values) if values else None


def run_suite(
    *,
    output: Path,
    freemhd_install_dir: Path,
    processed_root: Path,
) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    reference_output_root = freemhd_install_dir / "freemhd_output"
    has_transient_reference = all((reference_output_root / name).exists() for name in ("shercliff", "hunt"))
    has_processed_reference = processed_root.exists()
    if not has_transient_reference and not has_processed_reference:
        return _skip_payload(
            output,
            "FreeMHD reference outputs are not available on this runner; set "
            "LMX_FREEMHD_INSTALL_DIR or LMX_FREEMHD_PROCESSED_ROOT to enable this gate.",
        )

    summary: dict[str, Any] = {
        "status": "completed",
        "reason": "",
        "case_dir": str(freemhd_install_dir) if has_transient_reference else str(processed_root),
        "sample_output": "",
        "parity_output": "",
        "parity_report": {"metrics": {}},
        "runs": {},
    }

    y_errors: list[float] = []
    z_errors: list[float] = []
    u_diffs: list[float] = []
    observable_gate: dict[str, Any] | None = None

    if has_transient_reference:
        from examples import freemhd_closed_channel_parity as transient

        transient.OUTPUT_DIR = output / "closed_channel_parity"
        transient.FREEMHD_INSTALL_DIR = freemhd_install_dir
        transient_summary = transient.run_freemhd_closed_channel_parity()
        summary["runs"]["closed_channel_parity"] = transient_summary
        summary["sample_output"] = str(transient.OUTPUT_DIR)
        summary["parity_output"] = str(transient.OUTPUT_DIR / "freemhd_closed_channel_parity_summary.json")
        records = list(transient_summary.get("records", []))
        for key, target in (("y_l2_error", y_errors), ("z_l2_error", z_errors), ("u_max_abs_diff", u_diffs)):
            value = _max_record_metric(records, key)
            if value is not None:
                target.append(value)

    if has_processed_reference:
        from examples import freemhd_closed_channel_observable_parity as observable

        observable.OUTPUT_DIR = output / "closed_channel_observable_parity"
        observable.REFERENCE_ROOT = processed_root
        observable_summary = observable.run_freemhd_closed_channel_observable_parity()
        summary["runs"]["closed_channel_observable_parity"] = observable_summary
        gate = observable_summary.get("observable_gate")
        if isinstance(gate, dict):
            observable_gate = gate
        summary["parity_output"] = str(observable.OUTPUT_DIR / "freemhd_closed_channel_observable_parity_summary.json")
        records = list(observable_summary.get("records", []))
        y_value = _observable_max_l2(records, axis="y")
        z_value = _observable_max_l2(records, axis="z")
        if y_value is not None:
            y_errors.append(y_value)
        if z_value is not None:
            z_errors.append(z_value)

    summary["parity_report"]["metrics"] = {
        "reference_sample_y_l2_error": max(y_errors) if y_errors else None,
        "reference_sample_z_l2_error": max(z_errors) if z_errors else None,
        "u_max_abs_diff": max(u_diffs) if u_diffs else None,
    }
    if observable_gate is not None:
        summary["parity_report"]["observable_gate"] = observable_gate
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run available FreeMHD parity artifact checks.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--freemhd-install-dir",
        type=Path,
        default=Path(os.environ.get("LMX_FREEMHD_INSTALL_DIR", DEFAULT_FREEMHD_INSTALL_DIR)),
    )
    parser.add_argument(
        "--processed-root",
        type=Path,
        default=Path(os.environ.get("LMX_FREEMHD_PROCESSED_ROOT", DEFAULT_PROCESSED_ROOT)),
    )
    args = parser.parse_args(argv)

    summary = run_suite(
        output=args.output,
        freemhd_install_dir=args.freemhd_install_dir,
        processed_root=args.processed_root,
    )
    _write_json(args.output / "summary.json", summary)
    _write_markdown(args.output / "summary.md", summary)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
