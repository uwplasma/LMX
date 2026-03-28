#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lmx.validation import latest_field_minmax_record
from scripts import run_freemhd_case as freemhd_case
from scripts import run_freemhd_parity_report as parity_report
from scripts import sample_freemhd_profiles as sample_profiles


def case_needs_run(case_dir: Path, time_name: str) -> bool:
    if latest_field_minmax_record(case_dir) is None:
        return True
    return not (case_dir / time_name).is_dir()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the FreeMHD parity artifact suite when a recovered case directory is available.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/freemhd_parity"))
    parser.add_argument("--case-dir", type=Path, default=None)
    parser.add_argument("--case-kind", type=str, default="shercliff")
    parser.add_argument("--ha", type=float, default=20.0)
    parser.add_argument("--image", type=str, default="microfluidica/openfoam:2206")
    parser.add_argument("--platform", type=str, default="linux/amd64")
    parser.add_argument("--time", type=str, default="0.0001")
    parser.add_argument("--dict-name", type=str, default="lmxCiSampleDict")
    parser.add_argument("--run-case-if-needed", action="store_true")
    parser.add_argument("--run-image", type=str, default="lmx-freemhd-smoke")
    parser.add_argument("--run-cores", type=str, default="8")
    parser.add_argument("--run-end-time", type=str, default="1e-4")
    parser.add_argument("--run-write-interval", type=str, default="1e-4")
    parser.add_argument("--run-delta-t", type=str, default="1e-5")
    args = parser.parse_args(argv)

    case_dir = args.case_dir
    if case_dir is None:
        env_case_dir = os.environ.get("LMX_FREEMHD_CASE_DIR", "").strip()
        case_dir = Path(env_case_dir) if env_case_dir else None

    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "summary.json"

    if case_dir is None or not case_dir.exists():
        payload = {
            "status": "skipped",
            "reason": "freemhd-case-unavailable",
            "case_dir": "" if case_dir is None else str(case_dir),
            "sample_output": "",
            "parity_output": "",
        }
        summary_path.write_text(json.dumps(payload, indent=2))
        print(summary_path.read_text())
        return 0

    sample_output = args.output / "sample_profiles.json"
    parity_output = args.output / "parity_report.json"
    run_output = args.output / "run_case.json"

    if args.run_case_if_needed and case_needs_run(case_dir, args.time):
        run_exit = freemhd_case.main(
            [
                "--image",
                args.run_image,
                "--case-dir",
                str(case_dir),
                "--platform",
                args.platform,
                "--cores",
                args.run_cores,
                "--end-time",
                args.run_end_time,
                "--write-interval",
                args.run_write_interval,
                "--delta-t",
                args.run_delta_t,
                "--output",
                str(run_output),
            ]
        )
        if run_exit != 0:
            payload = {
                "status": "failed",
                "reason": "run-failed",
                "case_dir": str(case_dir.resolve()),
                "run_output": str(run_output.resolve()),
                "sample_output": "",
                "parity_output": "",
            }
            summary_path.write_text(json.dumps(payload, indent=2))
            print(summary_path.read_text())
            return run_exit

    sample_exit = sample_profiles.main(
        [
            "--case-dir",
            str(case_dir),
            "--image",
            args.image,
            "--platform",
            args.platform,
            "--time",
            args.time,
            "--dict-name",
            args.dict_name,
            "--output",
            str(sample_output),
        ]
    )
    if sample_exit != 0:
        payload = {
            "status": "failed",
            "reason": "sample-failed",
            "case_dir": str(case_dir.resolve()),
            "run_output": str(run_output.resolve()) if run_output.exists() else "",
            "sample_output": str(sample_output.resolve()),
            "parity_output": "",
        }
        summary_path.write_text(json.dumps(payload, indent=2))
        print(summary_path.read_text())
        return sample_exit

    parity_exit = parity_report.main(
        [
            "--case-kind",
            args.case_kind,
            "--ha",
            str(args.ha),
            "--freemhd-run-dir",
            str(case_dir),
            "--output",
            str(parity_output),
        ]
    )
    payload = {
        "status": "ok" if parity_exit == 0 else "failed",
        "reason": "" if parity_exit == 0 else "parity-report-failed",
        "case_dir": str(case_dir.resolve()),
        "run_output": str(run_output.resolve()) if run_output.exists() else "",
        "sample_output": str(sample_output.resolve()),
        "parity_output": str(parity_output.resolve()),
    }
    if parity_output.exists():
        payload["parity_report"] = json.loads(parity_output.read_text())
    summary_path.write_text(json.dumps(payload, indent=2))
    print(summary_path.read_text())
    return parity_exit


if __name__ == "__main__":
    raise SystemExit(main())
