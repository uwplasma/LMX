from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ValidationCaseSummary:
    case: str
    residual: float
    u_max: float
    l2_error: float | None = None
    linf_error: float | None = None
    y_l2_error: float | None = None
    z_l2_error: float | None = None
    slice_y_l2_error: float | None = None
    slice_z_l2_error: float | None = None


@dataclass(frozen=True)
class BenchmarkSummary:
    case: str
    cold_seconds: float
    warm_seconds: float
    mean_seconds: float
    repeats: float


@dataclass(frozen=True)
class ParitySummary:
    status: str
    reason: str
    case_dir: str
    sample_output: str
    parity_output: str
    freemhd_sample_y_l2_error: float | None = None
    freemhd_sample_z_l2_error: float | None = None
    u_max_abs_diff: float | None = None


@dataclass(frozen=True)
class SweepSummary:
    label: str
    case: str
    parameter: str
    first_value: float
    last_value: float
    first_y_l2_error: float | None
    last_y_l2_error: float | None
    first_z_l2_error: float | None
    last_z_l2_error: float | None


def _load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def _coerce_case_payload(case_name: str, payload: dict[str, Any]) -> ValidationCaseSummary:
    return ValidationCaseSummary(
        case=case_name,
        residual=float(payload["residual"]),
        u_max=float(payload["u_max"]),
        l2_error=float(payload["l2_error"]) if "l2_error" in payload else None,
        linf_error=float(payload["linf_error"]) if "linf_error" in payload else None,
        y_l2_error=float(payload["y_l2_error"]) if "y_l2_error" in payload else None,
        z_l2_error=float(payload["z_l2_error"]) if "z_l2_error" in payload else None,
        slice_y_l2_error=float(payload["slice_y_l2_error"]) if "slice_y_l2_error" in payload else None,
        slice_z_l2_error=float(payload["slice_z_l2_error"]) if "slice_z_l2_error" in payload else None,
    )


def summarize_validation_summary(summary_path: str | Path) -> list[ValidationCaseSummary]:
    payload = _load_json(summary_path)
    return [_coerce_case_payload(case_name, case_payload) for case_name, case_payload in sorted(payload.items())]


def summarize_benchmark_report(report_path: str | Path) -> BenchmarkSummary:
    payload = _load_json(report_path)
    return BenchmarkSummary(
        case=str(payload["case"]),
        cold_seconds=float(payload["cold_seconds"]),
        warm_seconds=float(payload["warm_seconds"]),
        mean_seconds=float(payload["mean_seconds"]),
        repeats=float(payload["repeats"]),
    )


def summarize_parity_report(report_path: str | Path) -> ParitySummary:
    payload = _load_json(report_path)
    parity_report = payload.get("parity_report", {})
    metrics = parity_report.get("metrics", {})
    return ParitySummary(
        status=str(payload.get("status", "")),
        reason=str(payload.get("reason", "")),
        case_dir=str(payload.get("case_dir", "")),
        sample_output=str(payload.get("sample_output", "")),
        parity_output=str(payload.get("parity_output", "")),
        freemhd_sample_y_l2_error=(
            None if "freemhd_sample_y_l2_error" not in metrics else float(metrics["freemhd_sample_y_l2_error"])
        ),
        freemhd_sample_z_l2_error=(
            None if "freemhd_sample_z_l2_error" not in metrics else float(metrics["freemhd_sample_z_l2_error"])
        ),
        u_max_abs_diff=None if "u_max_abs_diff" not in metrics else float(metrics["u_max_abs_diff"]),
    )


def summarize_sweep_report(report_path: str | Path, *, label: str) -> SweepSummary:
    payload = _load_json(report_path)
    levels = payload.get("levels", [])
    first = levels[0] if levels else {}
    last = levels[-1] if levels else {}
    return SweepSummary(
        label=label,
        case=str(payload.get("case", "")),
        parameter=str(payload.get("parameter", "")),
        first_value=float(first.get("parameter_value", 0.0)) if first else 0.0,
        last_value=float(last.get("parameter_value", 0.0)) if last else 0.0,
        first_y_l2_error=None if "y_l2_error" not in first else float(first["y_l2_error"]),
        last_y_l2_error=None if "y_l2_error" not in last else float(last["y_l2_error"]),
        first_z_l2_error=None if "z_l2_error" not in first else float(first["z_l2_error"]),
        last_z_l2_error=None if "z_l2_error" not in last else float(last["z_l2_error"]),
    )


def render_markdown(
    validation: list[ValidationCaseSummary],
    benchmark: BenchmarkSummary | None = None,
    parity: ParitySummary | None = None,
    time_convergence: SweepSummary | None = None,
    control_sweep: SweepSummary | None = None,
) -> str:
    lines = ["# LMX CI Summary", ""]
    if validation:
        lines.extend(
            [
                "## Validation",
                "",
                "| Case | Residual | U max | L2 error | Y L2 | Z L2 | Slice Y L2 | Slice Z L2 |",
                "| --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for item in validation:
            lines.append(
                "| "
                + " | ".join(
                    [
                        item.case,
                        f"{item.residual:.6g}",
                        f"{item.u_max:.6g}",
                        "-" if item.l2_error is None else f"{item.l2_error:.6g}",
                        "-" if item.y_l2_error is None else f"{item.y_l2_error:.6g}",
                        "-" if item.z_l2_error is None else f"{item.z_l2_error:.6g}",
                        "-" if item.slice_y_l2_error is None else f"{item.slice_y_l2_error:.6g}",
                        "-" if item.slice_z_l2_error is None else f"{item.slice_z_l2_error:.6g}",
                    ]
                )
                + " |"
            )
        lines.append("")
    if benchmark is not None:
        lines.extend(
            [
                "## Benchmark",
                "",
                f"- Case: `{benchmark.case}`",
                f"- Cold seconds: `{benchmark.cold_seconds:.6g}`",
                f"- Warm seconds: `{benchmark.warm_seconds:.6g}`",
                f"- Mean seconds: `{benchmark.mean_seconds:.6g}`",
                f"- Repeats: `{benchmark.repeats:.0f}`",
                "",
            ]
        )
    if parity is not None:
        lines.extend(
            [
                "## FreeMHD Parity",
                "",
                f"- Status: `{parity.status}`",
                f"- Reason: `{'-' if not parity.reason else parity.reason}`",
                f"- Case dir: `{'-' if not parity.case_dir else parity.case_dir}`",
                f"- U max abs diff: `{'-' if parity.u_max_abs_diff is None else f'{parity.u_max_abs_diff:.6g}'}`",
                f"- Sample Y L2: `{'-' if parity.freemhd_sample_y_l2_error is None else f'{parity.freemhd_sample_y_l2_error:.6g}'}`",
                f"- Sample Z L2: `{'-' if parity.freemhd_sample_z_l2_error is None else f'{parity.freemhd_sample_z_l2_error:.6g}'}`",
                "",
            ]
        )
    for sweep in [time_convergence, control_sweep]:
        if sweep is None:
            continue
        lines.extend(
            [
                f"## {sweep.label}",
                "",
                f"- Case: `{sweep.case}`",
                f"- Parameter: `{sweep.parameter}`",
                f"- First value: `{sweep.first_value:.6g}`",
                f"- Last value: `{sweep.last_value:.6g}`",
                f"- First Y L2: `{'-' if sweep.first_y_l2_error is None else f'{sweep.first_y_l2_error:.6g}'}`",
                f"- Last Y L2: `{'-' if sweep.last_y_l2_error is None else f'{sweep.last_y_l2_error:.6g}'}`",
                f"- First Z L2: `{'-' if sweep.first_z_l2_error is None else f'{sweep.first_z_l2_error:.6g}'}`",
                f"- Last Z L2: `{'-' if sweep.last_z_l2_error is None else f'{sweep.last_z_l2_error:.6g}'}`",
                "",
            ]
        )
    return "\n".join(lines)


def build_summary(
    validation_summary_path: str | Path | None = None,
    benchmark_report_path: str | Path | None = None,
    parity_summary_path: str | Path | None = None,
    time_convergence_summary_path: str | Path | None = None,
    control_sweep_summary_path: str | Path | None = None,
) -> dict[str, Any]:
    validation = summarize_validation_summary(validation_summary_path) if validation_summary_path is not None else []
    benchmark = summarize_benchmark_report(benchmark_report_path) if benchmark_report_path is not None else None
    parity = summarize_parity_report(parity_summary_path) if parity_summary_path is not None else None
    time_convergence = (
        summarize_sweep_report(time_convergence_summary_path, label="Time Convergence")
        if time_convergence_summary_path is not None
        else None
    )
    control_sweep = (
        summarize_sweep_report(control_sweep_summary_path, label="Control Sweep")
        if control_sweep_summary_path is not None
        else None
    )
    payload: dict[str, Any] = {
        "validation": [item.__dict__ for item in validation],
        "benchmark": None if benchmark is None else benchmark.__dict__,
        "parity": None if parity is None else parity.__dict__,
        "time_convergence": None if time_convergence is None else time_convergence.__dict__,
        "control_sweep": None if control_sweep is None else control_sweep.__dict__,
        "markdown": render_markdown(validation, benchmark, parity, time_convergence, control_sweep),
    }
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize LMX CI validation and benchmark artifacts.")
    parser.add_argument("--validation-summary", type=Path, default=None)
    parser.add_argument("--benchmark-report", type=Path, default=None)
    parser.add_argument("--parity-summary", type=Path, default=None)
    parser.add_argument("--time-convergence-summary", type=Path, default=None)
    parser.add_argument("--control-sweep-summary", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, default=None)
    args = parser.parse_args(argv)

    summary = build_summary(
        args.validation_summary,
        args.benchmark_report,
        args.parity_summary,
        args.time_convergence_summary,
        args.control_sweep_summary,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2))
    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(summary["markdown"])
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
