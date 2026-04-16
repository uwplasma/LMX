from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import run_manual_solver_family_validation as manual_validation
from scripts import run_validation_suite as validation_suite


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_csv(rows: list[dict[str, object]], path: Path) -> Path:
    fieldnames: set[str] = set()
    for row in rows:
        fieldnames.update(row.keys())
    ordered = [
        "benchmark",
        "key",
        "case",
        "geometry_kind",
        "solver_kind",
        "ha",
        "resolution",
        "accepted",
        "conservation_pass",
        "physics_pass",
        "validation_pass",
        "charge_balance_residual",
        "interface_current_residual",
        "max_charge_balance_residual",
        "max_wall_current_leakage",
        "net_boundary_current_residual",
        "volumetric_flow_rate_span",
        "field_mean_velocity_correlation",
    ]
    remaining = sorted(name for name in fieldnames if name not in ordered)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[*ordered, *remaining], extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return path


def _render_markdown(
    *,
    benchmark_a: dict[str, dict],
    benchmark_b: dict[str, dict],
    a_pass: bool,
    b_pass: bool,
    overall_pass: bool,
    a_thresholds: dict[str, float],
    b_thresholds: dict[str, float],
) -> str:
    lines = [
        "# Full Validation Exercise",
        "",
        "## Gate Summary",
        "",
        f"- Benchmark A pass: `{int(a_pass)}`",
        f"- Benchmark B pass: `{int(b_pass)}`",
        f"- Overall pass: `{int(overall_pass)}`",
        "",
        "## Benchmark A",
        "",
        f"- Hartmann L2 threshold: `{a_thresholds['hartmann_l2_threshold']:.6g}`",
        f"- Hartmann Linf threshold: `{a_thresholds['hartmann_linf_threshold']:.6g}`",
        f"- Charge-balance threshold: `{a_thresholds['max_charge_balance']:.6g}`",
        f"- Interface-current threshold: `{a_thresholds['max_interface_current']:.6g}`",
        "",
        "| Key | Case | Accepted | Charge balance | Interface current | U max |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for key, payload in sorted(benchmark_a.items()):
        lines.append(
            "| "
            + " | ".join(
                [
                    key,
                    str(payload.get("case", key)),
                    str(int(bool(payload.get("accepted", payload.get("conservation_pass", 0.0))))),
                    f"{float(payload.get('charge_balance_residual', 0.0)):.6g}",
                    f"{float(payload.get('interface_current_residual', 0.0)):.6g}",
                    f"{float(payload.get('u_max', 0.0)):.6g}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Benchmark B",
            "",
            f"- Charge-balance threshold: `{b_thresholds['max_charge_balance']:.6g}`",
            f"- Wall-current leakage threshold: `{b_thresholds['max_wall_current_leakage']:.6g}`",
            f"- Boundary-current threshold: `{b_thresholds['max_boundary_current']:.6g}`",
            f"- Throughput-span threshold: `{b_thresholds['max_flow_span']:.6g}`",
            f"- Field/velocity correlation threshold: `{b_thresholds['max_field_velocity_correlation']:.6g}`",
            "",
            "| Key | Geometry | Validation pass | Charge balance | Flow span | Field/velocity correlation |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for key, payload in sorted(benchmark_b.items()):
        lines.append(
            "| "
            + " | ".join(
                [
                    key,
                    str(payload.get("geometry_kind", "")),
                    str(int(bool(payload.get("validation_pass", 0.0)))),
                    f"{float(payload.get('max_charge_balance_residual', 0.0)):.6g}",
                    f"{float(payload.get('volumetric_flow_rate_span', 0.0)):.6g}",
                    f"{float(payload.get('field_mean_velocity_correlation', 0.0)):.6g}",
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the combined Benchmark A/B LMX validation exercise.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/validation/full_validation_exercise"))
    parser.add_argument("--ha-values", type=str, default="10,20")
    parser.add_argument("--resolution", type=int, default=12)
    parser.add_argument("--fringing-resolutions", type=str, default="8,12")
    parser.add_argument("--reference-root", type=Path, default=None)
    parser.add_argument("--x-slice", type=str, default="1m")
    parser.add_argument("--skip-paraview", action="store_true")
    parser.add_argument("--hartmann-l2-threshold", type=float, default=0.05)
    parser.add_argument("--hartmann-linf-threshold", type=float, default=0.1)
    parser.add_argument("--max-charge-balance", type=float, default=8.0e-1)
    parser.add_argument("--max-interface-current", type=float, default=2.5e-1)
    parser.add_argument("--max-fringing-wall-current-leakage", type=float, default=1.0e-1)
    parser.add_argument("--max-fringing-boundary-current", type=float, default=1.0e-5)
    parser.add_argument("--max-fringing-flow-span", type=float, default=5.0e-3)
    parser.add_argument("--max-field-velocity-correlation", type=float, default=-5.0e-1)
    parser.add_argument("--fringing-geometries", type=str, default="rect_duct,layered_duct,pipe_ogrid")
    parser.add_argument("--fringing-nx", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--potential-iterations", type=int, default=48)
    parser.add_argument("--coupling-iterations", type=int, default=6)
    parser.add_argument("--write-plot", action="store_true")
    parser.add_argument("--fail-on-threshold", action="store_true")
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    benchmark_a_root = args.output / "benchmark_a"
    benchmark_b_root = args.output / "benchmark_b"
    summary_json = args.output / "full_validation_summary.json"
    summary_csv = args.output / "full_validation_summary.csv"
    summary_md = args.output / "full_validation_summary.md"

    ha_values = [float(item) for item in args.ha_values.split(",") if item.strip()]
    benchmark_a_payload: dict[str, dict] = {}
    benchmark_b_payload: dict[str, dict] = {}

    for ha in ha_values:
        case_output = benchmark_a_root / f"ha{int(ha)}"
        exit_code = validation_suite.main(
            [
                "--output",
                str(case_output),
                "--ha",
                str(ha),
                "--resolution",
                str(args.resolution),
                "--x-slice",
                args.x_slice,
                "--hartmann-l2-threshold",
                str(args.hartmann_l2_threshold),
                "--hartmann-linf-threshold",
                str(args.hartmann_linf_threshold),
                *(["--skip-paraview"] if args.skip_paraview else []),
                *([] if args.reference_root is None else ["--reference-root", str(args.reference_root)]),
            ]
        )
        if exit_code != 0:
            raise SystemExit(exit_code)
        summary = _load_json(case_output / "summary.json")
        for key, value in summary.items():
            benchmark_a_payload[f"{key}_ha{int(ha)}"] = value

    manual_exit_code = manual_validation.main(
        [
            "--output",
            str(benchmark_b_root / "solver_family_summary.json"),
            "--ha-values",
            args.ha_values,
            "--resolutions",
            args.fringing_resolutions,
            "--include-fringing",
            "--fringing-geometries",
            args.fringing_geometries,
            "--fringing-nx",
            str(args.fringing_nx),
            "--max-steps",
            str(args.max_steps),
            "--potential-iterations",
            str(args.potential_iterations),
            "--coupling-iterations",
            str(args.coupling_iterations),
            "--max-charge-balance",
            str(args.max_charge_balance),
            "--max-interface-current",
            str(args.max_interface_current),
            "--max-fringing-wall-current-leakage",
            str(args.max_fringing_wall_current_leakage),
            "--max-fringing-boundary-current",
            str(args.max_fringing_boundary_current),
            "--max-fringing-flow-span",
            str(args.max_fringing_flow_span),
            "--max-field-velocity-correlation",
            str(args.max_field_velocity_correlation),
            "--write-csv",
            *(["--write-plot"] if args.write_plot else []),
            *(["--fail-on-threshold"] if args.fail_on_threshold else []),
        ]
    )
    if manual_exit_code not in (0, 1):
        raise SystemExit(manual_exit_code)
    benchmark_b_payload = _load_json(benchmark_b_root / "solver_family_summary.json")

    a_pass = all(
        (
            bool(payload.get("accepted", 1.0))
            and float(payload.get("charge_balance_residual", 0.0)) <= args.max_charge_balance
            and float(payload.get("interface_current_residual", 0.0)) <= args.max_interface_current
        )
        for payload in benchmark_a_payload.values()
    )
    benchmark_b_rows = {
        key: payload
        for key, payload in benchmark_b_payload.items()
        if str(payload.get("solver_kind", "")) == "extruded_inductionless"
    }
    b_pass = all(bool(payload.get("validation_pass", 0.0)) for payload in benchmark_b_rows.values())
    overall_pass = a_pass and b_pass

    rows: list[dict[str, object]] = []
    for key, payload in benchmark_a_payload.items():
        rows.append({"benchmark": "A", "key": key, **payload})
    for key, payload in benchmark_b_rows.items():
        rows.append({"benchmark": "B", "key": key, **payload})
    _write_csv(rows, summary_csv)

    markdown = _render_markdown(
        benchmark_a=benchmark_a_payload,
        benchmark_b=benchmark_b_rows,
        a_pass=a_pass,
        b_pass=b_pass,
        overall_pass=overall_pass,
        a_thresholds={
            "hartmann_l2_threshold": args.hartmann_l2_threshold,
            "hartmann_linf_threshold": args.hartmann_linf_threshold,
            "max_charge_balance": args.max_charge_balance,
            "max_interface_current": args.max_interface_current,
        },
        b_thresholds={
            "max_charge_balance": args.max_charge_balance,
            "max_wall_current_leakage": args.max_fringing_wall_current_leakage,
            "max_boundary_current": args.max_fringing_boundary_current,
            "max_flow_span": args.max_fringing_flow_span,
            "max_field_velocity_correlation": args.max_field_velocity_correlation,
        },
    )
    summary_payload = {
        "benchmark_a": benchmark_a_payload,
        "benchmark_b": benchmark_b_rows,
        "gates": {
            "benchmark_a_pass": int(a_pass),
            "benchmark_b_pass": int(b_pass),
            "overall_pass": int(overall_pass),
        },
        "artifacts": {
            "benchmark_a_root": str(benchmark_a_root),
            "benchmark_b_summary": str(benchmark_b_root / "solver_family_summary.json"),
            "summary_csv": str(summary_csv),
            "summary_md": str(summary_md),
        },
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2))
    summary_md.write_text(markdown)
    print(summary_json.read_text())
    if args.fail_on_threshold and not overall_pass:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
