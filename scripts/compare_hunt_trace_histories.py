#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from bisect import bisect_left
from math import sqrt
from pathlib import Path


def _interpolate(xs: list[float], ys: list[float], x: float) -> float:
    if not xs:
        raise ValueError("Cannot interpolate an empty series")
    if len(xs) != len(ys):
        raise ValueError("xs and ys must have the same length")
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    index = bisect_left(xs, x)
    x0, x1 = xs[index - 1], xs[index]
    y0, y1 = ys[index - 1], ys[index]
    alpha = (x - x0) / (x1 - x0)
    return y0 + alpha * (y1 - y0)


def _normalize(values: list[float]) -> list[float]:
    if not values:
        return []
    baseline = values[0]
    if abs(baseline) < 1e-20:
        return [0.0 for _ in values]
    return [value / baseline for value in values]


def _pressure_final_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[float, str], dict[str, object]] = {}
    for record in records:
        if record.get("kind") != "pressure":
            continue
        time = float(record["time"])
        region = str(record.get("region", ""))
        key = (time, region)
        existing = grouped.get(key)
        corr = int(record.get("corr", -1))
        if existing is None or corr > int(existing.get("corr", -1)):
            grouped[key] = record
    return [grouped[key] for key in sorted(grouped)]


def _epot_records(records: list[dict[str, object]]) -> list[dict[str, object]]:
    epot = [record for record in records if record.get("kind") == "epot"]
    return sorted(epot, key=lambda record: float(record["time"]))


def _build_alignment(
    freemhd_times: list[float],
    freemhd_values: list[float],
    lmx_times: list[float],
    lmx_values: list[float],
) -> dict[str, object]:
    lmx_aligned = [_interpolate(lmx_times, lmx_values, time) for time in freemhd_times]
    freemhd_normalized = _normalize(freemhd_values)
    lmx_normalized = _normalize(lmx_aligned)
    abs_diffs = [abs(a - b) for a, b in zip(freemhd_normalized, lmx_normalized)]
    max_index = max(range(len(abs_diffs)), key=lambda idx: abs_diffs[idx])
    l2_error = sqrt(sum(diff * diff for diff in abs_diffs) / len(abs_diffs))
    samples = [
        {
            "time": time,
            "freemhd_raw": freemhd_value,
            "lmx_raw": lmx_value,
            "freemhd_normalized": freemhd_norm,
            "lmx_normalized": lmx_norm,
            "abs_diff": diff,
        }
        for time, freemhd_value, lmx_value, freemhd_norm, lmx_norm, diff in zip(
            freemhd_times,
            freemhd_values,
            lmx_aligned,
            freemhd_normalized,
            lmx_normalized,
            abs_diffs,
        )
    ]
    return {
        "l2_error": l2_error,
        "max_abs_diff": abs_diffs[max_index],
        "max_abs_diff_time": freemhd_times[max_index],
        "samples": samples,
    }


def compare_trace_histories(freemhd_diag_json: Path, lmx_report_json: Path) -> dict[str, object]:
    freemhd_payload = json.loads(freemhd_diag_json.read_text())
    lmx_payload = json.loads(lmx_report_json.read_text())

    records = freemhd_payload["records"]
    pressure_records = _pressure_final_records(records)
    epot_records = _epot_records(records)
    lmx_trace = lmx_payload["lmx_solver"]["trace"]

    u_times = [float(record["time"]) for record in pressure_records]
    u_values = [float(record["maxU"]) for record in pressure_records]
    j_times = [float(record["time"]) for record in epot_records]
    j_values = [float(record["maxJ"]) for record in epot_records]
    lorentz_times = [float(record["time"]) for record in epot_records]
    lorentz_values = [float(record["maxJxB"]) for record in epot_records]

    lmx_times = [float(value) for value in lmx_trace["time_history"]]
    u_history = [float(value) for value in lmx_trace["u_max_history"]]
    current_history = [float(value) for value in lmx_trace["current_max_history"]]
    lorentz_history = [float(value) for value in lmx_trace["lorentz_max_history"]]

    return {
        "freemhd_diag_json": str(freemhd_diag_json.resolve()),
        "lmx_report_json": str(lmx_report_json.resolve()),
        "u_max": _build_alignment(u_times, u_values, lmx_times, u_history),
        "current_max": _build_alignment(j_times, j_values, lmx_times, current_history),
        "lorentz_max": _build_alignment(lorentz_times, lorentz_values, lmx_times, lorentz_history),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Align normalized FreeMHD and LMX Hunt trace histories on the same time axis.")
    parser.add_argument("--freemhd-diag-json", type=Path, required=True)
    parser.add_argument("--lmx-report-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    payload = compare_trace_histories(args.freemhd_diag_json, args.lmx_report_json)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
