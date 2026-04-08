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
    grouped: dict[tuple[float, str, int], dict[str, object]] = {}
    for record in records:
        if record.get("kind") != "epot":
            continue
        time = float(record["time"])
        region = str(record.get("region", ""))
        ocorr = int(record.get("oCorr", -1))
        key = (time, region, ocorr)
        existing = grouped.get(key)
        if existing is None or len(record) > len(existing):
            grouped[key] = record
    return [grouped[key] for key in sorted(grouped)]


def _compact_records(records: list[dict[str, object]], keys: list[str]) -> list[dict[str, object]]:
    compact: list[dict[str, object]] = []
    for record in records:
        compact.append({key: record[key] for key in keys if key in record})
    return compact


def _build_alignment(
    freemhd_times: list[float],
    freemhd_values: list[float],
    lmx_times: list[float],
    lmx_values: list[float],
) -> dict[str, object]:
    if not freemhd_times:
        return {
            "l2_error": None,
            "max_abs_diff": None,
            "max_abs_diff_time": None,
            "samples": [],
        }
    lmx_aligned = [_interpolate(lmx_times, lmx_values, time) for time in freemhd_times]
    raw_relative_errors = [
        abs(freemhd_value - lmx_value) / max(abs(freemhd_value), 1e-20)
        for freemhd_value, lmx_value in zip(freemhd_values, lmx_aligned)
    ]
    freemhd_normalized = _normalize(freemhd_values)
    lmx_normalized = _normalize(lmx_aligned)
    abs_diffs = [abs(a - b) for a, b in zip(freemhd_normalized, lmx_normalized)]
    max_index = max(range(len(abs_diffs)), key=lambda idx: abs_diffs[idx])
    l2_error = sqrt(sum(diff * diff for diff in abs_diffs) / len(abs_diffs))
    raw_max_index = max(range(len(raw_relative_errors)), key=lambda idx: raw_relative_errors[idx])
    samples = [
        {
            "time": time,
            "freemhd_raw": freemhd_value,
            "lmx_raw": lmx_value,
            "raw_relative_error": raw_relative_error,
            "freemhd_normalized": freemhd_norm,
            "lmx_normalized": lmx_norm,
            "abs_diff": diff,
        }
        for time, freemhd_value, lmx_value, raw_relative_error, freemhd_norm, lmx_norm, diff in zip(
            freemhd_times,
            freemhd_values,
            lmx_aligned,
            raw_relative_errors,
            freemhd_normalized,
            lmx_normalized,
            abs_diffs,
        )
    ]
    return {
        "l2_error": l2_error,
        "max_abs_diff": abs_diffs[max_index],
        "max_abs_diff_time": freemhd_times[max_index],
        "mean_raw_relative_error": sum(raw_relative_errors) / len(raw_relative_errors),
        "max_raw_relative_error": raw_relative_errors[raw_max_index],
        "max_raw_relative_error_time": freemhd_times[raw_max_index],
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
    p_times = [float(record["time"]) for record in pressure_records if "maxP" in record]
    p_values = [float(record["maxP"]) for record in pressure_records if "maxP" in record]
    pspan_times = [float(record["time"]) for record in pressure_records if "pSpan" in record]
    pspan_values = [float(record["pSpan"]) for record in pressure_records if "pSpan" in record]
    j_times = [float(record["time"]) for record in epot_records]
    j_values = [float(record["maxJ"]) for record in epot_records]
    jn_times = [float(record["time"]) for record in epot_records if "maxJn" in record]
    jn_values = [float(record["maxJn"]) for record in epot_records if "maxJn" in record]
    jn_density_times = [float(record["time"]) for record in epot_records if "maxJnDensity" in record]
    jn_density_values = [float(record["maxJnDensity"]) for record in epot_records if "maxJnDensity" in record]
    psiub_times = [float(record["time"]) for record in epot_records if "maxPsiub" in record]
    psiub_values = [float(record["maxPsiub"]) for record in epot_records if "maxPsiub" in record]
    psiub_density_times = [float(record["time"]) for record in epot_records if "maxPsiubDensity" in record]
    psiub_density_values = [float(record["maxPsiubDensity"]) for record in epot_records if "maxPsiubDensity" in record]
    lorentz_times = [float(record["time"]) for record in epot_records]
    lorentz_values = [float(record["maxJxB"]) for record in epot_records]

    lmx_times = [float(value) for value in lmx_trace["time_history"]]
    u_history = [float(value) for value in lmx_trace["u_max_history"]]
    mean_velocity_history = [float(value) for value in lmx_trace.get("mean_velocity_history", [])]
    applied_forcing_history = [float(value) for value in lmx_trace.get("applied_forcing_history", [])]
    pressure_proxy_history = [float(value) for value in lmx_trace.get("pressure_proxy_history", [])]
    current_history = [float(value) for value in lmx_trace["current_max_history"]]
    face_current_history = [float(value) for value in lmx_trace.get("face_current_max_history", [])]
    emf_history = [float(value) for value in lmx_trace.get("emf_max_history", [])]
    lorentz_history = [float(value) for value in lmx_trace["lorentz_max_history"]]
    face_lorentz_history = [float(value) for value in lmx_trace.get("face_lorentz_max_history", [])]

    payload = {
        "freemhd_diag_json": str(freemhd_diag_json.resolve()),
        "lmx_report_json": str(lmx_report_json.resolve()),
        "freemhd_pressure_final_records": _compact_records(
            pressure_records,
            [
                "time",
                "corr",
                "maxU",
                "pInitialResidual",
                "pFinalResidual",
                "pIterations",
                "maxP",
                "minP",
                "pSpan",
                "maxPRgh",
                "minPRgh",
                "pRghSpan",
                "maxJxB",
            ],
        ),
        "freemhd_epot_records": _compact_records(
            epot_records,
            [
                "time",
                "oCorr",
                "potEInitialResidual",
                "potEFinalResidual",
                "potEIterations",
                "maxJ",
                "maxJn",
                "maxJnDensity",
                "maxPsiub",
                "maxPsiubDensity",
                "maxJxB",
            ],
        ),
    }
    if u_times:
        payload["u_max"] = _build_alignment(u_times, u_values, lmx_times, u_history)
    if pressure_proxy_history:
        if pspan_times:
            payload["primary_pressure_metric"] = "pSpan"
            payload["pressure_proxy"] = _build_alignment(pspan_times, pspan_values, lmx_times, pressure_proxy_history)
        elif p_times:
            payload["primary_pressure_metric"] = "maxP"
            payload["pressure_proxy"] = _build_alignment(p_times, p_values, lmx_times, pressure_proxy_history)
    if u_times and mean_velocity_history:
        payload["mean_velocity"] = _build_alignment(u_times, u_values, lmx_times, mean_velocity_history)
    if p_times and applied_forcing_history:
        payload["applied_forcing"] = _build_alignment(p_times, p_values, lmx_times, applied_forcing_history)
    if j_times:
        payload["current_max"] = _build_alignment(j_times, j_values, lmx_times, current_history)
    if jn_times and face_current_history:
        payload["face_current_max"] = _build_alignment(jn_times, jn_values, lmx_times, face_current_history)
        payload["primary_current_metric"] = "face_current_max"
        payload["primary_current_max"] = payload["face_current_max"]
    elif j_times:
        payload["primary_current_metric"] = "current_max"
        payload["primary_current_max"] = payload["current_max"]
    if jn_density_times and face_current_history:
        payload["face_current_density_max"] = _build_alignment(
            jn_density_times,
            jn_density_values,
            lmx_times,
            face_current_history,
        )
    if lorentz_times:
        payload["lorentz_max"] = _build_alignment(lorentz_times, lorentz_values, lmx_times, lorentz_history)
    if lorentz_times and face_lorentz_history:
        payload["face_lorentz_max"] = _build_alignment(lorentz_times, lorentz_values, lmx_times, face_lorentz_history)
        payload["primary_lorentz_metric"] = "face_lorentz_max"
        payload["primary_lorentz_max"] = payload["face_lorentz_max"]
    elif lorentz_times:
        payload["primary_lorentz_metric"] = "lorentz_max"
        payload["primary_lorentz_max"] = payload["lorentz_max"]
    if psiub_times and emf_history:
        payload["emf_max"] = _build_alignment(psiub_times, psiub_values, lmx_times, emf_history)
    if psiub_density_times and emf_history:
        payload["emf_density_max"] = _build_alignment(
            psiub_density_times,
            psiub_density_values,
            lmx_times,
            emf_history,
        )
    return payload


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
