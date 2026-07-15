#!/usr/bin/env python3
"""Analyze a converged three-level Benchmark-A parity ladder.

Raw profile arrays remain external artifacts.  This script writes only compact
orders, extrapolated errors, gates, and source checksums suitable for Git.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


MINIMUM_RELIABLE_ORDER = 0.5
RESULT_FILENAMES = {
    "richardson": "benchmark-a-ha20-richardson.json",
    "continuum": "benchmark-a-ha20-continuum-reference.json",
    "power": "benchmark-a-ha20-power-balance.json",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _records_by_case(level: dict[str, Any]) -> dict[str, dict[str, Any]]:
    records = level.get("records")
    if not isinstance(records, list):
        raise ValueError("Benchmark ladder level must contain a records list")
    by_case = {record.get("case_kind"): record for record in records}
    if set(by_case) != {"shercliff", "hunt"} or len(records) != 2:
        raise ValueError(
            "Benchmark ladder level must contain one Shercliff and one Hunt record"
        )
    return by_case


def _primary_errors(record: dict[str, Any]) -> dict[str, float]:
    return {
        "velocity_y_l2": float(record["observables"]["velocity"]["y"]["l2_error"]),
        "lorentz_y_l2": float(record["observables"]["lorentz"]["y"]["l2_error"]),
        "pressure_gradient_relative": float(
            record["integral_observables"]["pressure_gradient_relative_error"]
        ),
    }


def compact_evidence(
    summary: dict[str, Any], *, source_sha256: str
) -> dict[str, dict[str, Any]]:
    """Return the compact acceptance artifacts for a full Benchmark-A ladder."""

    implementation = summary.get("implementation")
    if not isinstance(implementation, dict) or not all(
        isinstance(implementation.get(key), str) and implementation[key]
        for key in ("runner_sha256", "solver_core_sha256", "lmx_version", "solvax_version")
    ):
        raise ValueError("Benchmark ladder summary is missing its implementation fingerprint")
    ladder = summary.get("ladder")
    if not isinstance(ladder, list) or len(ladder) < 4:
        raise ValueError("Frozen Benchmark-A evidence requires at least four ladder levels")
    confirmation = ladder[-1]
    if confirmation.get("label") != summary.get("best_level_label"):
        raise ValueError("Best level must be the final confirmation level")
    records = _records_by_case(confirmation)

    richardson = copy.deepcopy(summary.get("richardson"))
    if not isinstance(richardson, dict):
        raise ValueError("Benchmark ladder summary is missing Richardson analysis")
    if richardson.get("levels") != [level.get("label") for level in ladder[-3:]]:
        raise ValueError("Richardson analysis must use the final three ladder levels")
    richardson.update(
        confirmation_level=confirmation["label"],
        implementation=copy.deepcopy(implementation),
        confirmation_current_balance={
            case: copy.deepcopy(record["current_balance"])
            for case, record in records.items()
        },
        source_artifact_sha256=source_sha256,
    )

    continuum_cases: dict[str, Any] = {}
    power_cases: dict[str, Any] = {}
    for case, record in records.items():
        continuum = copy.deepcopy(record["continuum_velocity_audit"])
        continuum["reference_file"] = Path(continuum.pop("reference_path")).name
        continuum["benchmark_spec"] = copy.deepcopy(record["benchmark_spec"])
        continuum["settings"] = copy.deepcopy(record["settings"])
        continuum_cases[case] = continuum
        power_cases[case] = {
            "benchmark_spec": copy.deepcopy(record["benchmark_spec"]),
            "settings": copy.deepcopy(record["settings"]),
            "primary_errors": _primary_errors(record),
            "current_balance": copy.deepcopy(record["current_balance"]),
            "power_balance": copy.deepcopy(record["power_balance"]),
        }

    return {
        "richardson": richardson,
        "continuum": {
            "schema_version": 1,
            "description": (
                "Finest-level LMX and processed-FreeMHD velocity errors against the "
                "supplied analytical continuum profiles on one shared physical scale."
            ),
            "implementation": copy.deepcopy(implementation),
            "confirmation_level": confirmation["label"],
            "cases": continuum_cases,
            "source_artifact_sha256": source_sha256,
        },
        "power": {
            "schema_version": 1,
            "implementation": copy.deepcopy(implementation),
            "confirmation_level": confirmation["label"],
            "cases": power_cases,
            "source_artifact_sha256": source_sha256,
        },
    }


def freeze_summary(summary_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    """Write deterministic compact evidence from a raw campaign summary."""

    source = Path(summary_path)
    evidence = compact_evidence(
        json.loads(source.read_text(encoding="utf-8")), source_sha256=_sha256(source)
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {}
    for kind, payload in evidence.items():
        path = destination / RESULT_FILENAMES[kind]
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        paths[kind] = path
    return paths


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(values, dtype=float) ** 2)))


def _representative_h(record: dict[str, Any]) -> float:
    settings = record["settings"]
    return 1.0 / math.sqrt(float(settings["ny"]) * float(settings["nz"]))


def _observed_order(
    h_coarse: float,
    h_medium: float,
    h_fine: float,
    coarse_medium_difference: float,
    medium_fine_difference: float,
) -> float | None:
    if not (h_coarse > h_medium > h_fine > 0.0):
        raise ValueError("Representative spacings must decrease from coarse to fine")
    if coarse_medium_difference <= 1.0e-15 or medium_fine_difference <= 1.0e-15:
        return None
    target = coarse_medium_difference / medium_fine_difference

    def ratio(order: float) -> float:
        numerator = h_coarse**order - h_medium**order
        denominator = h_medium**order - h_fine**order
        return numerator / denominator

    orders = np.linspace(0.02, 12.0, 12001)
    residuals = np.asarray([ratio(float(order)) - target for order in orders])
    crossings = np.flatnonzero(residuals[:-1] * residuals[1:] <= 0.0)
    if crossings.size == 0:
        return None
    lower = float(orders[int(crossings[0])])
    upper = float(orders[int(crossings[0]) + 1])
    for _ in range(60):
        middle = 0.5 * (lower + upper)
        if (ratio(lower) - target) * (ratio(middle) - target) <= 0.0:
            upper = middle
        else:
            lower = middle
    return 0.5 * (lower + upper)


def _richardson_extrapolate(
    medium: np.ndarray | float,
    fine: np.ndarray | float,
    h_medium: float,
    h_fine: float,
    order: float,
) -> np.ndarray:
    medium_array = np.asarray(medium, dtype=float)
    fine_array = np.asarray(fine, dtype=float)
    return (fine_array * h_medium**order - medium_array * h_fine**order) / (
        h_medium**order - h_fine**order
    )


def _case_records(levels: list[dict[str, Any]], case_kind: str) -> list[dict[str, Any]]:
    records = []
    for level in levels:
        matches = [record for record in level["records"] if record["case_kind"] == case_kind]
        if len(matches) != 1:
            raise ValueError(f"Level {level['label']!r} must contain exactly one {case_kind} record")
        records.append(matches[0])
    spec_ids = {record["benchmark_spec"]["id"] for record in records}
    spec_hashes = {record["benchmark_spec"]["sha256"] for record in records}
    if len(spec_ids) != 1 or len(spec_hashes) != 1:
        raise ValueError(f"Benchmark specification changed inside the {case_kind} ladder")
    return records


def _profile_analysis(
    records: list[dict[str, Any]], observable: str, axis: str
) -> dict[str, Any]:
    cuts = [record["observables"][observable][axis] for record in records]
    coordinates = [np.asarray(cut["coordinate"], dtype=float) for cut in cuts]
    if any(not np.array_equal(coordinates[0], coordinate) for coordinate in coordinates[1:]):
        raise ValueError(f"Reference coordinates changed for {observable}/{axis}")
    references = [np.asarray(cut["reference"], dtype=float) for cut in cuts]
    if any(not np.array_equal(references[0], reference) for reference in references[1:]):
        raise ValueError(f"Reference profile changed for {observable}/{axis}")
    simulated = [np.asarray(cut["simulated"], dtype=float) for cut in cuts]
    spacings = [_representative_h(record) for record in records]
    coarse_medium = _rms(simulated[0] - simulated[1])
    medium_fine = _rms(simulated[1] - simulated[2])
    order = _observed_order(*spacings, coarse_medium, medium_fine)
    payload: dict[str, Any] = {
        "coarse_l2": float(cuts[0]["l2_error"]),
        "medium_l2": float(cuts[1]["l2_error"]),
        "fine_l2": float(cuts[2]["l2_error"]),
        "coarse_medium_solution_l2": coarse_medium,
        "medium_fine_solution_l2": medium_fine,
        "observed_order": order,
        "fine_pass": float(cuts[2]["l2_error"]) <= 0.01,
    }
    if order is None:
        payload.update(
            {
                "extrapolated_l2": None,
                "extrapolated_linf": None,
                "order_reliable": False,
                "extrapolated_pass": False,
                "extrapolation_status": "order_unavailable",
            }
        )
        return payload
    extrapolated = _richardson_extrapolate(
        simulated[1], simulated[2], spacings[1], spacings[2], order
    )
    difference = extrapolated - references[0]
    extrapolated_l2 = _rms(difference)
    payload.update(
        {
            "extrapolated_l2": extrapolated_l2,
            "extrapolated_linf": float(np.max(np.abs(difference))),
            "order_reliable": order >= MINIMUM_RELIABLE_ORDER,
            "extrapolated_pass": order >= MINIMUM_RELIABLE_ORDER and extrapolated_l2 <= 0.01,
            "extrapolation_status": (
                "available" if order >= MINIMUM_RELIABLE_ORDER else "order_below_reliability_floor"
            ),
        }
    )
    return payload


def _pressure_analysis(records: list[dict[str, Any]]) -> dict[str, Any]:
    spacings = [_representative_h(record) for record in records]
    values = [float(record["integral_observables"]["applied_pressure_gradient"]) for record in records]
    reference = float(records[0]["integral_observables"]["reference_pressure_gradient"])
    order = _observed_order(
        *spacings,
        abs(values[0] - values[1]),
        abs(values[1] - values[2]),
    )
    extrapolated = None
    extrapolated_error = None
    if order is not None:
        extrapolated = float(_richardson_extrapolate(values[1], values[2], spacings[1], spacings[2], order))
        extrapolated_error = abs(extrapolated - reference) / abs(reference)
    order_reliable = order is not None and order >= MINIMUM_RELIABLE_ORDER
    return {
        "reference": reference,
        "coarse": values[0],
        "medium": values[1],
        "fine": values[2],
        "fine_relative_error": abs(values[2] - reference) / abs(reference),
        "observed_order": order,
        "order_reliable": order_reliable,
        "extrapolated": extrapolated,
        "extrapolated_relative_error": extrapolated_error,
        "fine_pass": abs(values[2] - reference) / abs(reference) <= 0.01,
        "extrapolated_pass": order_reliable
        and extrapolated_error is not None
        and extrapolated_error <= 0.01,
    }


def analyze_ladder(levels: list[dict[str, Any]]) -> dict[str, Any]:
    if len(levels) != 3:
        raise ValueError("Benchmark-A Richardson analysis requires exactly three levels")
    cases: dict[str, Any] = {}
    for case_kind in ("shercliff", "hunt"):
        records = _case_records(levels, case_kind)
        profiles = {
            f"{observable}_{axis}": _profile_analysis(records, observable, axis)
            for observable in ("velocity", "potential", "current", "lorentz")
            for axis in ("y", "z")
        }
        pressure = _pressure_analysis(records)
        primary = [profiles["velocity_y"], profiles["lorentz_y"]]
        cases[case_kind] = {
            "spec_id": records[0]["benchmark_spec"]["id"],
            "spec_sha256": records[0]["benchmark_spec"]["sha256"],
            "representative_h": [_representative_h(record) for record in records],
            "potential_iterations_used": [
                int(record["solver_diagnostics"]["potential_iterations_used"]) for record in records
            ],
            "potential_residual": [
                float(record["solver_diagnostics"]["potential_residual"]) for record in records
            ],
            "profiles": profiles,
            "pressure_gradient": pressure,
            "fine_primary_pass": all(item["fine_pass"] for item in primary) and pressure["fine_pass"],
            "extrapolated_primary_pass": all(item["extrapolated_pass"] for item in primary)
            and pressure["extrapolated_pass"],
        }
    return {
        "schema_version": 1,
        "method": "generalized three-level Richardson extrapolation using h=1/sqrt(ny*nz)",
        "minimum_reliable_order": MINIMUM_RELIABLE_ORDER,
        "levels": [level["label"] for level in levels],
        "cases": cases,
        "research_grade_validation_pass": all(
            case["fine_primary_pass"] and case["extrapolated_primary_pass"] for case in cases.values()
        ),
    }


def _load_level(label: str, path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError(f"Ladder input {path} does not contain a records list")
    return {
        "label": label,
        "records": records,
        "source": {"path": str(path), "sha256": _sha256(path)},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--level",
        action="append",
        metavar="LABEL=SUMMARY.json",
        help="repeat exactly three times from coarse to fine",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--freeze-summary", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"))
    args = parser.parse_args()
    if args.freeze_summary is not None:
        if args.level or args.output:
            parser.error("--freeze-summary cannot be combined with --level or --output")
        for kind, path in freeze_summary(args.freeze_summary, args.output_dir).items():
            print(f"{kind}: {path} sha256={_sha256(path)}")
        return 0
    if args.level is None or len(args.level) != 3 or args.output is None:
        parser.error("analysis requires exactly three --level values and --output")
    levels = []
    for value in args.level:
        label, separator, raw_path = value.partition("=")
        if not separator or not label or not raw_path:
            parser.error(f"invalid --level value {value!r}")
        levels.append(_load_level(label, Path(raw_path)))
    result = analyze_ladder(levels)
    result["sources"] = [level["source"] for level in levels]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["research_grade_validation_pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
