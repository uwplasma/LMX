#!/usr/bin/env python3
"""Freeze compact, portable evidence from a Benchmark-A ladder artifact."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
from typing import Any


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
        raise ValueError("Benchmark ladder level must contain one Shercliff and one Hunt record")
    return by_case


def _primary_errors(record: dict[str, Any]) -> dict[str, float]:
    return {
        "velocity_y_l2": float(record["observables"]["velocity"]["y"]["l2_error"]),
        "lorentz_y_l2": float(record["observables"]["lorentz"]["y"]["l2_error"]),
        "pressure_gradient_relative": float(
            record["integral_observables"]["pressure_gradient_relative_error"]
        ),
    }


def compact_evidence(summary: dict[str, Any], *, source_sha256: str) -> dict[str, dict[str, Any]]:
    """Return the three compact acceptance artifacts for a full ladder summary."""

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
    expected_levels = [level.get("label") for level in ladder[-3:]]
    if richardson.get("levels") != expected_levels:
        raise ValueError("Richardson analysis must use the final three ladder levels")
    richardson["confirmation_level"] = confirmation["label"]
    richardson["implementation"] = copy.deepcopy(implementation)
    richardson["confirmation_current_balance"] = {
        case: copy.deepcopy(record["current_balance"]) for case, record in records.items()
    }
    richardson["source_artifact_sha256"] = source_sha256

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

    continuum_payload = {
        "schema_version": 1,
        "description": (
            "Finest-level LMX and processed-FreeMHD velocity errors against the supplied "
            "analytical continuum profiles on one shared physical scale."
        ),
        "implementation": copy.deepcopy(implementation),
        "confirmation_level": confirmation["label"],
        "cases": continuum_cases,
        "source_artifact_sha256": source_sha256,
    }
    power_payload = {
        "schema_version": 1,
        "implementation": copy.deepcopy(implementation),
        "confirmation_level": confirmation["label"],
        "cases": power_cases,
        "source_artifact_sha256": source_sha256,
    }
    return {
        "richardson": richardson,
        "continuum": continuum_payload,
        "power": power_payload,
    }


def freeze_summary(summary_path: str | Path, output_dir: str | Path) -> dict[str, Path]:
    """Read a raw campaign summary and write deterministic compact evidence."""

    source = Path(summary_path)
    summary = json.loads(source.read_text(encoding="utf-8"))
    evidence = compact_evidence(summary, source_sha256=_sha256(source))
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for kind, payload in evidence.items():
        path = destination / RESULT_FILENAMES[kind]
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        paths[kind] = path
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"))
    args = parser.parse_args()
    paths = freeze_summary(args.summary, args.output_dir)
    for kind, path in paths.items():
        print(f"{kind}: {path} sha256={_sha256(path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
