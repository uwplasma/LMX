#!/usr/bin/env python3
"""Freeze SOLVAX-PCG Benchmark-A and CPU-equivalence evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


EXPECTED_ROWS = {
    (case, ha) for case in ("hunt", "shercliff") for ha in (500, 5000, 10000, 15000)
}
HA20_FILES = {
    "continuum": "benchmark-a-ha20-continuum-reference.json",
    "power": "benchmark-a-ha20-power-balance.json",
    "richardson": "benchmark-a-ha20-richardson.json",
}


def _read(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _normalized_gate(values: dict[str, Any]) -> bool:
    target = float(values["acceptance_target"])
    metrics = [
        float(value)
        for key, value in values.items()
        if key != "acceptance_target"
        and (key.endswith("_normalized") or key.endswith("_relative_error"))
    ]
    return bool(metrics) and max(metrics) <= target


def _compact_table_record(record: dict[str, Any]) -> dict[str, Any]:
    finest = record["levels"][-1]
    return {
        "analytical_flow_rate": record["analytical_flow_rate"],
        "case_kind": record["case_kind"],
        "finest_level": {
            "analytical_relative_error": finest["analytical_relative_error"],
            "balances": finest["balances"],
            "mesh": finest["mesh"],
            "q_tilde": finest["q_tilde"],
            "solver": {
                key: finest["solver"][key]
                for key in (
                    "linear_iterations_used",
                    "linear_residual",
                    "potential_iterations_used",
                    "potential_residual",
                    "residual",
                )
            },
        },
        "finest_level_pass": record["finest_level_pass"],
        "hartmann_number": record["hartmann_number"],
        "hartmann_wall_conductance": record["hartmann_wall_conductance"],
        "published_numerical_flow_rate": record["published_numerical_flow_rate"],
        "refinement": record["refinement"],
    }


def validate_ha20_evidence(
    results_dir: Path, campaign_implementation: dict[str, Any]
) -> dict[str, Any]:
    """Validate compact Ha=20 SOLVAX/FreeMHD evidence and summarize it."""

    paths = {kind: results_dir / name for kind, name in HA20_FILES.items()}
    payloads = {kind: _read(path) for kind, path in paths.items()}
    implementations = {
        json.dumps(payload.get("implementation"), sort_keys=True)
        for payload in payloads.values()
    }
    source_hashes = {
        payload.get("source_artifact_sha256") for payload in payloads.values()
    }
    if len(implementations) != 1 or len(source_hashes) != 1 or None in source_hashes:
        raise ValueError("Ha=20 evidence does not share one source and implementation")
    implementation = json.loads(next(iter(implementations)))
    for key in ("solver_core_sha256", "lmx_version", "solvax_version"):
        if implementation.get(key) != campaign_implementation.get(key):
            raise ValueError(f"Ha=20/Table-I implementation differs: {key}")

    power = payloads["power"]
    continuum = payloads["continuum"]
    richardson = payloads["richardson"]
    confirmations = {
        payload.get("confirmation_level") for payload in payloads.values()
    }
    if confirmations != {"confirmation_85x63"}:
        raise ValueError("Ha=20 evidence must use confirmation_85x63")

    cases: dict[str, Any] = {}
    for case in ("shercliff", "hunt"):
        power_case = power["cases"][case]
        primary = {
            key: float(value) for key, value in power_case["primary_errors"].items()
        }
        axes = continuum["cases"][case]["axes"]
        continuum_pass = all(
            float(axis["lmx_raw_analytical"]["l2_error"])
            <= float(axis["processed_freemhd_raw_analytical"]["l2_error"])
            for axis in axes.values()
        )
        case_pass = (
            max(primary.values()) <= 0.01
            and _normalized_gate(power_case["current_balance"])
            and _normalized_gate(power_case["power_balance"])
            and continuum_pass
            and bool(richardson["cases"][case]["fine_primary_pass"])
        )
        cases[case] = {
            "pass": case_pass,
            "primary_errors": primary,
            "current_balance": power_case["current_balance"],
            "power_balance": power_case["power_balance"],
            "continuum_pass": continuum_pass,
            "fine_primary_pass": bool(
                richardson["cases"][case]["fine_primary_pass"]
            ),
            "extrapolated_primary_pass": bool(
                richardson["cases"][case]["extrapolated_primary_pass"]
            ),
        }
    return {
        "pass": all(case["pass"] for case in cases.values()),
        "confirmation_level": "confirmation_85x63",
        "implementation": implementation,
        "source_artifact_sha256": next(iter(source_hashes)),
        "sources": {
            path.name: _sha256(path) for path in paths.values()
        },
        "cases": cases,
    }


def build_acceptance(
    campaign_path: Path,
    confirmation_path: Path,
    cpu_comparison_path: Path,
    gpu_comparison_path: Path | None = None,
    ha20_results_dir: Path | None = None,
) -> dict[str, Any]:
    campaign = _read(campaign_path)
    confirmation = _read(confirmation_path)
    cpu = _read(cpu_comparison_path)
    if campaign.get("implementation") != confirmation.get("implementation"):
        raise ValueError("SOLVAX campaign implementation fingerprints differ")
    if campaign.get("controls") != confirmation.get("controls"):
        raise ValueError("SOLVAX campaign controls differ")
    implementation = campaign["implementation"]
    if implementation.get("solvax_version") != "0.5.1":
        raise ValueError("SOLVAX Benchmark A must use released version 0.5.1")
    if campaign["controls"].get("linear_solver") != "solvax_pcg":
        raise ValueError("SOLVAX Benchmark A must select linear_solver=solvax_pcg")
    records = {
        (str(record["case_kind"]), int(record["hartmann_number"])): record
        for record in campaign.get("records", [])
    }
    for record in confirmation.get("records", []):
        records[(str(record["case_kind"]), int(record["hartmann_number"]))] = record
    if set(records) != EXPECTED_ROWS:
        raise ValueError(
            f"SOLVAX Benchmark A rows differ: {sorted(set(records) ^ EXPECTED_ROWS)}"
        )
    ordered = [records[key] for key in sorted(records)]
    compact_records = [_compact_table_record(record) for record in ordered]
    table_i_pass = all(bool(record.get("finest_level_pass")) for record in ordered)
    cpu_acceptance = cpu.get("acceptance", {})
    cpu_pass = bool(
        cpu_acceptance.get(
            "backend_promotion_pass", cpu_acceptance.get("cpu_promotion_pass")
        )
    )
    if cpu.get("environment", {}).get("backend") != "cpu":
        raise ValueError("CPU comparison must record environment.backend=cpu")
    if cpu.get("environment", {}).get("dtype") != "float64":
        raise ValueError("CPU comparison must use float64")
    sources = [
        {"path": str(campaign_path), "sha256": _sha256(campaign_path)},
        {"path": str(confirmation_path), "sha256": _sha256(confirmation_path)},
        {"path": str(cpu_comparison_path), "sha256": _sha256(cpu_comparison_path)},
    ]
    gpu: dict[str, Any]
    gpu_pass = False
    if gpu_comparison_path is None:
        gpu = {
            "pass": False,
            "status": "not_run",
            "reason": "No JAX GPU device is available on the evidence host.",
        }
    else:
        gpu = _read(gpu_comparison_path)
        if gpu.get("environment", {}).get("backend") != "gpu":
            raise ValueError("GPU comparison must record environment.backend=gpu")
        if gpu.get("environment", {}).get("dtype") != "float64":
            raise ValueError("GPU comparison must use float64")
        if gpu.get("problem") != cpu.get("problem"):
            raise ValueError("CPU/GPU comparison problem definitions differ")
        for key in ("benchmark_sha256", "linear_sha256", "solvax_version"):
            if gpu.get("implementation", {}).get(key) != cpu.get(
                "implementation", {}
            ).get(key):
                raise ValueError(f"CPU/GPU comparison implementation differs: {key}")
        gpu_pass = bool(
            gpu.get("acceptance", {}).get(
                "backend_promotion_pass",
                gpu.get("acceptance", {}).get("gpu_promotion_pass"),
            )
        )
        gpu = {**gpu, "pass": gpu_pass, "status": "accepted" if gpu_pass else "failed"}
        sources.append(
            {
                "path": str(gpu_comparison_path),
                "sha256": _sha256(gpu_comparison_path),
            }
        )
    if ha20_results_dir is None:
        ha20 = {"pass": False, "status": "not_run"}
    else:
        ha20 = validate_ha20_evidence(ha20_results_dir, implementation)
        ha20["status"] = "accepted" if ha20["pass"] else "failed"
        sources.extend(
            {
                "path": str(ha20_results_dir / filename),
                "sha256": ha20["sources"][filename],
            }
            for filename in HA20_FILES.values()
        )
    m3_pass = table_i_pass and cpu_pass and gpu_pass and bool(ha20["pass"])
    blockers = []
    if not gpu_pass:
        blockers.append(
            "GPU forward/gradient/performance equivalence is not yet accepted."
        )
    if not ha20["pass"]:
        blockers.append("Ha=20 SOLVAX/FreeMHD acceptance is not yet accepted.")
    return {
        "schema_version": 1,
        "benchmark": "LMX released-SOLVAX PCG promotion acceptance",
        "implementation": implementation,
        "controls": campaign["controls"],
        "sources": sources,
        "literature_table_i": {
            "pass": table_i_pass,
            "row_count": len(compact_records),
            "records": compact_records,
        },
        "cpu_equivalence": cpu,
        "cpu_acceptance_pass": table_i_pass and cpu_pass,
        "gpu_equivalence": gpu,
        "ha20_freemhd": ha20,
        "m3_promotion_pass": m3_pass,
        "promotion_blockers": blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--campaign",
        type=Path,
        default=Path("artifacts/samper/table-i-solvax-pcg.json"),
    )
    parser.add_argument(
        "--confirmation",
        type=Path,
        default=Path("artifacts/samper/table-i-solvax-pcg-shercliff-ha15000.json"),
    )
    parser.add_argument(
        "--cpu-comparison",
        type=Path,
        default=Path("benchmarks/results/solvax-pcg-equivalence-cpu.json"),
    )
    parser.add_argument("--gpu-comparison", type=Path)
    parser.add_argument("--ha20-results-dir", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/solvax-pcg-acceptance.json"),
    )
    args = parser.parse_args()
    payload = build_acceptance(
        args.campaign,
        args.confirmation,
        args.cpu_comparison,
        args.gpu_comparison,
        args.ha20_results_dir,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(
        f"{args.output} cpu_acceptance_pass={payload['cpu_acceptance_pass']} "
        f"m3_promotion_pass={payload['m3_promotion_pass']}"
    )
    required_pass = (
        payload["m3_promotion_pass"]
        if args.gpu_comparison is not None
        else payload["cpu_acceptance_pass"]
    )
    return 0 if required_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
