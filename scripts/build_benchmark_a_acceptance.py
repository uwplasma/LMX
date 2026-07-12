#!/usr/bin/env python3
"""Build the compact, fingerprint-consistent Benchmark A acceptance record."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


CASES = ("shercliff", "hunt")
HARTMANN_NUMBERS = (500, 5000, 10000, 15000)
PROFILE_TARGET = 0.01


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read Benchmark A evidence: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Benchmark A evidence must be a JSON object: {path}")
    return payload


def _implementation(payload: dict[str, Any], path: Path) -> dict[str, str]:
    implementation = payload.get("implementation")
    required = ("runner_sha256", "solver_core_sha256", "lmx_version", "solvax_version")
    if not isinstance(implementation, dict) or not all(
        isinstance(implementation.get(key), str) and implementation[key] for key in required
    ):
        raise ValueError(f"Evidence lacks an implementation fingerprint: {path}")
    return {key: implementation[key] for key in required}


def _solver_identity(implementation: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(
        sorted(
            (key, implementation[key])
            for key in ("solver_core_sha256", "lmx_version", "solvax_version")
        )
    )


def _balance_pass(values: dict[str, Any]) -> bool:
    target = float(values["acceptance_target"])
    metrics = [
        float(value)
        for key, value in values.items()
        if key != "acceptance_target"
        and (key.endswith("_normalized") or key.endswith("_relative_error"))
    ]
    return bool(metrics) and max(metrics) <= target


def build_acceptance(results_dir: Path) -> dict[str, Any]:
    """Validate compact component evidence and return one acceptance record."""

    ha20_names = {
        "finite_grid_and_power": "benchmark-a-ha20-power-balance.json",
        "analytical_continuum": "benchmark-a-ha20-continuum-reference.json",
        "richardson_diagnostic": "benchmark-a-ha20-richardson.json",
    }
    ha20_paths = {name: results_dir / filename for name, filename in ha20_names.items()}
    ha20 = {name: _load(path) for name, path in ha20_paths.items()}
    ha20_implementations = {
        tuple(sorted(_implementation(payload, ha20_paths[name]).items()))
        for name, payload in ha20.items()
    }
    source_hashes = {payload.get("source_artifact_sha256") for payload in ha20.values()}
    if len(ha20_implementations) != 1 or len(source_hashes) != 1 or None in source_hashes:
        raise ValueError("Ha=20 compact records do not share one source and implementation")
    ha20_implementation = dict(next(iter(ha20_implementations)))
    solver_identity = _solver_identity(ha20_implementation)

    finite_cases: dict[str, Any] = {}
    conservation_cases: dict[str, Any] = {}
    power = ha20["finite_grid_and_power"]
    richardson = ha20["richardson_diagnostic"]
    continuum = ha20["analytical_continuum"]
    for case in CASES:
        case_power = power["cases"][case]
        primary = {key: float(value) for key, value in case_power["primary_errors"].items()}
        fine_primary_pass = bool(richardson["cases"][case]["fine_primary_pass"])
        finite_pass = fine_primary_pass and max(primary.values()) <= PROFILE_TARGET
        current_pass = _balance_pass(case_power["current_balance"])
        power_pass = _balance_pass(case_power["power_balance"])
        finite_cases[case] = {"pass": finite_pass, "primary_errors": primary}
        conservation_cases[case] = {
            "pass": current_pass and power_pass,
            "current": case_power["current_balance"],
            "power": case_power["power_balance"],
        }

    continuum_cases: dict[str, Any] = {}
    for case in CASES:
        axes: dict[str, Any] = {}
        for axis in ("y", "z"):
            audit = continuum["cases"][case]["axes"][axis]
            lmx_error = float(audit["lmx_raw_analytical"]["l2_error"])
            freemhd_error = float(
                audit["processed_freemhd_raw_analytical"]["l2_error"]
            )
            axes[axis] = {
                "lmx_l2": lmx_error,
                "processed_freemhd_l2": freemhd_error,
                "lmx_not_worse_than_processed_freemhd": lmx_error <= freemhd_error,
            }
        continuum_cases[case] = {
            "pass": all(
                axis["lmx_not_worse_than_processed_freemhd"] for axis in axes.values()
            ),
            "axes": axes,
        }

    table_rows: list[dict[str, Any]] = []
    table_sources: dict[str, str] = {}
    table_implementations: set[tuple[tuple[str, str], ...]] = set()
    table_runner_sha256: str | None = None
    for case in CASES:
        for ha in HARTMANN_NUMBERS:
            filename = f"samper-table-i-{case}-ha{ha}.json"
            path = results_dir / filename
            payload = _load(path)
            table_sources[filename] = _sha256(path)
            row_implementation = _implementation(payload, path)
            table_implementations.add(_solver_identity(row_implementation))
            if table_runner_sha256 is None:
                table_runner_sha256 = row_implementation["runner_sha256"]
            elif table_runner_sha256 != row_implementation["runner_sha256"]:
                raise ValueError("Table I records do not share one campaign runner")
            records = payload.get("records")
            if (
                not payload.get("research_grade_validation_pass")
                or not isinstance(records, list)
                or len(records) != 1
            ):
                raise ValueError(f"Table I evidence is not a passing single-row record: {path}")
            record = records[0]
            if record.get("case_kind") != case or int(record.get("hartmann_number", -1)) != ha:
                raise ValueError(f"Table I evidence row does not match its filename: {path}")
            if not record.get("finest_level_pass"):
                raise ValueError(f"Table I finest-level gate failed: {path}")
            finest = record["levels"][-1]
            table_rows.append(
                {
                    "case": case,
                    "hartmann_number": ha,
                    "pass": True,
                    "finest_mesh": finest["mesh"],
                    "analytical_flow_relative_error": finest["analytical_relative_error"],
                    "finest_mesh_change_relative": record["refinement"][
                        "finest_mesh_change_relative"
                    ],
                    "observed_order": record["refinement"][
                        "observed_order_against_analytical"
                    ],
                    "current_balance": finest["balances"]["current"],
                    "power_balance": finest["balances"]["power"],
                    "steady_residual": finest["solver"]["residual"],
                    "source": filename,
                    "source_sha256": table_sources[filename],
                }
            )
    if table_implementations != {solver_identity}:
        raise ValueError("Benchmark A records were not generated by one solver implementation")

    implementation = {
        **dict(solver_identity),
        "runners": {
            "acceptance_builder_sha256": _sha256(Path(__file__)),
            "ha20_ladder_sha256": ha20_implementation["runner_sha256"],
            "table_i_sha256": table_runner_sha256,
        },
    }

    finite_pass = all(item["pass"] for item in finite_cases.values())
    conservation_pass = all(item["pass"] for item in conservation_cases.values())
    continuum_pass = all(item["pass"] for item in continuum_cases.values())
    literature_pass = len(table_rows) == 8 and all(row["pass"] for row in table_rows)
    result = {
        "schema_version": 1,
        "benchmark": "A: fully developed laminar inductionless MHD ducts",
        "implementation": implementation,
        "research_grade_validation_pass": (
            finite_pass and conservation_pass and continuum_pass and literature_pass
        ),
        "finite_grid_freemhd": {
            "pass": finite_pass,
            "target": PROFILE_TARGET,
            "confirmation_level": power["confirmation_level"],
            "cases": finite_cases,
        },
        "analytical_continuum_audit": {
            "pass": continuum_pass,
            "criterion": "LMX is no worse than the processed finite-grid FreeMHD reference against the supplied analytical profile",
            "cases": continuum_cases,
        },
        "conservation_and_power": {
            "pass": conservation_pass,
            "cases": conservation_cases,
        },
        "richardson_diagnostic": {
            "acceptance_gate": False,
            "all_extrapolated_primary_pass": bool(
                richardson.get("research_grade_validation_pass")
            ),
            "cases": {
                case: {
                    "fine_primary_pass": richardson["cases"][case]["fine_primary_pass"],
                    "extrapolated_primary_pass": richardson["cases"][case][
                        "extrapolated_primary_pass"
                    ],
                }
                for case in CASES
            },
        },
        "literature_table_i": {"pass": literature_pass, "rows": table_rows},
        "sources": {
            **{path.name: _sha256(path) for path in ha20_paths.values()},
            **table_sources,
        },
    }
    if not result["research_grade_validation_pass"]:
        raise ValueError("Benchmark A component evidence does not satisfy the acceptance gates")
    return result


def write_acceptance(results_dir: Path, output: Path) -> dict[str, Any]:
    payload = build_acceptance(results_dir)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(output)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("benchmarks/results"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/results/benchmark-a-acceptance.json"),
    )
    args = parser.parse_args()
    try:
        payload = write_acceptance(args.results_dir, args.output)
    except ValueError as exc:
        parser.error(str(exc))
    print(f"{args.output} sha256={_sha256(args.output)} rows={len(payload['literature_table_i']['rows'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
