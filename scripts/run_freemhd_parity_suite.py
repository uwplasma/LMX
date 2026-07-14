#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lmx.freemhd import artifact_sha256, audit_freemhd_case_against_spec, load_benchmark_a_spec
from lmx.reference_data import default_closed_channel_reference_root


DEFAULT_FREEMHD_INSTALL_DIR = Path("/Users/rogerio/local/tests/freemhd_install")
DEFAULT_PROCESSED_ROOT = default_closed_channel_reference_root()


def _replace(path: Path, pattern: str, replacement: str, *, required: bool = True) -> int:
    updated, count = re.subn(pattern, replacement, path.read_text(encoding="utf-8"))
    if required and count == 0:
        raise ValueError(f"Expected input was not found in FreeMHD template file {path}")
    if count:
        path.write_text(updated, encoding="utf-8")
    return count


def materialize_matched_freemhd_case(
    template_dir: str | Path,
    output_dir: str | Path,
    *,
    case_kind: str,
    spec_dir: str | Path | None = None,
) -> dict[str, object]:
    """Copy and patch a demo into an audited canonical Benchmark-A smoke case."""

    source, destination = Path(template_dir).resolve(), Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing FreeMHD case {destination}")
    spec = load_benchmark_a_spec(case_kind, spec_dir)
    shutil.copytree(source, destination)
    vector = " ".join(f"{float(value):.16g}" for value in spec["magnetic_field"]["vector"])
    old_b0 = r"\(\s*0(?:\.0+)?\s+10(?:\.0+)?\s+0(?:\.0+)?\s*\)"
    changed: dict[str, int] = {}
    for path in sorted((destination / "0").glob("*/B0")):
        if count := _replace(path, old_b0, f"( {vector} )", required=False):
            changed[path.relative_to(destination).as_posix()] = count
    for path in sorted((destination / "system").glob("*/changeDictionaryDict")):
        if re.search(r"\bB0\s*\{", path.read_text(encoding="utf-8")):
            changed[path.relative_to(destination).as_posix()] = _replace(path, old_b0, f"( {vector} )")

    velocity, flow_rate = float(spec["drive"]["reference_mean_velocity"]), float(spec["drive"]["target_flow_rate"])
    for path in (destination / "0/liquid/U", destination / "system/liquid/changeDictionaryDict"):
        count = _replace(path, r"(?<![0-9.])0\.9725(?![0-9.])", f"{velocity:.16g}")
        count += _replace(path, r"(volumetricFlowRate\s+(?:constant\s+)?)[0-9eE+.\-]+", rf"\g<1>{flow_rate:.16g}")
        changed[path.relative_to(destination).as_posix()] = count

    fluid = spec["fluid"]
    liquid = destination / "constant/liquid/thermophysicalProperties.liquidMetal"
    substitutions = (
        (r"(\brho\s+)[0-9eE+.\-]+(\s*;)", float(fluid["density"])),
        (r"(\bmu\s+)[0-9eE+.\-]+(\s*;)", float(fluid["dynamic_viscosity"])),
        (r"(\belcond(?:\s+\[[^\]]+\])?\s*)[0-9eE+.\-]+(\s*;)", float(fluid["conductivity"])),
    )
    changed[liquid.relative_to(destination).as_posix()] = sum(
        _replace(liquid, pattern, rf"\g<1>{value:.16g}\g<2>") for pattern, value in substitutions
    )
    wall = spec["wall"]
    for region, conductivity in (("solidWalls", wall["conducting_wall_conductivity"]), ("insulator", wall["insulating_wall_conductivity"])):
        path = destination / "constant" / region / "thermophysicalProperties"
        changed[path.relative_to(destination).as_posix()] = _replace(
            path, r"(\belcond\s+)[0-9eE+.\-]+(\s*;)", rf"\g<1>{float(conductivity):.16g}\g<2>"
        )

    geometry = spec["geometry"]
    mesh = destination / "system/blockMeshDict"
    mesh_values = {
        "Ly": float(geometry["length_scale"]),
        "Ly_wall": float(geometry["length_scale"]) + float(geometry["wall_thickness"]),
        "Ha": float(spec["magnetic_field"]["hartmann_number"]),
        "N_wall": int(geometry["wall_cells"]),
    }
    changed[mesh.relative_to(destination).as_posix()] = sum(
        _replace(mesh, rf"(?m)^(\s*{key}\s+)[^;]+;", rf"\g<1>{value:.16g};")
        for key, value in mesh_values.items()
    )
    audit = audit_freemhd_case_against_spec(destination, case_kind=case_kind, spec_dir=spec_dir)
    if not audit["matched"]:
        failures = [check["name"] for check in audit["checks"] if not check["pass"]]
        raise ValueError(f"Generated FreeMHD case failed its canonical audit: {failures}")
    manifest = {
        "schema_version": 1,
        "case_kind": case_kind,
        "run_profile": "docker_smoke_only",
        "source_template": source.name,
        "source_template_sha256": artifact_sha256(source, "tree"),
        "spec_id": spec["id"],
        "spec_path": spec["path"],
        "spec_sha256": spec["sha256"],
        "changed_files": changed,
        "case_tree_sha256_before_manifest": artifact_sha256(destination, "tree"),
        "audit": {**audit, "reference_case_dir": "."},
    }
    (destination / "lmx-benchmark-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


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
    case_audits = payload.get("runs", {}).get("matched_case_audit", {})
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
    if case_audits:
        lines.extend(["", "## Matched case audit", ""])
        for case_kind, audit in sorted(case_audits.items()):
            lines.append(
                f"- {case_kind}: matched=`{audit.get('matched', False)}`, "
                f"failed checks=`{audit.get('failed_check_count', '-')}`"
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
        case_audits = {
            case_kind: audit_freemhd_case_against_spec(
                reference_output_root / case_kind,
                case_kind=case_kind,
            )
            for case_kind in ("shercliff", "hunt")
        }
        summary["runs"]["matched_case_audit"] = case_audits
        summary["matched_case_gate"] = all(bool(audit["matched"]) for audit in case_audits.values())
        if summary["matched_case_gate"]:
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
        else:
            summary["status"] = "invalid_reference"
            summary["reason"] = (
                "FreeMHD case inputs do not match the canonical Benchmark-A specifications; "
                "profile errors are not reported as parity evidence."
            )

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
        "--materialize",
        choices=("shercliff", "hunt"),
        help="materialize an audited smoke case at --output and exit without running a solver",
    )
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

    if args.materialize:
        manifest = materialize_matched_freemhd_case(
            args.freemhd_install_dir / "cases" / f"{args.materialize}_demo",
            args.output,
            case_kind=args.materialize,
        )
        print(json.dumps(manifest, indent=2, sort_keys=True))
        return 0

    summary = run_suite(
        output=args.output,
        freemhd_install_dir=args.freemhd_install_dir,
        processed_root=args.processed_root,
    )
    _write_json(args.output / "summary.json", summary)
    _write_markdown(args.output / "summary.md", summary)
    print(json.dumps(summary, indent=2))
    return 2 if summary["status"] == "invalid_reference" else 0


if __name__ == "__main__":
    raise SystemExit(main())
