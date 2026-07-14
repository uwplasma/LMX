#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lmx.benchmarks import build_benchmark_b_problem, load_benchmark_b_reference, load_benchmark_b_spec
from lmx.freemhd import artifact_sha256, audit_freemhd_case_against_spec, load_benchmark_a_spec
from lmx.reference_data import default_closed_channel_reference_root


DEFAULT_FREEMHD_INSTALL_DIR = Path("/Users/rogerio/local/tests/freemhd_install")
DEFAULT_PROCESSED_ROOT = default_closed_channel_reference_root()

_FREEMHD_SOURCE_NAMES = ("momentum", "electric", "limiter", "nvd", "vector_transform")


def materialize_freemhd_source_snapshot(
    source_repo: str | Path,
    output_dir: str | Path,
    case_id: str = "B2-fringing-square",
    spec_root: str | Path | None = None,
) -> dict[str, object]:
    """Copy the exact, clean FreeMHD/OpenFOAM source bytes frozen by Benchmark B."""

    repository, destination = Path(source_repo).resolve(), Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing FreeMHD source snapshot {destination}")
    reference = load_benchmark_b_spec(case_id, spec_root)["free_mhd_discretization_reference"]

    def git(*args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(["git", "-C", str(repository), *args], capture_output=True, text=True)

    top, head = git("rev-parse", "--show-toplevel"), git("rev-parse", "HEAD")
    if top.returncode or Path(top.stdout.strip()).resolve() != repository:
        raise ValueError("FreeMHD source repository must be its Git worktree root")
    if head.returncode or head.stdout.strip() != reference["repository_commit"]:
        raise ValueError("FreeMHD repository HEAD does not match the frozen commit")
    paths: list[str] = []
    files: dict[str, str] = {}
    for name in _FREEMHD_SOURCE_NAMES:
        source_key = f"{name}_source"
        relative = reference.get(source_key)
        pure = PurePosixPath(relative) if isinstance(relative, str) else PurePosixPath()
        if not pure.parts or pure.is_absolute() or ".." in pure.parts or relative != pure.as_posix() or "\\" in relative:
            raise ValueError(f"Noncanonical FreeMHD source path for {source_key}")
        path = repository / relative
        try:
            path.resolve(strict=True).relative_to(repository)
        except (OSError, ValueError) as exc:
            raise ValueError(f"FreeMHD source is missing or escapes its repository: {relative}") from exc
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"FreeMHD source is not a regular nonsymlink: {relative}")
        tracked = git("ls-files", "--stage", "--error-unmatch", "--", relative)
        if tracked.returncode or not tracked.stdout.startswith("100"):
            raise ValueError(f"FreeMHD source is not a tracked regular file: {relative}")
        paths.append(relative)
        files[relative] = str(reference[f"{source_key}_sha256"])
    if git("diff", "--quiet", "--", *paths).returncode or git("diff", "--cached", "--quiet", "--", *paths).returncode:
        raise ValueError("Frozen FreeMHD source paths have staged or unstaged changes")
    for relative, expected in files.items():
        if artifact_sha256(repository / relative, "file") != expected:
            raise ValueError(f"FreeMHD source SHA-256 does not match the frozen specification: {relative}")

    manifest: dict[str, object] = {
        "schema_version": 1,
        "project": "FreeMHD",
        "commit": reference["repository_commit"],
        "openfoam_release": reference["openfoam_release"],
        "files": dict(sorted(files.items())),
    }
    destination.mkdir(parents=True)
    for relative in sorted(paths):
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository / relative, target)
    (destination / "source-pin.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def _tiny_b2_problem(spec_root: str | Path | None = None) -> tuple[object, dict[str, object]]:
    from lmx._fringing_types import ExtrudedInductionlessProblem, FringingProfile
    from lmx.fringing import _cross_section_mesh

    problem = build_benchmark_b_problem("B2-fringing-square", mesh_level="coarse", root=spec_root)
    dt = 1.0 / 540000.0
    case = replace(
        problem.case,
        name="alex_b2-fringing-square_harness-smoke",
        geometry=replace(problem.case.geometry, nx=8, ny=5, nz=5, wall_cells=(1, 1, 1, 1)),
        time_stepper=replace(problem.case.time_stepper, dt=dt, t_final=2.0 * dt, max_steps=2),
    )
    mesh = _cross_section_mesh(case)
    reference = load_benchmark_b_reference("B2-fringing-square", spec_root)
    anchors_x = np.asarray(reference["x_over_L"], dtype=float)
    anchors_b = np.asarray(reference["b_over_B0"], dtype=float)
    sample_x = np.asarray(mesh.x_centers, dtype=float)
    sample_b = np.interp(sample_x, anchors_x, anchors_b)
    profile = FringingProfile(x=sample_x, field_scale=sample_b, axis="y")
    spec = load_benchmark_b_spec("B2-fringing-square", spec_root)
    anchors = {"x_over_L": anchors_x.tolist(), "b_over_B0": anchors_b.tolist()}
    anchor_bytes = json.dumps(anchors, sort_keys=True, separators=(",", ":")).encode()
    payload: dict[str, object] = {
        "schema_version": 1,
        "kind": "lmx-matched-b2-input",
        "case_id": "B2-fringing-square",
        "case": asdict(case),
        "scaling": {
            "length_scale": "duct half-width",
            "half_width_m": float(spec["geometry"]["half_width_m"]),
            "nondimensional_length": 1.0,
            "velocity": 1.0,
            "density": 1.0,
            "conductivity": 1.0,
        },
        "mesh": {
            "coordinate_system": "Cartesian x-y-z faces in duct-half-width units",
            **{
                f"{axis}_faces": np.asarray(getattr(mesh, f"{axis}_faces"), dtype=float).tolist()
                for axis in "xyz"
            },
        },
        "field_profile": {
            "axis": "y",
            "interpolation": "linear",
            "extrapolation": "forbidden",
            "source_name": Path(spec["reference"]["data_path"]).name,
            "source_sha256": spec["reference"]["data_sha256"],
            "anchors_sha256": hashlib.sha256(anchor_bytes).hexdigest(),
            "anchor_x_over_L": anchors_x.tolist(),
            "anchor_b_over_B0": anchors_b.tolist(),
            "sample_x_over_L": sample_x.tolist(),
            "sample_b_over_B0": sample_b.tolist(),
        },
        "effective_controls": {
            "dt": dt,
            "electric_iterations": 600,
            "electric_tolerance": 1.0e-12,
            "projection_iterations": 4000,
            "projection_tolerance": 1.0e-12,
            "momentum_iterations": 400,
            "momentum_tolerance": 1.0e-10,
            "executed_steps": 2,
            "steady_steps_required": 3,
            "expected_stop_reason": "step_limit",
        },
    }
    return ExtrudedInductionlessProblem(case=case, profile=profile), payload


def materialize_matched_b2_lmx_input(
    output_file: str | Path, *, spec_root: str | Path | None = None
) -> dict[str, object]:
    """Write the deterministic real LMX input for the tiny matched-B2 smoke."""

    destination = Path(output_file)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing LMX B2 input {destination}")
    _, payload = _tiny_b2_problem(spec_root)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


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


def _max_metric(records: list[dict[str, Any]], path: tuple[str | None, ...]) -> float | None:
    values: list[Any] = list(records)
    for segment in path:
        values = [
            child
            for value in values
            if isinstance(value, dict)
            for child in (
                value.values()
                if segment is None
                else ([value[segment]] if segment in value else [])
            )
        ]
    return max(map(float, values)) if values else None


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
                value = _max_metric(records, (key,))
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
        y_value = _max_metric(records, ("observables", None, "y", "l2_error"))
        z_value = _max_metric(records, ("observables", None, "z", "l2_error"))
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
