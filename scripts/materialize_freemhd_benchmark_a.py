#!/usr/bin/env python3
"""Generate audited Ha=20 FreeMHD smoke cases from the local demo templates."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

from lmx.freemhd import audit_freemhd_case_against_spec, load_benchmark_a_spec


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


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

    source = Path(template_dir).resolve()
    destination = Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing FreeMHD case {destination}")
    spec = load_benchmark_a_spec(case_kind, spec_dir)
    shutil.copytree(source, destination)
    vector = " ".join(f"{float(value):.16g}" for value in spec["magnetic_field"]["vector"])
    old_b0 = r"\(\s*0(?:\.0+)?\s+10(?:\.0+)?\s+0(?:\.0+)?\s*\)"
    changed: dict[str, int] = {}

    for path in sorted((destination / "0").glob("*/B0")):
        count = _replace(path, old_b0, f"( {vector} )", required=False)
        if count:
            changed[path.relative_to(destination).as_posix()] = count
    for path in sorted((destination / "system").glob("*/changeDictionaryDict")):
        if re.search(r"\bB0\s*\{", path.read_text(encoding="utf-8")):
            changed[path.relative_to(destination).as_posix()] = _replace(path, old_b0, f"( {vector} )")

    velocity = float(spec["drive"]["reference_mean_velocity"])
    flow_rate = float(spec["drive"]["target_flow_rate"])
    for path in (destination / "0" / "liquid" / "U", destination / "system" / "liquid" / "changeDictionaryDict"):
        count = _replace(path, r"(?<![0-9.])0\.9725(?![0-9.])", f"{velocity:.16g}")
        count += _replace(
            path,
            r"(volumetricFlowRate\s+(?:constant\s+)?)[0-9eE+.\-]+",
            rf"\g<1>{flow_rate:.16g}",
        )
        changed[path.relative_to(destination).as_posix()] = count

    fluid = spec["fluid"]
    liquid = destination / "constant" / "liquid" / "thermophysicalProperties.liquidMetal"
    count = _replace(liquid, r"(\brho\s+)[0-9eE+.\-]+(\s*;)", rf"\g<1>{float(fluid['density']):.16g}\g<2>")
    count += _replace(liquid, r"(\bmu\s+)[0-9eE+.\-]+(\s*;)", rf"\g<1>{float(fluid['dynamic_viscosity']):.16g}\g<2>")
    count += _replace(
        liquid,
        r"(\belcond(?:\s+\[[^\]]+\])?\s*)[0-9eE+.\-]+(\s*;)",
        rf"\g<1>{float(fluid['conductivity']):.16g}\g<2>",
    )
    changed[liquid.relative_to(destination).as_posix()] = count

    wall = spec["wall"]
    for region, conductivity in (("solidWalls", wall["conducting_wall_conductivity"]), ("insulator", wall["insulating_wall_conductivity"])):
        path = destination / "constant" / region / "thermophysicalProperties"
        changed[path.relative_to(destination).as_posix()] = _replace(
            path, r"(\belcond\s+)[0-9eE+.\-]+(\s*;)", rf"\g<1>{float(conductivity):.16g}\g<2>"
        )

    geometry = spec["geometry"]
    mesh = destination / "system" / "blockMeshDict"
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
    portable_audit = {**audit, "reference_case_dir": "."}
    manifest = {
        "schema_version": 1,
        "case_kind": case_kind,
        "run_profile": "docker_smoke_only",
        "source_template": source.name,
        "source_template_sha256": _tree_sha256(source),
        "spec_id": spec["id"],
        "spec_path": spec["path"],
        "spec_sha256": spec["sha256"],
        "changed_files": changed,
        "case_tree_sha256_before_manifest": _tree_sha256(destination),
        "audit": portable_audit,
    }
    (destination / "lmx-benchmark-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case_kind", choices=("shercliff", "hunt"))
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--template-root",
        type=Path,
        default=Path("/Users/rogerio/local/tests/freemhd_install/cases"),
        help="directory containing shercliff_demo and hunt_demo",
    )
    args = parser.parse_args()
    manifest = materialize_matched_freemhd_case(
        args.template_root / f"{args.case_kind}_demo",
        args.output_dir,
        case_kind=args.case_kind,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
