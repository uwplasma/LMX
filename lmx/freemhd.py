from __future__ import annotations

import hashlib
import math
import os
import re
import stat
from dataclasses import replace
from pathlib import Path, PurePosixPath
import unicodedata

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

from .cases import make_hartmann_case, make_hunt_case, make_shercliff_case
from .specs import BoundaryCondition, CaseSpec
from .units import (
    dynamic_to_kinematic_viscosity,
    hartmann_number,
    wall_conductance_ratio,
)


BENCHMARK_A_SPEC_DIR = Path(__file__).resolve().parents[1] / "benchmarks" / "specs"
SAMPER_TABLE_I_PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "references" / "samper-table-i.toml"
_MATCHED_B_ARTIFACT_NAMES = (
    "lmx_source", "freemhd_source", "lmx_input", "freemhd_input", "evaluator", "lmx_output", "freemhd_output"
)
_TREE_HASH_TAG = b"LMX-ARTIFACT-TREE-v1\0"


def _frame(value: bytes) -> bytes:
    return len(value).to_bytes(8, "big") + value


def _file_sha256(path: Path) -> str:
    before = os.stat(path, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode) or before.st_size == 0:
        raise ValueError("content.empty")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = os.stat(path, follow_symlinks=False)
    before_signature = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    after_signature = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if before_signature != after_signature:
        raise ValueError("content.changed")
    return digest.hexdigest()


def _tree_entries(root: Path) -> list[tuple[str, Path, bytes]]:
    entries, aliases, files = [], set(), set()
    for path in root.rglob("*"):
        relative = path.relative_to(root).as_posix()
        alias = unicodedata.normalize("NFC", relative).casefold()
        if alias in aliases:
            raise ValueError("tree.name_collision")
        aliases.add(alias)
        metadata = os.lstat(path)
        mode = metadata.st_mode
        if stat.S_ISLNK(mode):
            raise ValueError("tree.symlink")
        if stat.S_ISDIR(mode):
            kind = b"d"
        elif stat.S_ISREG(mode):
            kind = b"f"
            identity = (metadata.st_dev, metadata.st_ino)
            if identity in files:
                raise ValueError("tree.hardlink")
            files.add(identity)
        else:
            raise ValueError("tree.special")
        entries.append((relative, path, kind))
    if not entries:
        raise ValueError("content.empty")
    return sorted(entries)


def artifact_sha256(path: str | Path, kind: str) -> str:
    """Hash one immutable evidence file or portable directory tree."""

    source = Path(path)
    mode = os.lstat(source).st_mode
    if stat.S_ISLNK(mode):
        raise ValueError("path.symlink")
    if kind == "file":
        return _file_sha256(source)
    if kind != "tree" or not stat.S_ISDIR(mode):
        raise ValueError("kind")
    entries = _tree_entries(source)
    digest = hashlib.sha256(_TREE_HASH_TAG)
    for relative, child, entry_kind in entries:
        digest.update(_frame(entry_kind))
        digest.update(_frame(relative.encode("utf-8")))
        if entry_kind == b"f":
            digest.update(_frame(bytes.fromhex(_file_sha256(child))))
    if [(name, kind) for name, _, kind in entries] != [(name, kind) for name, _, kind in _tree_entries(source)]:
        raise ValueError("tree.changed")
    return digest.hexdigest()


def _resolve_artifact(root: Path, entry: object) -> tuple[Path, str, str]:
    if not isinstance(entry, dict) or set(entry) != {"path", "kind", "sha256"}:
        raise ValueError("entry")
    raw, kind, expected = entry["path"], entry["kind"], entry["sha256"]
    if not isinstance(raw, str) or not isinstance(kind, str) or not isinstance(expected, str):
        raise ValueError("entry")
    portable = PurePosixPath(raw)
    if portable.is_absolute() or not portable.parts or portable.parts[0].endswith(":"):
        raise ValueError("path.absolute")
    if "\\" in raw or any(part in {"", ".", ".."} for part in portable.parts) or portable.as_posix() != raw:
        raise ValueError("path.noncanonical")
    candidate = root.joinpath(*portable.parts)
    current = root
    for part in portable.parts:
        current /= part
        if current.is_symlink():
            raise ValueError("path.symlink")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (FileNotFoundError, ValueError) as error:
        raise ValueError("path.missing_or_escape") from error
    return resolved, kind, expected


def candidate_u_paths(case_dir: str | Path) -> list[Path]:
    root = Path(case_dir)
    return [
        root / "case" / "0" / "liquid" / "U",
        root / "case" / "0" / "fluid" / "U",
        root / "case" / "0" / "U",
        root / "0" / "liquid" / "U",
        root / "0" / "fluid" / "U",
        root / "0" / "U",
        root / "latestTime" / "liquid" / "U",
        root / "latestTime" / "fluid" / "U",
        root / "latestTime" / "U",
    ]


def _candidate_paths(case_dir: str | Path, *relative_paths: str) -> list[Path]:
    root = Path(case_dir)
    return [root / relative for relative in relative_paths]


def _first_existing(case_dir: str | Path, *relative_paths: str) -> Path | None:
    for path in _candidate_paths(case_dir, *relative_paths):
        if path.exists():
            return path
    return None


def _extract_first_scalar(text: str, *patterns: str) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match is not None:
            return float(match.group(1))
    return None


def infer_initial_velocity_x(case_dir: str | Path) -> float | None:
    pattern = re.compile(r"internalField\s+uniform\s+\(\s*(\S+)\s+\S+\s+\S+\s*\)")
    for path in candidate_u_paths(case_dir):
        if not path.exists():
            continue
        text = path.read_text()
        match = pattern.search(text)
        if match is not None:
            return float(match.group(1))
    return None


def _extract_inlet_block(text: str) -> str | None:
    boundary_match = re.search(r"boundaryField\s*\{", text)
    if boundary_match is None:
        return None
    boundary_text = text[boundary_match.end() :]
    inlet_match = re.search(r"\binlet\b\s*\{", boundary_text)
    if inlet_match is None:
        return None
    start = inlet_match.end()
    depth = 1
    index = start
    while index < len(boundary_text) and depth > 0:
        char = boundary_text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        index += 1
    if depth != 0:
        return None
    return boundary_text[start : index - 1]


def infer_inlet_flow_rate(case_dir: str | Path) -> float | None:
    pattern = re.compile(r"volumetricFlowRate\s+(?:constant\s+)?([0-9eE+.\-]+)\s*;")
    for path in candidate_u_paths(case_dir):
        if not path.exists():
            continue
        inlet_block = _extract_inlet_block(path.read_text())
        if inlet_block is None:
            continue
        match = pattern.search(inlet_block)
        if match is not None:
            return float(match.group(1))
    return None


def summarize_observable_offenders(
    records: list[dict[str, object]],
    *,
    l2_target: float = 1.0e-2,
    min_reference_peak_fraction: float = 1.0e-3,
    top_n: int | None = None,
) -> list[dict[str, object]]:
    offenders: list[dict[str, object]] = []
    for record in records:
        observables = record.get("observables", {})
        if not isinstance(observables, dict):
            continue
        for observable_name, observable_payload in observables.items():
            if not isinstance(observable_payload, dict):
                continue
            observable_reference_peak = max(
                (float(cut.get("reference_peak_abs", 1.0)) for axis in ("y", "z") if isinstance((cut := observable_payload.get(axis)), dict)),
                default=1.0,
            )
            for axis in ("y", "z"):
                cut = observable_payload.get(axis)
                if not isinstance(cut, dict):
                    continue
                l2_error = float(cut.get("l2_error", 0.0))
                linf_error = float(cut.get("linf_error", 0.0))
                peak_ratio = float(cut.get("peak_ratio", observable_payload.get("peak_ratio", 1.0)))
                reference_peak_abs = float(cut.get("reference_peak_abs", observable_reference_peak))
                reference_peak_fraction = reference_peak_abs / max(observable_reference_peak, 1.0e-20)
                low_signal = reference_peak_fraction < float(min_reference_peak_fraction)
                status = "low_signal" if low_signal else ("pass" if l2_error <= l2_target else "offender")
                offenders.append(
                    {
                        "case_kind": str(record.get("case_kind", "")),
                        "drive_mode": str(record.get("drive_mode", "")),
                        "observable": str(observable_name),
                        "axis": axis,
                        "l2_error": l2_error,
                        "linf_error": linf_error,
                        "peak_ratio": peak_ratio,
                        "reference_peak_abs": reference_peak_abs,
                        "reference_peak_fraction": reference_peak_fraction,
                        "l2_target": float(l2_target),
                        "target_ratio": l2_error / max(float(l2_target), 1.0e-20),
                        "status": status,
                    }
                )
    status_rank = {"offender": 2, "pass": 1, "low_signal": 0}
    offenders.sort(
        key=lambda item: (
            status_rank.get(str(item["status"]), 0),
            float(item["target_ratio"]),
            float(item["linf_error"]),
        ),
        reverse=True,
    )
    if top_n is not None:
        return offenders[: max(0, int(top_n))]
    return offenders


def summarize_observable_gate(
    records: list[dict[str, object]],
    *,
    l2_target: float = 1.0e-2,
    required_observables: tuple[str, ...] = (
        "velocity",
        "potential",
        "current",
        "lorentz",
    ),
    required_axes: tuple[str, ...] = ("y", "z"),
    min_reference_peak_fraction: float = 1.0e-3,
) -> dict[str, object]:
    """Summarize whether a FreeMHD parity artifact has the required observables.

    The gate is intentionally based on physical outputs rather than image
    similarity: each case must carry the requested midplane cuts and every
    non-low-signal cut must stay below the configured normalized L2 target.
    """

    missing: list[dict[str, str]] = []
    for record in records:
        case_kind = str(record.get("case_kind", ""))
        observables = record.get("observables", {})
        if not isinstance(observables, dict):
            observables = {}
        for observable_name in required_observables:
            payload = observables.get(observable_name)
            if not isinstance(payload, dict):
                missing.append({"case_kind": case_kind, "observable": observable_name, "axis": "*"})
                continue
            for axis in required_axes:
                if not isinstance(payload.get(axis), dict):
                    missing.append(
                        {
                            "case_kind": case_kind,
                            "observable": observable_name,
                            "axis": axis,
                        }
                    )

    ranked = summarize_observable_offenders(
        records,
        l2_target=l2_target,
        min_reference_peak_fraction=min_reference_peak_fraction,
    )
    offender_count = sum(1 for item in ranked if item["status"] == "offender")
    low_signal_count = sum(1 for item in ranked if item["status"] == "low_signal")
    pass_count = sum(1 for item in ranked if item["status"] == "pass")
    return {
        "case_count": len(records),
        "cases": sorted(str(record.get("case_kind", "")) for record in records),
        "l2_target": float(l2_target),
        "required_observables": list(required_observables),
        "required_axes": list(required_axes),
        "observable_pass_count": pass_count,
        "observable_offender_count": offender_count,
        "low_signal_count": low_signal_count,
        "missing_observable_count": len(missing),
        "missing_observables": missing,
        "top_observable_offenders": ranked[:8],
        "research_grade_validation_pass": offender_count == 0 and len(missing) == 0,
    }


def side_jet_profile_metrics(
    coordinate: object,
    values: object,
    *,
    center_exclusion_fraction: float = 0.02,
) -> dict[str, float]:
    """Return side-jet peak locations and amplitudes for a Hunt-style profile."""

    coord = np.asarray(coordinate, dtype=float)
    value = np.asarray(values, dtype=float)
    if coord.size == 0 or value.size == 0:
        return {
            "negative_location": 0.0,
            "positive_location": 0.0,
            "negative_value": 0.0,
            "positive_value": 0.0,
            "center_value": 0.0,
            "peak_value": 0.0,
            "peak_to_center_ratio": 0.0,
        }
    order = np.argsort(coord)
    coord = coord[order]
    value = value[order]
    half_width = max(float(np.max(np.abs(coord))), 1.0e-20)
    center_cut = float(center_exclusion_fraction) * half_width
    negative_mask = coord <= -center_cut
    positive_mask = coord >= center_cut
    if not negative_mask.any():
        negative_mask = coord <= 0.0
    if not positive_mask.any():
        positive_mask = coord >= 0.0

    negative_indices = np.flatnonzero(negative_mask)
    positive_indices = np.flatnonzero(positive_mask)
    negative_index = int(negative_indices[np.argmax(value[negative_indices])]) if negative_indices.size else int(np.argmax(value))
    positive_index = int(positive_indices[np.argmax(value[positive_indices])]) if positive_indices.size else int(np.argmax(value))
    center_value = float(np.interp(0.0, coord, value))
    peak_value = float(max(value[negative_index], value[positive_index]))
    return {
        "negative_location": float(coord[negative_index]),
        "positive_location": float(coord[positive_index]),
        "negative_value": float(value[negative_index]),
        "positive_value": float(value[positive_index]),
        "center_value": center_value,
        "peak_value": peak_value,
        "peak_to_center_ratio": peak_value / max(abs(center_value), 1.0e-20),
    }


def compare_side_jet_profiles(
    simulated_coordinate: object,
    simulated_values: object,
    reference_coordinate: object,
    reference_values: object,
) -> dict[str, object]:
    """Compare Hunt side-jet observables between a simulation and reference cut."""

    simulated = side_jet_profile_metrics(simulated_coordinate, simulated_values)
    reference = side_jet_profile_metrics(reference_coordinate, reference_values)
    location_scale = max(
        abs(float(reference["negative_location"])),
        abs(float(reference["positive_location"])),
        1.0e-20,
    )
    peak_scale = max(abs(float(reference["peak_value"])), 1.0e-20)
    return {
        "simulated": simulated,
        "reference": reference,
        "negative_location_error": abs(float(simulated["negative_location"]) - float(reference["negative_location"])),
        "positive_location_error": abs(float(simulated["positive_location"]) - float(reference["positive_location"])),
        "normalized_location_error": max(
            abs(float(simulated["negative_location"]) - float(reference["negative_location"])),
            abs(float(simulated["positive_location"]) - float(reference["positive_location"])),
        )
        / location_scale,
        "peak_value_relative_error": abs(float(simulated["peak_value"]) - float(reference["peak_value"])) / peak_scale,
        "peak_to_center_ratio_error": abs(float(simulated["peak_to_center_ratio"]) - float(reference["peak_to_center_ratio"]))
        / max(abs(float(reference["peak_to_center_ratio"])), 1.0e-20),
    }


def infer_reduced_inlet_flow_rate(
    case_dir: str | Path,
    *,
    reduced_area: float,
    initial_velocity: float | None = None,
) -> float | None:
    recovered_flow_rate = infer_inlet_flow_rate(case_dir)
    if recovered_flow_rate is None:
        return None
    speed = infer_initial_velocity_x(case_dir) if initial_velocity is None else initial_velocity
    if speed is None or abs(speed) <= 1.0e-20:
        return None
    recovered_area = recovered_flow_rate / speed
    if abs(recovered_area) <= 1.0e-20:
        return None
    return recovered_flow_rate * (reduced_area / recovered_area)


def infer_inlet_drive_mode(case_dir: str | Path) -> str | None:
    type_pattern = re.compile(r"type\s+(\S+)\s*;")
    for path in candidate_u_paths(case_dir):
        if not path.exists():
            continue
        inlet_block = _extract_inlet_block(path.read_text())
        if inlet_block is None:
            continue
        match = type_pattern.search(inlet_block)
        if match is None:
            continue
        inlet_type = match.group(1)
        if inlet_type == "flowRateInletVelocity":
            return "inlet_flow_rate"
        return "inlet_velocity"
    return None


def infer_liquid_material_properties(case_dir: str | Path) -> dict[str, float] | None:
    """Read FreeMHD liquid properties and convert OpenFOAM ``mu`` to LMX ``nu``."""

    path = _first_existing(
        case_dir,
        "case/constant/liquid/thermophysicalProperties.liquidMetal",
        "constant/liquid/thermophysicalProperties.liquidMetal",
        "case/constant/liquid/thermophysicalProperties",
        "constant/liquid/thermophysicalProperties",
    )
    if path is None:
        return None
    text = path.read_text()
    conductivity = _extract_first_scalar(text, r"\belcond\s+(?:\[[^\]]*\])?\s*([0-9eE+.\-]+)\s*;")
    if conductivity is None:
        conductivity = _extract_first_scalar(text, r"\bsigma\s+(?:\[[^\]]*\])?\s*([0-9eE+.\-]+)\s*;")
    density = _extract_first_scalar(text, r"\brho\s+([0-9eE+.\-]+)\s*;")
    dynamic_viscosity = _extract_first_scalar(text, r"\bmu\s+([0-9eE+.\-]+)\s*;")
    kinematic_viscosity = _extract_first_scalar(text, r"\bnu\s+([0-9eE+.\-]+)\s*;")
    if conductivity is None or density is None:
        return None
    if kinematic_viscosity is None:
        if dynamic_viscosity is None:
            return None
        kinematic_viscosity = dynamic_to_kinematic_viscosity(dynamic_viscosity, density)
    if dynamic_viscosity is None:
        dynamic_viscosity = kinematic_viscosity * density
    return {
        "conductivity": float(conductivity),
        "density": float(density),
        "dynamic_viscosity": float(dynamic_viscosity),
        "kinematic_viscosity": float(kinematic_viscosity),
    }


def infer_liquid_properties(case_dir: str | Path) -> tuple[float, float, float] | None:
    """Return ``(sigma, rho, nu)`` using LMX's kinematic-viscosity convention."""

    properties = infer_liquid_material_properties(case_dir)
    if properties is None:
        return None
    return (
        properties["conductivity"],
        properties["density"],
        properties["kinematic_viscosity"],
    )


def infer_solid_conductivities(
    case_dir: str | Path,
) -> tuple[float | None, float | None]:
    solid_path = _first_existing(
        case_dir,
        "case/constant/solidWalls/thermophysicalProperties",
        "constant/solidWalls/thermophysicalProperties",
    )
    insulator_path = _first_existing(
        case_dir,
        "case/constant/insulator/thermophysicalProperties",
        "constant/insulator/thermophysicalProperties",
    )
    solid_conductivity = None
    insulator_conductivity = None
    if solid_path is not None:
        solid_conductivity = _extract_first_scalar(
            solid_path.read_text(),
            r"\belcond\s+(?:\[[^\]]*\])?\s*([0-9eE+.\-]+)\s*;",
        )
    if insulator_path is not None:
        insulator_conductivity = _extract_first_scalar(
            insulator_path.read_text(),
            r"\belcond\s+(?:\[[^\]]*\])?\s*([0-9eE+.\-]+)\s*;",
        )
    return solid_conductivity, insulator_conductivity


def infer_uniform_b0(case_dir: str | Path) -> tuple[float, float, float] | None:
    path = _first_existing(
        case_dir,
        "case/0/liquid/B0",
        "0/liquid/B0",
        "latestTime/liquid/B0",
        "case/0/B0",
        "0/B0",
        "latestTime/B0",
    )
    if path is None:
        return None
    match = re.search(r"internalField\s+uniform\s+\(\s*(\S+)\s+(\S+)\s+(\S+)\s*\)", path.read_text())
    if match is None:
        return None
    return float(match.group(1)), float(match.group(2)), float(match.group(3))


def infer_rectangular_geometry(
    case_dir: str | Path,
) -> tuple[float, float, float | None, int | None] | None:
    path = _first_existing(case_dir, "case/system/blockMeshDict", "system/blockMeshDict")
    if path is None:
        return None
    text = path.read_text()
    half_width = _extract_first_scalar(text, r"\bLy\s+([0-9eE+.\-]+)\s*;")
    outer_half_width = _extract_first_scalar(text, r"\bLy_wall\s+([0-9eE+.\-]+)\s*;")
    wall_cells = _extract_first_scalar(text, r"\bN_wall\s+([0-9eE+.\-]+)\s*;")
    if half_width is None:
        return None
    wall_thickness = None
    if outer_half_width is not None and outer_half_width >= half_width:
        wall_thickness = outer_half_width - half_width
    return (
        2.0 * half_width,
        2.0 * half_width,
        wall_thickness,
        None if wall_cells is None else int(round(wall_cells)),
    )


def _infer_control_dict_scalar(case_dir: str | Path, key: str) -> float | None:
    path = _first_existing(case_dir, "case/system/controlDict", "system/controlDict", "controlDict.used")
    if path is None:
        return None
    pattern = re.compile(rf"{re.escape(key)}\s+(\S+)\s*;")
    match = pattern.search(path.read_text())
    if match is None:
        return None
    return float(match.group(1))


def _infer_block_mesh_scalar(case_dir: str | Path, key: str) -> float | None:
    path = _first_existing(case_dir, "case/system/blockMeshDict", "system/blockMeshDict")
    if path is None:
        return None
    return _extract_first_scalar(path.read_text(), rf"\b{re.escape(key)}\s+([0-9eE+.\-]+)\s*;")


def load_benchmark_a_spec(case_kind: str, spec_dir: str | Path | None = None) -> dict[str, object]:
    """Load and internally validate a canonical matched Benchmark-A TOML spec."""

    if case_kind not in {"shercliff", "hunt"}:
        raise ValueError(f"Unsupported matched Benchmark-A case {case_kind!r}")
    root = BENCHMARK_A_SPEC_DIR if spec_dir is None else Path(spec_dir)
    path = root / f"{case_kind}-ha20.toml"
    payload = tomllib.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1 or payload.get("case_kind") != case_kind:
        raise ValueError(f"Invalid matched benchmark identity in {path}")

    fluid = payload["fluid"]
    geometry = payload["geometry"]
    field = payload["magnetic_field"]
    expected_nu = dynamic_to_kinematic_viscosity(float(fluid["dynamic_viscosity"]), float(fluid["density"]))
    if not math.isclose(expected_nu, float(fluid["kinematic_viscosity"]), rel_tol=1.0e-12):
        raise ValueError(f"Inconsistent dynamic and kinematic viscosity in {path}")
    vector = [float(value) for value in field["vector"]]
    expected_ha = hartmann_number(
        magnetic_field=math.sqrt(sum(value * value for value in vector)),
        length_scale=float(geometry["length_scale"]),
        conductivity=float(fluid["conductivity"]),
        density=float(fluid["density"]),
        kinematic_viscosity=float(fluid["kinematic_viscosity"]),
    )
    if not math.isclose(expected_ha, float(field["hartmann_number"]), rel_tol=1.0e-12):
        raise ValueError(f"Magnetic field and material properties do not reproduce Ha in {path}")
    if case_kind == "hunt":
        wall = payload["wall"]
        expected_c = wall_conductance_ratio(
            wall_conductivity=float(wall["conducting_wall_conductivity"]),
            wall_thickness=float(geometry["wall_thickness"]),
            fluid_conductivity=float(fluid["conductivity"]),
            length_scale=float(geometry["length_scale"]),
        )
        if not math.isclose(expected_c, float(wall["conductance_ratio"]), rel_tol=1.0e-12):
            raise ValueError(f"Wall properties do not reproduce the conductance ratio in {path}")
    levels = payload["mesh"]["levels"]
    if len(levels) < 3 or any(len(level) != 2 for level in levels):
        raise ValueError(f"Matched benchmark mesh ladder requires at least three 2D levels in {path}")
    spacings = [1.0 / math.sqrt(float(ny) * float(nz)) for ny, nz in levels]
    if any(coarse <= fine for coarse, fine in zip(spacings, spacings[1:])):
        raise ValueError(f"Matched benchmark mesh ladder is not monotonically refined in {path}")
    refinement_ratios = [coarse / fine for coarse, fine in zip(spacings, spacings[1:])]
    if max(refinement_ratios) / min(refinement_ratios) > 1.1:
        raise ValueError(f"Matched benchmark mesh refinement ratios are too uneven in {path}")
    payload["path"] = path.relative_to(path.parents[2]).as_posix()
    payload["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return payload


def validate_matched_b_record(
    record: dict[str, object], *, expected_case_id: str, artifact_root: str | Path | None = None
) -> dict[str, object]:
    """Validate matched Benchmark-B semantics and recompute comparison gates."""

    from .benchmarks import (
        BENCHMARK_B_SPEC_FILES,
        _MATCHED_CONTRACT_SECTIONS,
        canonical_matched_b_contract,
        load_benchmark_b_reference,
        load_benchmark_b_spec,
    )

    spec = load_benchmark_b_spec(expected_case_id)
    expected_role = {
        "B1-fringing-pipe": "b1-production",
        "B2-fringing-square": "b2-production",
    }[expected_case_id]
    schema_failed: list[str] = []
    required = {
        "schema_version",
        "case_id",
        "acceptance_role",
        "contract",
        "comparison",
        "provenance",
    }
    if set(record) != required or record.get("schema_version") != 2:
        schema_failed.append("schema")
    if record.get("case_id") != expected_case_id:
        schema_failed.append("case_id")
    role = record.get("acceptance_role")
    if role not in {"harness-smoke", "b1-production", "b2-production"}:
        schema_failed.append("acceptance_role")
    if "exact_case_match" in record:
        schema_failed.append("legacy.exact_case_match")

    contract = record.get("contract")
    lmx = contract.get("lmx") if isinstance(contract, dict) else None
    freemhd = contract.get("freemhd") if isinstance(contract, dict) else None
    contract_failed: list[str] = []
    try:
        expected_contract = canonical_matched_b_contract(spec, str(role))
    except ValueError:
        expected_contract = None
        contract_failed.append("contract.acceptance_role.unavailable")
    for section in _MATCHED_CONTRACT_SECTIONS:
        left = lmx.get(section) if isinstance(lmx, dict) else None
        right = freemhd.get(section) if isinstance(freemhd, dict) else None
        if not isinstance(left, dict) or not left or not isinstance(right, dict) or not right:
            contract_failed.append(f"contract.{section}.missing")
        elif left != right:
            contract_failed.append(f"contract.{section}.mismatch")
        elif expected_contract is not None and left != expected_contract[section]:
            contract_failed.append(f"contract.{section}.canonical")

    provenance = record.get("provenance") if isinstance(record.get("provenance"), dict) else {}
    artifacts = provenance.get("artifacts") if isinstance(provenance, dict) else None
    artifact_failed: list[str] = []
    calculated_artifacts: dict[str, str] = {}
    resolved_artifacts: list[Path] = []
    if artifact_root is None:
        artifact_failed.append("provenance.artifact_root")
    else:
        try:
            root = Path(artifact_root).resolve(strict=True)
            if not root.is_dir():
                raise ValueError
        except (FileNotFoundError, ValueError):
            artifact_failed.append("provenance.artifact_root")
        else:
            if not isinstance(artifacts, dict) or set(artifacts) != set(_MATCHED_B_ARTIFACT_NAMES):
                artifact_failed.append("provenance.artifacts")
            else:
                for name in _MATCHED_B_ARTIFACT_NAMES:
                    try:
                        path, kind, expected_hash = _resolve_artifact(root, artifacts[name])
                        calculated = artifact_sha256(path, kind)
                    except (OSError, ValueError) as error:
                        artifact_failed.append(f"provenance.{name}.{error}")
                        continue
                    resolved_artifacts.append(path)
                    calculated_artifacts[name] = calculated
                    if calculated != expected_hash:
                        artifact_failed.append(f"provenance.{name}.sha256.current")
                identities = [(os.stat(path).st_dev, os.stat(path).st_ino) for path in resolved_artifacts]
                overlap = len(set(identities)) != len(identities) or any(
                    left in right.parents or right in left.parents
                    for index, left in enumerate(resolved_artifacts)
                    for right in resolved_artifacts[index + 1 :]
                )
                if overlap:
                    artifact_failed.append("provenance.artifacts.overlap")
    spec_path = BENCHMARK_A_SPEC_DIR / BENCHMARK_B_SPEC_FILES[expected_case_id]
    if provenance.get("benchmark_spec_sha256") != hashlib.sha256(spec_path.read_bytes()).hexdigest():
        artifact_failed.append("provenance.benchmark_spec_sha256.current")

    comparison = record.get("comparison")
    comparison = comparison if isinstance(comparison, dict) else {}
    metrics: dict[str, float] = {}
    comparison_failed: list[str] = []
    try:
        x = np.asarray(comparison["x_over_L"], dtype=float)
        lmx_values = np.asarray(comparison["lmx_observable"], dtype=float)
        freemhd_values = np.asarray(comparison["freemhd_observable"], dtype=float)
        reference = load_benchmark_b_reference(expected_case_id)
        reference_x = np.asarray(reference["x_over_L"], dtype=float)
        valid = (
            x.ndim == 1
            and x.size >= 2
            and lmx_values.shape == x.shape == freemhd_values.shape
            and np.all(np.isfinite(x))
            and np.all(np.isfinite(lmx_values))
            and np.all(np.isfinite(freemhd_values))
            and np.all(np.diff(x) > 0.0)
            and x[0] >= reference_x[0]
            and x[-1] <= reference_x[-1]
        )
        if not valid:
            raise ValueError
        uncertainty = np.interp(x, reference_x, np.asarray(reference["pressure_uncertainty"], dtype=float))
        delta = lmx_values - freemhd_values
        metrics = {
            "weighted_rms": float(np.sqrt(np.mean((delta / uncertainty) ** 2))),
            "weighted_linf": float(np.max(np.abs(delta / uncertainty))),
            "integrated_relative": float(
                abs(np.trapezoid(delta, x))
                / max(
                    abs(np.trapezoid(freemhd_values, x)),
                    float(np.trapezoid(uncertainty, x)),
                )
            ),
        }
    except (KeyError, TypeError, ValueError):
        comparison_failed.append("arrays")
    if metrics:
        acceptance = spec["acceptance"]
        limits = {
            "weighted_rms": float(acceptance["weighted_rms_max"]),
            "weighted_linf": float(acceptance["weighted_linf_max"]),
            "integrated_relative": float(acceptance["integrated_pressure_relative_error_max"]),
        }
        comparison_failed = [name for name, value in metrics.items() if value > limits[name]]

    schema_complete = not schema_failed
    artifact_pass = not artifact_failed
    contract_pass = schema_complete and not contract_failed
    comparison_pass = bool(metrics) and not comparison_failed
    observation_failed = ["contract.observers.unavailable"]
    observation_pass = False
    role_allows_acceptance = role == expected_role
    all_failed = schema_failed + contract_failed + artifact_failed + observation_failed + [f"comparison.{name}" for name in comparison_failed]
    return {
        "schema_complete": schema_complete,
        "artifact_pass": artifact_pass,
        "contract_pass": contract_pass,
        "observation_pass": observation_pass,
        "comparison_pass": comparison_pass,
        "role_allows_acceptance": role_allows_acceptance,
        "acceptance_pass": contract_pass and artifact_pass and observation_pass and comparison_pass and role_allows_acceptance,
        "failed_checks": all_failed,
        "metrics": metrics,
        "calculated_artifact_sha256": calculated_artifacts,
    }


def load_samper_table_i(path: str | Path | None = None) -> dict[str, object]:
    """Load and validate the supplied Samper et al. Benchmark-A Table I."""

    source = SAMPER_TABLE_I_PATH if path is None else Path(path)
    payload = tomllib.loads(source.read_text(encoding="utf-8"))
    cases = payload.get("cases", [])
    if payload.get("schema_version") != 1 or len(cases) != 8:
        raise ValueError(f"Invalid Samper Table I reference in {source}")
    expected_ha = {500, 5000, 10000, 15000}
    for case_kind, expected_conductance in (("shercliff", 0.0), ("hunt", 0.01)):
        subset = [case for case in cases if case.get("case_kind") == case_kind]
        if {int(case["hartmann_number"]) for case in subset} != expected_ha:
            raise ValueError(f"Incomplete {case_kind} Hartmann ladder in {source}")
        if any(not math.isclose(float(case["hartmann_wall_conductance"]), expected_conductance) for case in subset):
            raise ValueError(f"Incorrect {case_kind} wall conductance in {source}")
        if any(float(case["analytical_flow_rate"]) <= 0.0 for case in subset):
            raise ValueError(f"Non-positive {case_kind} flow-rate reference in {source}")
    payload["path"] = source.relative_to(source.parents[2]).as_posix()
    payload["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    return payload


def _audit_check(name: str, expected: object, observed: object, *, rel_tol: float = 1.0e-9) -> dict[str, object]:
    if isinstance(expected, (int, float)) and isinstance(observed, (int, float)):
        passed = math.isclose(float(observed), float(expected), rel_tol=rel_tol, abs_tol=1.0e-14)
    else:
        passed = observed == expected
    return {"name": name, "expected": expected, "observed": observed, "pass": passed}


def audit_freemhd_case_against_spec(
    case_dir: str | Path,
    *,
    case_kind: str,
    spec_dir: str | Path | None = None,
) -> dict[str, object]:
    """Audit a FreeMHD case against the matched spec without fitting parameters."""

    spec = load_benchmark_a_spec(case_kind, spec_dir)
    geometry = infer_rectangular_geometry(case_dir)
    fluid = infer_liquid_material_properties(case_dir)
    b0 = infer_uniform_b0(case_dir)
    solid_conductivity, insulator_conductivity = infer_solid_conductivities(case_dir)
    drive_mode = infer_inlet_drive_mode(case_dir)
    checks: list[dict[str, object]] = []

    expected_geometry = spec["geometry"]
    if geometry is None:
        checks.append(_audit_check("geometry.available", True, False))
    else:
        width, height, wall_thickness, wall_cells = geometry
        checks.extend(
            [
                _audit_check("geometry.width", expected_geometry["width"], width),
                _audit_check("geometry.height", expected_geometry["height"], height),
                _audit_check(
                    "geometry.wall_thickness",
                    expected_geometry["wall_thickness"],
                    wall_thickness,
                ),
                _audit_check("geometry.wall_cells", expected_geometry["wall_cells"], wall_cells),
            ]
        )

    expected_fluid = spec["fluid"]
    if fluid is None:
        checks.append(_audit_check("fluid.available", True, False))
    else:
        for key in (
            "conductivity",
            "density",
            "dynamic_viscosity",
            "kinematic_viscosity",
        ):
            checks.append(_audit_check(f"fluid.{key}", expected_fluid[key], fluid[key]))

    expected_field = tuple(float(value) for value in spec["magnetic_field"]["vector"])
    checks.append(_audit_check("magnetic_field.vector", expected_field, b0))
    declared_ha = _infer_block_mesh_scalar(case_dir, "Ha")
    checks.append(
        _audit_check(
            "mesh.declared_hartmann",
            spec["magnetic_field"]["hartmann_number"],
            declared_ha,
        )
    )
    physical_ha = None
    if fluid is not None and b0 is not None and geometry is not None:
        physical_ha = hartmann_number(
            magnetic_field=math.sqrt(sum(value * value for value in b0)),
            length_scale=float(expected_geometry["length_scale"]),
            conductivity=fluid["conductivity"],
            density=fluid["density"],
            kinematic_viscosity=fluid["kinematic_viscosity"],
        )
    checks.append(_audit_check("physics.hartmann", spec["magnetic_field"]["hartmann_number"], physical_ha))

    expected_wall = spec["wall"]
    checks.extend(
        [
            _audit_check(
                "wall.conducting_wall_conductivity",
                expected_wall["conducting_wall_conductivity"],
                solid_conductivity,
            ),
            _audit_check(
                "wall.insulating_wall_conductivity",
                expected_wall["insulating_wall_conductivity"],
                insulator_conductivity,
            ),
            _audit_check("drive.mode", spec["drive"]["mode"], drive_mode),
            _audit_check(
                "drive.target_flow_rate",
                spec["drive"].get("target_flow_rate"),
                infer_inlet_flow_rate(case_dir),
            ),
        ]
    )

    failed = [check for check in checks if not bool(check["pass"])]
    return {
        "case_kind": case_kind,
        "spec_id": spec["id"],
        "spec_path": spec["path"],
        "spec_sha256": spec["sha256"],
        "reference_case_dir": str(Path(case_dir)),
        "matched": not failed,
        "failed_check_count": len(failed),
        "checks": checks,
        "physical_hartmann_number": physical_ha,
        "declared_mesh_hartmann_number": declared_ha,
    }


def infer_magnetic_ramp(case_dir: str | Path) -> tuple[float, float]:
    start = _infer_control_dict_scalar(case_dir, "BtStartTime")
    duration = _infer_control_dict_scalar(case_dir, "BtDuration")
    return (0.0 if start is None else start, 0.0 if duration is None else duration)


def build_case_from_freemhd_reference(
    *,
    case_kind: str,
    ha: float,
    ny: int,
    nz: int,
    dt: float,
    t_final: float,
    max_steps: int,
    reference_run_dir: str | Path,
    forcing: float | None = None,
) -> CaseSpec:
    liquid_properties = infer_liquid_properties(reference_run_dir)
    geometry = infer_rectangular_geometry(reference_run_dir)
    b0 = infer_uniform_b0(reference_run_dir)
    solid_conductivity, insulator_conductivity = infer_solid_conductivities(reference_run_dir)
    conductivity = 1.0
    density = 1.0
    viscosity = 1.0
    if liquid_properties is not None:
        conductivity, density, viscosity = liquid_properties
    width = 2.0
    height = 2.0
    wall_thickness = 0.1
    wall_cells = 8
    if geometry is not None:
        width, height, inferred_wall_thickness, inferred_wall_cells = geometry
        if inferred_wall_thickness is not None and inferred_wall_thickness > 0.0:
            wall_thickness = inferred_wall_thickness
        if inferred_wall_cells is not None and inferred_wall_cells > 0:
            wall_cells = inferred_wall_cells
    if case_kind == "hartmann":
        case = make_hartmann_case(
            ha=ha,
            width=width,
            height=height,
            ny=ny,
            nz=nz,
            conductivity=conductivity,
            density=density,
            viscosity=viscosity,
        )
    elif case_kind == "shercliff":
        case = make_shercliff_case(
            ha=ha,
            width=width,
            height=height,
            ny=ny,
            nz=nz,
            conductivity=conductivity,
            density=density,
            viscosity=viscosity,
        )
    elif case_kind == "hunt":
        case = make_hunt_case(
            ha=ha,
            width=width,
            height=height,
            ny=ny,
            nz=nz,
            wall_cells=wall_cells,
            wall_thickness=wall_thickness,
            fluid_conductivity=conductivity,
            wall_conductivity=solid_conductivity,
            insulator_conductivity=insulator_conductivity,
            density=density,
            viscosity=viscosity,
        )
    else:
        raise ValueError(f"Unsupported FreeMHD reference case kind {case_kind!r}")
    initial_velocity = infer_initial_velocity_x(reference_run_dir) or 0.0
    ramp_start, ramp_duration = infer_magnetic_ramp(reference_run_dir)
    boundary_conditions = case.boundary_conditions
    if forcing is None:
        forcing_value = case.forcing
    else:
        forcing_value = forcing

    if forcing is None:
        drive_mode = infer_inlet_drive_mode(reference_run_dir)
        reduced_area = case.geometry.width * case.geometry.height
        reduced_inlet_flow_rate = infer_reduced_inlet_flow_rate(
            reference_run_dir,
            reduced_area=reduced_area,
            initial_velocity=initial_velocity,
        )
        if drive_mode == "inlet_flow_rate":
            forcing_value = 0.0
            flow_rate = reduced_inlet_flow_rate
            if flow_rate is None:
                flow_rate = initial_velocity * reduced_area
            inlet_bc = BoundaryCondition("inlet", "inlet_flow_rate", value=flow_rate, axis="x")
            boundary_conditions = boundary_conditions + (inlet_bc,)
        elif drive_mode == "inlet_velocity":
            forcing_value = 0.0
            inlet_bc = BoundaryCondition("inlet", "inlet_velocity", value=(initial_velocity, 0.0, 0.0), axis="x")
            boundary_conditions = boundary_conditions + (inlet_bc,)

    return replace(
        case,
        boundary_conditions=boundary_conditions,
        magnetic_field=replace(
            case.magnetic_field,
            value=case.magnetic_field.value if b0 is None else b0,
            ramp_start=ramp_start,
            ramp_duration=ramp_duration,
        ),
        initial_velocity=initial_velocity,
        forcing=forcing_value,
        time_stepper=replace(case.time_stepper, dt=dt, t_final=t_final, max_steps=max_steps),
    )
