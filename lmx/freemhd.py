from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import unicodedata
from pathlib import Path, PurePosixPath

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

from .physics import (
    dynamic_to_kinematic_viscosity,
    hartmann_number,
    interaction_parameter,
    reynolds_number,
    wall_conductance_ratio,
)

_PACKAGE_DATA = Path(__file__).with_name("data")
BENCHMARK_A_SPEC_DIR = _PACKAGE_DATA / "benchmarks" / "specs"
SAMPER_TABLE_I_PATH = _PACKAGE_DATA / "benchmarks" / "references" / "samper-table-i.toml"
_MATCHED_B_ARTIFACT_NAMES = (
    "lmx_source",
    "freemhd_source",
    "lmx_input",
    "freemhd_input",
    "evaluator",
    "lmx_output",
    "freemhd_output",
)
_MATCHED_B_ARTIFACT_KINDS = {
    "lmx_source": "tree",
    "freemhd_source": "tree",
    "lmx_input": "file",
    "freemhd_input": "tree",
    "evaluator": "file",
    "lmx_output": "file",
    "freemhd_output": "file",
}
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
    before_signature = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
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
    if [(name, kind) for name, _, kind in entries] != [
        (name, kind) for name, _, kind in _tree_entries(source)
    ]:
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


def _first_existing(case_dir: str | Path, *relative_paths: str) -> Path | None:
    for relative in relative_paths:
        path = Path(case_dir) / relative
        if path.exists():
            return path
    return None


def _extract_first_scalar(text: str, *patterns: str) -> float | None:
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.MULTILINE)
        if match is not None:
            return float(match.group(1))
    return None


def _extract_foam_block(text: str, name: str) -> str | None:
    match = re.search(rf"(?<![\w.]){re.escape(name)}\s*\{{", text)
    if match is None:
        return None
    start = match.end()
    depth = 1
    index = start
    while index < len(text) and depth > 0:
        char = text[index]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
        index += 1
    return None if depth else text[start : index - 1]


def _infer_inlet_value(case_dir: str | Path, pattern: str) -> str | None:
    root = Path(case_dir)
    expression = re.compile(pattern)
    for base in ("case/0", "0", "latestTime"):
        for region in ("liquid", "fluid", ""):
            path = root / base / region / "U"
            if not path.exists():
                continue
            boundary = _extract_foam_block(path.read_text(), "boundaryField")
            inlet = None if boundary is None else _extract_foam_block(boundary, "inlet")
            match = expression.search(inlet) if inlet is not None else None
            if match is not None:
                return match.group(1)
    return None


def infer_inlet_flow_rate(case_dir: str | Path) -> float | None:
    value = _infer_inlet_value(case_dir, r"volumetricFlowRate\s+(?:constant\s+)?([0-9eE+.\-]+)\s*;")
    return None if value is None else float(value)


def infer_inlet_drive_mode(case_dir: str | Path) -> str | None:
    inlet_type = _infer_inlet_value(case_dir, r"type\s+(\S+)\s*;")
    return (
        None
        if inlet_type is None
        else ("inlet_flow_rate" if inlet_type == "flowRateInletVelocity" else "inlet_velocity")
    )


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


def infer_solid_conductivities(
    case_dir: str | Path,
) -> tuple[float | None, float | None]:
    def conductivity(region: str) -> float | None:
        path = _first_existing(
            case_dir,
            f"case/constant/{region}/thermophysicalProperties",
            f"constant/{region}/thermophysicalProperties",
        )
        return (
            None
            if path is None
            else _extract_first_scalar(path.read_text(), r"\belcond\s+(?:\[[^\]]*\])?\s*([0-9eE+.\-]+)\s*;")
        )

    return conductivity("solidWalls"), conductivity("insulator")


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


def _decode_matched_b2_lmx_input(path: str | Path):
    from dataclasses import fields

    from ._fringing_duct import _cross_section_mesh
    from .specs import (
        BoundaryCondition,
        CaseSpec,
        ExtrudedInductionlessProblem,
        FringingProfile,
        GeometrySpec,
        MagneticFieldSpec,
        OutputSpec,
        RegionSpec,
        SolverConfig,
        TimeStepperConfig,
    )

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_top = {
        "schema_version",
        "kind",
        "case_id",
        "case",
        "scaling",
        "mesh",
        "field_profile",
        "effective_controls",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != expected_top
        or (payload.get("schema_version"), payload.get("kind"), payload.get("case_id"))
        != (1, "lmx-matched-b2-input", "B2-fringing-square")
    ):
        raise ValueError("Invalid matched B2 LMX input schema")

    def checked(cls, value, name):
        if not isinstance(value, dict) or set(value) != {item.name for item in fields(cls)}:
            raise ValueError(f"Invalid matched B2 {name} schema")
        return dict(value)

    raw = checked(CaseSpec, payload["case"], "case")
    geometry = checked(GeometrySpec, raw.pop("geometry"), "geometry")
    geometry["wall_thickness"], geometry["wall_cells"] = (
        tuple(geometry["wall_thickness"]),
        tuple(geometry["wall_cells"]),
    )
    magnetic = checked(MagneticFieldSpec, raw.pop("magnetic_field"), "magnetic field")
    magnetic["value"] = None if magnetic["value"] is None else tuple(magnetic["value"])
    boundary_payload = raw.pop("boundary_conditions")
    region_payload = raw.pop("regions")
    if not isinstance(boundary_payload, list) or not isinstance(region_payload, list):
        raise ValueError("Invalid matched B2 region or boundary schema")
    boundaries = []
    for item in boundary_payload:
        item = checked(BoundaryCondition, item, "boundary")
        item["value"] = tuple(item["value"]) if isinstance(item["value"], list) else item["value"]
        boundaries.append(BoundaryCondition(**item))
    regions = tuple(RegionSpec(**checked(RegionSpec, item, "region")) for item in region_payload)
    time_stepper = TimeStepperConfig(**checked(TimeStepperConfig, raw.pop("time_stepper"), "time stepper"))
    solver = SolverConfig(**checked(SolverConfig, raw.pop("solver"), "solver"))
    output = OutputSpec(**checked(OutputSpec, raw.pop("output"), "output"))
    raw["reference_phi_cell"] = tuple(raw["reference_phi_cell"])
    case = CaseSpec(
        **raw,
        geometry=GeometrySpec(**geometry),
        regions=regions,
        magnetic_field=MagneticFieldSpec(**magnetic),
        boundary_conditions=tuple(boundaries),
        time_stepper=time_stepper,
        solver=solver,
        output=output,
    )
    canonical_names = {
        "alex_b2-fringing-square_harness-smoke",
        "alex_b2-fringing-square_scaling-calibration",
    }
    if case.name not in canonical_names or case.geometry.kind != "layered_duct":
        raise ValueError("Matched B2 LMX input does not select the canonical solver path")
    mesh = _cross_section_mesh(case)
    mesh_payload = payload["mesh"]
    if (
        not isinstance(mesh_payload, dict)
        or set(mesh_payload) != {"coordinate_system", "x_faces", "y_faces", "z_faces"}
        or mesh_payload["coordinate_system"] != "Cartesian x-y-z faces in duct-half-width units"
    ):
        raise ValueError("Invalid matched B2 mesh schema")
    if any(
        not np.allclose(
            np.asarray(mesh_payload[f"{axis}_faces"], dtype=float),
            np.asarray(getattr(mesh, f"{axis}_faces")),
            rtol=0.0,
            atol=1.0e-15,
        )
        for axis in "xyz"
    ):
        raise ValueError("Matched B2 stored mesh faces do not reproduce the case")

    profile = payload["field_profile"]
    profile_keys = {
        "axis",
        "interpolation",
        "extrapolation",
        "source_name",
        "source_sha256",
        "anchors_sha256",
        "anchor_x_over_L",
        "anchor_b_over_B0",
        "sample_x_over_L",
        "sample_b_over_B0",
    }
    if (
        not isinstance(profile, dict)
        or set(profile) != profile_keys
        or (profile["axis"], profile["interpolation"], profile["extrapolation"])
        != ("y", "linear", "forbidden")
    ):
        raise ValueError("Invalid matched B2 field-profile schema")
    anchors_x = np.asarray(profile["anchor_x_over_L"], dtype=float)
    anchors_b = np.asarray(profile["anchor_b_over_B0"], dtype=float)
    sample_x = np.asarray(mesh.x_centers, dtype=float)
    sample_b = np.asarray(profile["sample_b_over_B0"], dtype=float)
    encoded = json.dumps(
        {"x_over_L": anchors_x.tolist(), "b_over_B0": anchors_b.tolist()},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    if (
        anchors_x.ndim != 1
        or anchors_x.shape != anchors_b.shape
        or anchors_x.size < 2
        or np.any(~np.isfinite(anchors_x))
        or np.any(~np.isfinite(anchors_b))
        or np.any(np.diff(anchors_x) <= 0.0)
        or np.any(np.diff(anchors_b) > 1.0e-12)
        or sample_x[0] < anchors_x[0]
        or sample_x[-1] > anchors_x[-1]
        or not np.array_equal(np.asarray(profile["sample_x_over_L"], dtype=float), sample_x)
        or not np.allclose(sample_b, np.interp(sample_x, anchors_x, anchors_b), rtol=0.0, atol=1.0e-15)
        or hashlib.sha256(encoded).hexdigest() != profile["anchors_sha256"]
        or re.fullmatch(r"[0-9a-f]{64}", str(profile["source_sha256"])) is None
    ):
        raise ValueError("Matched B2 field samples do not reproduce their anchors and mesh")

    scaling, controls = payload["scaling"], payload["effective_controls"]
    if (
        not isinstance(scaling, dict)
        or set(scaling)
        != {
            "length_scale",
            "half_width_m",
            "nondimensional_length",
            "velocity",
            "density",
            "conductivity",
        }
        or scaling["length_scale"] != "duct half-width"
    ):
        raise ValueError("Invalid matched B2 scaling schema")
    fluid = [region for region in regions if region.kind == "fluid"]
    wall = [region for region in regions if region.kind == "solid"]
    inlet = [bc for bc in boundaries if bc.kind == "inlet_flow_rate"]
    outlet = [bc for bc in boundaries if bc.kind == "outlet_pressure"]
    if len(fluid) != 1 or len(wall) != 1 or len(inlet) != 1 or len(outlet) != 1:
        raise ValueError("Matched B2 input requires one fluid, wall, and inlet-flow region")
    length, velocity = (
        float(scaling["nondimensional_length"]),
        float(scaling["velocity"]),
    )
    field_vector = np.asarray(case.magnetic_field.value, dtype=float)
    base_b = float(np.linalg.norm(field_vector))
    mean_velocity = float(inlet[0].value) / (case.geometry.width * case.geometry.height)
    ha = hartmann_number(
        magnetic_field=base_b,
        length_scale=length,
        conductivity=float(fluid[0].conductivity),
        density=float(fluid[0].density),
        kinematic_viscosity=float(fluid[0].viscosity),
    )
    interaction = interaction_parameter(
        magnetic_field=base_b,
        length_scale=length,
        conductivity=float(fluid[0].conductivity),
        density=float(fluid[0].density),
        velocity=velocity,
    )
    reynolds = reynolds_number(
        velocity=velocity,
        length_scale=length,
        kinematic_viscosity=float(fluid[0].viscosity),
    )
    conductance = wall_conductance_ratio(
        wall_conductivity=wall[0].conductivity,
        wall_thickness=float(wall[0].wall_thickness),
        fluid_conductivity=fluid[0].conductivity,
        length_scale=length,
    )
    if not (
        math.isclose(length, case.geometry.width / 2.0)
        and math.isclose(velocity, mean_velocity)
        and math.isclose(float(scaling["density"]), float(fluid[0].density))
        and math.isclose(float(scaling["conductivity"]), float(fluid[0].conductivity))
        and float(scaling["half_width_m"]) > 0.0
        and np.array_equal(field_vector[[0, 2]], np.zeros(2))
        and math.isclose(float(case.geometry.target_ha), ha)
        and math.isclose(ha * ha / interaction, reynolds)
        and conductance > 0.0
        and outlet[0].value == 0.0
    ):
        raise ValueError("Matched B2 materials, drive, and scaling are inconsistent")
    expected_controls = {
        "dt": min(float(case.time_stepper.dt), 0.001 / interaction),
        "electric_iterations": max(case.time_stepper.potential_iterations, 600),
        "electric_tolerance": min(case.solver.coupling_tolerance, 1.0e-12),
        "projection_iterations": max(case.time_stepper.potential_iterations, 4000),
        "projection_tolerance": min(case.solver.coupling_tolerance, 1.0e-12),
        "momentum_iterations": max(case.time_stepper.potential_iterations, 400),
        "momentum_tolerance": min(case.solver.coupling_tolerance, 1.0e-10),
        "executed_steps": case.time_stepper.max_steps,
        "steady_steps_required": 3,
        "expected_stop_reason": "step_limit",
    }
    expected_name = (
        "alex_b2-fringing-square_harness-smoke"
        if case.time_stepper.max_steps == 2
        else "alex_b2-fringing-square_scaling-calibration"
    )
    if (
        controls != expected_controls
        or not math.isclose(float(case.time_stepper.dt), float(controls.get("dt", math.nan)))
        or case.name != expected_name
    ):
        raise ValueError("Matched B2 effective controls do not reproduce the solver contract")
    problem = ExtrudedInductionlessProblem(
        case=case,
        profile=FringingProfile(x=sample_x, field_scale=sample_b, axis="y"),
    )
    return problem, mesh, payload


def load_matched_b2_lmx_input(path: str | Path):
    """Return the real solver input after independently validating stored facts."""

    return _decode_matched_b2_lmx_input(path)[0]


def _matched_b2_evaluator(
    path: str | Path | None,
) -> tuple[dict[str, object], dict[str, object]]:
    if path is None:
        return (
            {
                "primary": "excess transverse pressure difference between published A/B taps",
                "tap_geometry": "top and side wall midpoints at each axial station",
                "signed_orientation": "side (+z) minus top (+y)",
            },
            {
                "field": "B_y / B0",
                "pressure": "Delta p_AB / (sigma * U * B0^2 * half-width) minus plateau",
                "coordinate": "x / half-width",
            },
        )
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "case_id",
        "observable",
        "normalization",
    }:
        raise ValueError("Invalid matched B2 evaluator schema")
    if (payload["schema_version"], payload["case_id"]) != (1, "B2-fringing-square"):
        raise ValueError("Invalid matched B2 evaluator identity")
    observable, normalization = payload["observable"], payload["normalization"]
    if not isinstance(observable, dict) or not isinstance(normalization, dict):
        raise ValueError("Invalid matched B2 evaluator sections")
    return observable, normalization


def _contract_array(values) -> list[float]:
    return [float(f"{float(value):.15g}") for value in values]


def _contract_scalar(value: float) -> float:
    return float(f"{float(value):.14g}")


def observe_lmx_b2_contract(path: str | Path, evaluator: str | Path | None = None) -> dict[str, object]:
    """Derive the matched-B2 contract from a real LMX input, never its expected spec."""

    problem, mesh, payload = _decode_matched_b2_lmx_input(path)
    case, scaling, profile, controls = (
        problem.case,
        payload["scaling"],
        payload["field_profile"],
        payload["effective_controls"],
    )
    fluid = next(region for region in case.regions if region.kind == "fluid")
    wall = next(region for region in case.regions if region.kind == "solid")
    inlet = next(bc for bc in case.boundary_conditions if bc.kind == "inlet_flow_rate")
    length, velocity = (
        float(scaling["nondimensional_length"]),
        float(scaling["velocity"]),
    )
    magnetic_field = float(np.linalg.norm(np.asarray(case.magnetic_field.value, dtype=float)))
    ha = hartmann_number(
        magnetic_field=magnetic_field,
        length_scale=length,
        conductivity=fluid.conductivity,
        density=fluid.density,
        kinematic_viscosity=fluid.viscosity,
    )
    interaction = interaction_parameter(
        magnetic_field=magnetic_field,
        length_scale=length,
        conductivity=fluid.conductivity,
        density=fluid.density,
        velocity=velocity,
    )
    observable, normalization = _matched_b2_evaluator(evaluator)
    contract: dict[str, object] = {
        "equations": {
            "momentum": "transient incompressible Navier-Stokes-Lorentz",
            "inertia": "conservative div(rhoPhi,U)",
            "time_discretization": "Euler",
            "advection_discretization": "Gauss limitedLinear 1.0",
            "advection_assembly": "implicit fvm::div with frozen rhoPhi and limiter weights",
            "advection_vector_limiter": "single magSqr(U) limiter applied to all components",
            "gradient_discretization": "cellLimited leastSquares 1.0",
            "viscous_stress": "laminar divDevRhoReff",
            "electric_model": "inductionless Ohm law with div(J)=0",
            "phase_reduction": "alpha=1 invariant",
            "thermal_reduction": "constant temperature and properties",
        },
        "nondimensional_groups": {
            "hartmann_number": ha,
            "interaction_parameter": interaction,
            "reynolds_number": reynolds_number(
                velocity=velocity,
                length_scale=length,
                kinematic_viscosity=fluid.viscosity,
            ),
            "magnetic_reynolds_number_assumption": "Rm << 1",
        },
        "geometry": {
            "kind": "square_duct",
            "length_scale": scaling["length_scale"],
            "half_width_m": scaling["half_width_m"],
            "x_over_L_min": float(mesh.x_faces[0]),
            "x_over_L_max": float(mesh.x_faces[-1]),
            "constant_cross_section": True,
        },
        "magnetic_field": {
            "representation": "tabulated monotone interpolation",
            "components": "B = (0, B_y(x), 0) in the global Cartesian frame",
            "coordinate": "x / half-width",
            "normalization": "B_y / B0",
            "no_extrapolation": profile["extrapolation"] == "forbidden",
            "normal_current_at_axial_ends": 0.0,
        },
        "wall": {
            "model": "uniform thin conducting wall",
            "wall_conductance_ratio": wall_conductance_ratio(
                wall_conductivity=wall.conductivity,
                wall_thickness=wall.wall_thickness,
                fluid_conductivity=fluid.conductivity,
                length_scale=length,
            ),
            "numerical_realization": "explicit volumetric shell preserving c_w",
            "thickness_over_L": wall.wall_thickness / length,
            "outer_electric_boundary": "zero normal current",
        },
        "boundary_drive": {
            "velocity_inlet": "integral flow rate with extrapolated profile",
            "velocity_outlet": "zero normal gradient",
            "velocity_walls": "no slip",
            "pressure_inlet": "zero normal gradient",
            "pressure_outlet": "fixed gauge",
            "pressure_outlet_gauge": 0.0,
            "flow_constraint_scope": "inlet face only",
            "nondimensional_flow_rate": float(inlet.value),
            "electric_axial_ends": "zero normal current",
        },
        "observable": observable,
        "normalization": normalization,
        "mesh_coordinates": {
            "coordinate_system": payload["mesh"]["coordinate_system"],
            "family": "uniform 5x5 fluid grid with one explicit wall cell per side",
            "exact_coordinate_arrays_required": True,
            **{f"{axis}_faces": _contract_array(getattr(mesh, f"{axis}_faces")) for axis in "xyz"},
            "field_source": profile["source_name"],
            "field_source_sha256": profile["source_sha256"],
            "field_anchors_sha256": profile["anchors_sha256"],
            "field_sample_x_over_L": _contract_array(profile["sample_x_over_L"]),
            "field_sample_b_over_B0": _contract_array(profile["sample_b_over_B0"]),
        },
        "stopping_rules": dict(controls),
    }
    return contract


def observe_lmx_b2_output(
    output_dir: str | Path, input_path: str | Path, evaluator: str | Path
) -> dict[str, object]:
    """Replay compact LMX B2 restart evidence without trusting summary metrics."""

    from types import SimpleNamespace

    from .io import (
        load_extruded_restart_bundle,
        validate_extruded_restart_bundle,
    )
    from .validation import benchmark_b_pressure_observable

    root = Path(output_dir)
    required = {"run.json", "checkpoint.npz", "direct.npz", "resumed.npz"}
    if not root.is_dir() or {path.name for path in root.iterdir()} != required:
        raise ValueError("LMX B2 output tree is incomplete")
    metadata = json.loads((root / "run.json").read_text())
    keys = {
        "schema_version",
        "code",
        "case_id",
        "input_sha256",
        "evaluator_sha256",
        "wall_seconds",
        "num_devices",
        "float_precision",
    }
    if set(metadata) != keys or metadata.get("schema_version") != 1 or metadata.get("code") != "LMX":
        raise ValueError("LMX B2 output metadata are invalid")
    if (
        metadata.get("case_id") != "B2-fringing-square"
        or metadata.get("input_sha256") != artifact_sha256(input_path, "file")
        or metadata.get("evaluator_sha256") != artifact_sha256(evaluator, "file")
        or metadata.get("float_precision") != "float64"
        or int(metadata.get("num_devices", 0)) < 1
        or not math.isfinite(float(metadata.get("wall_seconds", math.nan)))
    ):
        raise ValueError("LMX B2 output provenance differs")
    problem = load_matched_b2_lmx_input(input_path)
    requested_steps = int(problem.case.time_stepper.max_steps)
    checkpoint_step = (requested_steps + 1) // 2
    checkpoint, direct, resumed = (
        load_extruded_restart_bundle(root / name) for name in ("checkpoint.npz", "direct.npz", "resumed.npz")
    )
    for restart in (checkpoint, direct, resumed):
        validate_extruded_restart_bundle(restart, case=problem.case)
    acceleration = problem.case.solver.coupling_acceleration
    if acceleration == "anderson":
        acceleration_name, schema, label = (
            "anderson_state",
            "extruded_anderson_v1",
            "Anderson",
        )
    elif acceleration == "aitken":
        acceleration_name, schema, label = (
            "aitken_state",
            "extruded_aitken_v1",
            "Aitken",
        )
    else:
        raise ValueError("LMX B2 output acceleration is unsupported")
    if any(
        restart.metadata.get("restart_schema") != schema or getattr(restart.bundle, acceleration_name) is None
        for restart in (checkpoint, direct, resumed)
    ):
        raise ValueError(f"LMX B2 output {label} restart state is invalid")
    if (
        checkpoint.bundle.stopping_state[0] != checkpoint_step
        or direct.bundle.stopping_state != resumed.bundle.stopping_state
        or direct.bundle.stopping_state[0] != requested_steps
        or direct.bundle.stopping_state[2] != "step_limit"
    ):
        raise ValueError("LMX B2 restart stopping state differs")
    # Keep replay-driving state separate from recomputed fields and solver histories.
    state_names = """x y z field_scale u v w p phi
        axial_pressure_loss_gradient""".split()
    flux_names = "rho_phi_plus rho_phi_inlet".split()
    derived_names = """residual volumetric_flow_rate mean_velocity
        transverse_pressure_difference jx jy jz lorentz_x lorentz_y lorentz_z axial_current
        wall_current_leakage current_scaled_pressure_proxy charge_balance_residual
        boundary_current_residual""".split()
    history_names = """iteration_component_residual_history
        iteration_pressure_linear_history iteration_electric_linear_history
        iteration_potential_residual_history iteration_residual_history
        iteration_pressure_residual_history iteration_courant_history""".split()
    grouped_differences = {name: [] for name in ("state", "flux", "derived", "history")}
    state_relative, state_tolerance_ratio, flux_relative = [], [], []
    state_tolerance_by_array = {}
    for group, names in (
        ("state", state_names),
        ("flux", flux_names),
        ("derived", derived_names),
        ("history", history_names),
    ):
        for name in names:
            left, right = (np.asarray(getattr(bundle.bundle, name)) for bundle in (direct, resumed))
            if left.shape != right.shape or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
                raise ValueError(f"LMX B2 output array {name} is invalid")
            if left.size:
                grouped_differences[group].append(float(np.max(np.abs(left - right))))
                if group in {"state", "flux"}:
                    scale = max(np.linalg.norm(left), np.linalg.norm(right), 1.0e-30)
                    relative = float(np.linalg.norm(left - right) / scale)
                    (state_relative if group == "state" else flux_relative).append(relative)
                    if group == "state":
                        tolerance = 2.0e-9 + 2.0e-8 * np.maximum(np.abs(left), np.abs(right))
                        ratio = float(np.max(np.abs(left - right) / tolerance))
                        state_tolerance_ratio.append(ratio)
                        state_tolerance_by_array[name] = ratio
    acceleration_components = (
        ("mapped", "residual", "rho_phi_plus", "rho_phi_inlet")
        if acceleration == "anderson"
        else ("residual", "relaxation", "steady_streak")
    )
    for component, left, right in zip(
        acceleration_components,
        getattr(direct.bundle, acceleration_name),
        getattr(resumed.bundle, acceleration_name),
        strict=True,
    ):
        left, right = (np.asarray(()) if value is None else np.asarray(value) for value in (left, right))
        if left.shape != right.shape or not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
            raise ValueError(f"LMX B2 output {label} state is invalid")
        if left.size:
            grouped_differences["state"].append(float(np.max(np.abs(left - right))))
            scale = max(np.linalg.norm(left), np.linalg.norm(right), 1.0e-30)
            state_relative.append(float(np.linalg.norm(left - right) / scale))
            tolerance = 2.0e-9 + 2.0e-8 * np.maximum(np.abs(left), np.abs(right))
            ratio = float(np.max(np.abs(left - right) / tolerance))
            state_tolerance_ratio.append(ratio)
            state_tolerance_by_array[f"{label.lower()}_{component}"] = ratio
    restart_differences = {name: max(values, default=0.0) for name, values in grouped_differences.items()}
    courant = np.asarray(direct.bundle.iteration_courant_history, dtype=float)
    pressure = np.asarray(
        benchmark_b_pressure_observable(SimpleNamespace(bundle=direct.bundle), "B2-fringing-square")
    )
    if (
        courant.shape != (requested_steps, 3)
        or pressure.shape != (problem.case.geometry.nx,)
        or direct.bundle.stopping_state[0] != requested_steps
    ):
        raise ValueError("LMX B2 output execution shape differs")
    return {
        "steps": requested_steps,
        "stop_reason": direct.bundle.stopping_state[2],
        "steady_streak": direct.bundle.stopping_state[1],
        "dt": courant[:, 0].tolist(),
        "courant_mean": courant[:, 1].tolist(),
        "courant_max": courant[:, 2].tolist(),
        "mass_balance": float(np.max(np.abs(np.asarray(direct.bundle.volumetric_flow_rate) - 4.0)) / 4.0),
        "current_balance": float(np.max(np.abs(np.asarray(direct.bundle.boundary_current_residual)))),
        "interface_current_balance": float(np.max(np.abs(np.asarray(direct.bundle.charge_balance_residual)))),
        "interface_current_activity": float(
            max(
                np.max(np.abs(np.asarray(direct.bundle.jx))),
                np.max(np.abs(np.asarray(direct.bundle.jz))),
            )
        ),
        "x_over_L": np.asarray(direct.bundle.x).tolist(),
        "pressure_observable": pressure.tolist(),
        "restart_max_abs": max(restart_differences.values()),
        **{f"restart_{name}_max_abs": value for name, value in restart_differences.items()},
        "restart_state_relative_l2": max(state_relative, default=0.0),
        "restart_state_tolerance_ratio": max(state_tolerance_ratio, default=0.0),
        "restart_state_tolerance_ratio_by_array": state_tolerance_by_array,
        "restart_flux_relative_l2": max(flux_relative, default=0.0),
        "wall_seconds": float(metadata["wall_seconds"]),
    }


def observe_freemhd_b2_output(
    output_dir: str | Path, input_dir: str | Path, evaluator: str | Path
) -> dict[str, object]:
    """Recompute the two-update FreeMHD smoke observables from native text output."""

    root, case = Path(output_dir), Path(input_dir)
    required = {"run.json", "controlDict.used", "run.log", "postProcessing"}
    if not root.is_dir() or {path.name for path in root.iterdir()} != required:
        raise ValueError("FreeMHD B2 output tree is incomplete")
    metadata = json.loads((root / "run.json").read_text(encoding="utf-8"))
    keys = {
        "schema_version",
        "code",
        "case_id",
        "input_sha256",
        "evaluator_sha256",
        "wall_seconds",
        "nproc",
        "image",
        "float_precision",
    }
    control = root / "controlDict.used"
    if set(metadata) != keys or (
        metadata.get("schema_version"),
        metadata.get("code"),
        metadata.get("case_id"),
    ) != (1, "FreeMHD", "B2-fringing-square"):
        raise ValueError("FreeMHD B2 output metadata are invalid")
    if (
        metadata.get("input_sha256") != artifact_sha256(case, "tree")
        or metadata.get("evaluator_sha256") != artifact_sha256(evaluator, "file")
        or control.read_bytes() != (case / "system/controlDict").read_bytes()
        or int(metadata.get("nproc", 0)) < 1
        or not str(metadata.get("image", "")).strip()
        or metadata.get("float_precision") != "float64"
        or not math.isfinite(float(metadata.get("wall_seconds", math.nan)))
        or float(metadata["wall_seconds"]) < 0.0
    ):
        raise ValueError("FreeMHD B2 output provenance differs")

    control_text = control.read_text(encoding="utf-8")
    dt_expected = 1.0 / 540000.0
    control_scalars = {
        name: _extract_first_scalar(control_text, rf"\b{name}\s+([0-9eE+.\-]+)\s*;")
        for name in ("startTime", "endTime", "deltaT", "maxDeltaT", "writeInterval")
    }
    if (
        re.search(r"\bapplication\s+epotMultiRegionInterFoam\s*;", control_text) is None
        or re.search(r"\badjustTimeStep\s+off\s*;", control_text) is None
        or re.search(r"\bwriteControl\s+timeStep\s*;", control_text) is None
        or control_scalars
        != {
            "startTime": 0.0,
            "endTime": 2.0 * dt_expected,
            "deltaT": dt_expected,
            "maxDeltaT": dt_expected,
            "writeInterval": 2.0,
        }
    ):
        raise ValueError("FreeMHD B2 effective controls differ")

    log = (root / "run.log").read_text(encoding="utf-8")
    if (
        any(
            marker.lower() in log.lower()
            for marker in ("FOAM FATAL", "Segmentation fault", "MPI_ABORT", "killed")
        )
        or re.search(r"(?im)^(?!.*trapping enabled).*Floating point exception", log)
        or re.search(r"(?i)(?:^|[\s=,(])(?:nan|[-+]?inf)(?:$|[\s,;)])", log)
    ):
        raise ValueError("FreeMHD B2 log reports a fatal failure")
    times = np.asarray([float(value) for value in re.findall(r"(?m)^Time = ([0-9eE+.\-]+)\s*$", log)])
    courant = np.asarray(
        [
            [float(mean), float(maximum)]
            for line in log.splitlines()
            if line.startswith("Region: liquid Courant Number mean:")
            for mean, maximum in re.findall(
                r"Courant Number mean:\s*([0-9eE+.\-]+)\s+max:\s*([0-9eE+.\-]+)", line
            )
        ]
    )[-2:]
    if (
        times.shape != (2,)
        or not np.all(np.isfinite(times))
        or np.any(np.diff(times) <= 0.0)
        or courant.shape != (2, 2)
        or not np.all(np.isfinite(courant))
        or re.search(r"(?m)^End\s*$", log) is None
    ):
        raise ValueError("FreeMHD B2 log execution shape differs")
    dt = np.diff(np.concatenate(([0.0], times)))

    post = root / "postProcessing"
    objects = {
        "b2PressureTaps",
        "massIn",
        "massOut",
        "currentIn",
        "currentOut",
        "currentIntoSolid",
        "currentIntoSolidMagnitude",
    }
    if not post.is_dir() or {path.name for path in post.iterdir()} != objects:
        raise ValueError("FreeMHD B2 postprocessing tree differs")

    def table(
        name: str,
        filename: str = "surfaceFieldValue.dat",
        width: int = 2,
        header: str | None = None,
    ) -> np.ndarray:
        matches = list((post / name).rglob(filename))
        if len(matches) != 1 or matches[0].is_symlink() or not matches[0].is_file():
            raise ValueError(f"FreeMHD B2 output table {name} is unavailable")
        if (
            header is not None
            and re.search(rf"(?m)^# Time\s+{re.escape(header)}\s*$", matches[0].read_text()) is None
        ):
            raise ValueError(f"FreeMHD B2 output table {name} header differs")
        rows = [
            [float(value) for value in re.findall(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?", line)]
            for line in matches[0].read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        if any(len(row) != width for row in rows):
            raise ValueError(f"FreeMHD B2 table width differs: {matches[0].name}")
        values = np.asarray(rows, dtype=float)
        if (
            values.shape != (2, width)
            or not np.all(np.isfinite(values))
            or np.any(np.diff(values[:, 0]) <= 0.0)
        ):
            raise ValueError(f"FreeMHD B2 table rows differ: {matches[0].name}")
        if not np.array_equal(values[:, 0], times):
            raise ValueError(f"FreeMHD B2 output table {name} times differ")
        return values[:, 1:]

    probe_files = list((post / "b2PressureTaps").rglob("p"))
    if len(probe_files) != 1 or probe_files[0].is_symlink() or not probe_files[0].is_file():
        raise ValueError("FreeMHD B2 pressure probe table is unavailable")
    probe_path = probe_files[0]
    probe_text = probe_path.read_text(encoding="utf-8")
    locations = re.findall(r"(?m)^# Probe (\d+) \(([^)]+)\)$", probe_text)
    if len(locations) != 16 or [int(index) for index, _ in locations] != list(range(16)):
        raise ValueError("FreeMHD B2 pressure probe headers differ")
    time_header = re.search(r"(?m)^#[ \t]+Time(?:[ \t]+(.*))?$", probe_text)
    if time_header is None or (
        time_header.group(1) and time_header.group(1).split() != list(map(str, range(16)))
    ):
        raise ValueError("FreeMHD B2 pressure probe columns differ")
    probe_points = np.asarray([[float(value) for value in point.split()] for _, point in locations])
    if not (
        np.all(np.diff(probe_points[:8, 0]) > 0.0)
        and np.array_equal(probe_points[:8, 0], probe_points[8:, 0])
        and np.allclose(probe_points[:8, 1:], (0.8, 0.0))
        and np.allclose(probe_points[8:, 1:], (0.0, 0.8))
    ):
        raise ValueError("FreeMHD B2 pressure probe geometry differs")
    probes = table("b2PressureTaps", "p", 17)
    fields = {
        "massIn": "sum(rhoPhi)",
        "massOut": "sum(rhoPhi)",
        "currentIn": "sum(jn)",
        "currentOut": "sum(jn)",
        "currentIntoSolid": "sum(jn)",
        "currentIntoSolidMagnitude": "sumMag(jn)",
    }
    fluxes = {name: table(name, header=header)[:, 0] for name, header in fields.items()}
    if np.any(fluxes["currentIntoSolidMagnitude"] < 0.0):
        raise ValueError("FreeMHD B2 interface current magnitude is negative")
    mesh = (case / "system/blockMeshDict").read_text(encoding="utf-8")
    x_min, x_max, nx = (
        _extract_first_scalar(mesh, rf"\b{name}\s+([0-9eE+.\-]+)\s*;") for name in ("xMin", "xMax", "Nx")
    )
    if None in (x_min, x_max, nx) or int(nx) != 8:
        raise ValueError("FreeMHD B2 pressure stations differ")
    x = np.linspace(float(x_min), float(x_max), int(nx) + 1)
    x = 0.5 * (x[:-1] + x[1:])
    pressure = probes[-1, 8:] - probes[-1, :8]
    pressure = (pressure - np.mean(pressure[(x <= -7.5) | (x >= 5.0)])) / 540.0
    activity = np.abs(fluxes["currentIntoSolidMagnitude"]) / math.sqrt(540.0)
    residuals: dict[str, float] = {}
    for field, value in re.findall(
        r"Solving for ([^,\n]+), Initial residual =\s*[0-9eE+.\-]+, Final residual =\s*([0-9eE+.\-]+)",
        log,
    ):
        residuals[field] = max(residuals.get(field, 0.0), float(value))
    return {
        "steps": 2,
        "stop_reason": "step_limit",
        "dt": dt.tolist(),
        "courant_mean": courant[:, 0].tolist(),
        "courant_max": courant[:, 1].tolist(),
        "mass_balance": float(np.max(np.abs(fluxes["massIn"] + fluxes["massOut"])) / 4.0),
        "current_balance": float(
            np.max(np.abs(fluxes["currentIn"] + fluxes["currentOut"])) / math.sqrt(540.0)
        ),
        "interface_current_balance": float(
            np.max(
                np.abs(fluxes["currentIntoSolid"])
                / np.maximum(np.abs(fluxes["currentIntoSolidMagnitude"]), 1.0e-30)
            )
        ),
        "interface_current_activity": float(np.max(activity)),
        "x_over_L": x.tolist(),
        "pressure_observable": pressure.tolist(),
        "residual_max": residuals,
        "wall_seconds": float(metadata["wall_seconds"]),
    }


def observe_freemhd_b2_contract(
    case_dir: str | Path, source_dir: str | Path, evaluator: str | Path | None = None
) -> dict[str, object]:
    """Derive the tiny B2 contract from effective OpenFOAM dictionaries and source bytes."""

    case, source = Path(case_dir), Path(source_dir)

    def read(relative: str) -> str:
        return (case / relative).read_text(encoding="utf-8")

    def scalar(text: str, key: str) -> float:
        value = _extract_first_scalar(text, rf"\b{re.escape(key)}\s+([0-9eE+.\-]+)\s*;")
        if value is None:
            raise ValueError(f"FreeMHD B2 input omits {key}")
        return value

    def block(text: str, name: str) -> str:
        value = _extract_foam_block(text, name)
        if value is None:
            raise ValueError(f"FreeMHD B2 input omits {name}")
        return value

    source_pin = json.loads((source / "source-pin.json").read_text(encoding="utf-8"))
    files = source_pin.get("files") if isinstance(source_pin, dict) else None
    if not isinstance(files, dict) or not files:
        raise ValueError("FreeMHD B2 source snapshot is incomplete")
    source_text: dict[str, str] = {}
    for relative, expected in files.items():
        path = source / relative
        if artifact_sha256(path, "file") != expected:
            raise ValueError(f"FreeMHD B2 source snapshot changed: {relative}")
        source_text[Path(relative).name] = path.read_text(encoding="utf-8")
    required_sources = {
        "mhdUEqn.H",
        "ePotEqn.H",
        "limitedLinear.H",
        "limitedLinear.C",
        "LimitedScheme.H",
        "NVDTVD.H",
        "LimitFuncs.C",
    }
    if not required_sources <= set(source_text):
        raise ValueError("FreeMHD B2 source snapshot lacks solver or limiter evidence")
    momentum, electric = source_text["mhdUEqn.H"], source_text["ePotEqn.H"]
    source_semantics = all(
        (
            "fvm::ddt(rho, U) + fvm::div(rhoPhi, U)" in momentum,
            "turbulence.divDevRhoReff(U)" in momentum,
            "fvm::laplacian(elcond,potE)" in electric,
            "fvc::div(psiub)" in electric,
            "JConservativeForm" in electric,
            "makeLimitedSurfaceInterpolationScheme(limitedLinear, limitedLinearLimiter)"
            in source_text["limitedLinear.C"],
            "makeLimitedSurfaceInterpolationTypeScheme(SS,LIMITER,NVDTVD,magSqr,vector)"
            in source_text["LimitedScheme.H"],
            "return Foam::magSqr(phi);" in source_text["LimitFuncs.C"],
        )
    )
    if not source_semantics:
        raise ValueError("FreeMHD B2 pinned sources do not implement the matched equations")

    mesh_text, schemes = read("system/blockMeshDict"), read("system/liquid/fvSchemes")
    x_min, x_max, half, outer = (scalar(mesh_text, key) for key in ("xMin", "xMax", "Ly", "Ly_wall"))
    nx, ny, nz, wall_cells = (int(scalar(mesh_text, key)) for key in ("Nx", "Ny", "Nz", "N_wall"))
    if (
        len(re.findall(r"\bhex\s*\(", mesh_text)) != 9
        or len(re.findall(r"\bsolidWalls\s*\(", mesh_text)) != 8
    ):
        raise ValueError("FreeMHD B2 block zones do not form one fluid plus one shell")
    x_faces = np.linspace(x_min, x_max, nx + 1)
    fluid_faces = np.linspace(-half, half, ny + 1)
    y_faces = np.concatenate(([-outer], fluid_faces, [outer]))
    z_faces = np.concatenate(([-outer], np.linspace(-half, half, nz + 1), [outer]))

    field_text = read("system/liquid/setExprFieldsDict")
    if field_text != read("system/solidWalls/setExprFieldsDict"):
        raise ValueError("FreeMHD B2 fluid and wall fields differ")
    variables = dict(re.findall(r'"([A-Za-z][A-Za-z0-9]*)=([^";]+)"', field_text))
    labels = sorted(name[1:] for name in variables if re.fullmatch(r"x[a-z]", name))
    if labels != [chr(97 + index) for index in range(len(labels))] or {f"b{label}" for label in labels} - set(
        variables
    ):
        raise ValueError("FreeMHD B2 field anchors are incomplete")
    anchors_x = np.asarray([float(variables[f"x{label}"]) for label in labels])
    anchors_b = np.asarray([float(variables[f"b{label}"]) for label in labels])
    slopes = [f"(b{right}-b{left})/(x{right}-x{left})" for left, right in zip(labels, labels[1:])]
    terms = [f"b{labels[0]}", f"{slopes[0]}*(x-x{labels[0]})"]
    terms += [
        f"({slopes[index]}-{slopes[index - 1]})*pos(x-x{labels[index]})*(x-x{labels[index]})"
        for index in range(1, len(labels) - 1)
    ]
    expression = "+".join(terms)
    actual_expression = re.search(
        r"expression\s*#\{\s*vector\(0,Bscale\*\((.*)\),0\)\s*#\};",
        field_text,
        re.DOTALL,
    )
    if actual_expression is None or re.sub(r"\s+", "", actual_expression.group(1)) != expression:
        raise ValueError("FreeMHD B2 field expression differs from its anchors")
    field_scale = math.sqrt(float(re.fullmatch(r"sqrt\(([^)]+)\)", variables["Bscale"]).group(1)))
    sample_x = 0.5 * (x_faces[:-1] + x_faces[1:])
    sample_b = np.interp(sample_x, anchors_x, anchors_b)
    anchors_sha = hashlib.sha256(
        json.dumps(
            {"x_over_L": anchors_x.tolist(), "b_over_B0": anchors_b.tolist()},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()

    def quoted(key):
        return re.search(rf"\b{key}\s+\"([^\"]+)\"\s*;", field_text).group(1)

    if quoted("lmxFieldAnchorsSHA256") != anchors_sha:
        raise ValueError("FreeMHD B2 field anchor hash differs")

    fluid = infer_liquid_material_properties(case)
    wall_conductivity, insulator = infer_solid_conductivities(case)
    if fluid is None or wall_conductivity is None or insulator is not None:
        raise ValueError("FreeMHD B2 material topology is incomplete")
    change, solution, control = (
        read("system/liquid/changeDictionaryDict"),
        read("system/liquid/fvSolution"),
        read("system/controlDict"),
    )
    u, pressure, potential = (block(change, name) for name in ("U", "p_rgh", "potE"))
    inlet, sink = (
        block(block(u, "boundaryField"), "inlet"),
        block(block(u, "boundaryField"), "sink"),
    )
    flow_rate = scalar(inlet, "volumetricFlowRate")
    velocity = flow_rate / (4.0 * half * half)
    length = half
    ha = hartmann_number(
        magnetic_field=field_scale,
        length_scale=length,
        conductivity=fluid["conductivity"],
        density=fluid["density"],
        kinematic_viscosity=fluid["kinematic_viscosity"],
    )
    interaction = interaction_parameter(
        magnetic_field=field_scale,
        length_scale=length,
        conductivity=fluid["conductivity"],
        density=fluid["density"],
        velocity=velocity,
    )
    solvers = block(solution, "solvers")
    p_solver, u_solver, e_solver = (
        block(solvers, "p_rgh"),
        block(solvers, '"(U).*"'),
        block(solvers, "potE"),
    )
    solid_e_solver = block(block(read("system/solidWalls/fvSolution"), "solvers"), "potE")
    alpha, temperature = block(change, "alpha.liquidMetal"), block(change, "T")
    solid_potential = block(read("system/solidWalls/changeDictionaryDict"), "potE")
    reductions_hold = all(
        (
            "internalField uniform 1;" in alpha,
            "internalField uniform 300;" in temperature,
            "simulationType laminar;" in read("constant/liquid/turbulenceProperties"),
            "limitVelocity" not in read("constant/liquid/fvOptions"),
            "value (0 0 0);" in read("constant/g"),
            scalar(control, "BtStartTime") == scalar(control, "BtDuration") == 0.0,
            "JConservativeForm true;" in control and "adjustTimeStep off;" in control,
            scalar(solid_e_solver, "maxIter") == scalar(e_solver, "maxIter"),
            scalar(solid_e_solver, "tolerance") == scalar(e_solver, "tolerance"),
            "type zeroGradient" in block(block(solid_potential, "boundaryField"), "outerWalls"),
        )
    )
    if not reductions_hold:
        raise ValueError("FreeMHD B2 phase, thermal, electric, or fixed-step reduction differs")
    observable, normalization = _matched_b2_evaluator(evaluator)
    zero_current = all(
        "type zeroGradient" in block(block(potential, "boundaryField"), name) for name in ("inlet", "sink")
    )
    return {
        "equations": {
            "momentum": "transient incompressible Navier-Stokes-Lorentz",
            "inertia": "conservative div(rhoPhi,U)",
            "time_discretization": "Euler" if re.search(r"default\s+Euler\s*;", schemes) else "unmatched",
            "advection_discretization": "Gauss limitedLinear 1.0"
            if "div(rhoPhi,U) Gauss limitedLinear 1.0;" in schemes
            else "unmatched",
            "advection_assembly": "implicit fvm::div with frozen rhoPhi and limiter weights",
            "advection_vector_limiter": "single magSqr(U) limiter applied to all components",
            "gradient_discretization": "cellLimited leastSquares 1.0"
            if "default cellLimited leastSquares 1.0;" in schemes
            else "unmatched",
            "viscous_stress": "laminar divDevRhoReff",
            "electric_model": "inductionless Ohm law with div(J)=0",
            "phase_reduction": "alpha=1 invariant",
            "thermal_reduction": "constant temperature and properties",
        },
        "nondimensional_groups": {
            "hartmann_number": ha,
            "interaction_parameter": interaction,
            "reynolds_number": reynolds_number(
                velocity=velocity,
                length_scale=length,
                kinematic_viscosity=fluid["kinematic_viscosity"],
            ),
            "magnetic_reynolds_number_assumption": "Rm << 1",
        },
        "geometry": {
            "kind": "square_duct",
            "length_scale": "duct half-width",
            "half_width_m": scalar(mesh_text, "physicalHalfWidth"),
            "x_over_L_min": x_min,
            "x_over_L_max": x_max,
            "constant_cross_section": True,
        },
        "magnetic_field": {
            "representation": "tabulated monotone interpolation",
            "components": "B = (0, B_y(x), 0) in the global Cartesian frame",
            "coordinate": "x / half-width",
            "normalization": "B_y / B0",
            "no_extrapolation": bool(re.search(r"\blmxExtrapolation\s+forbidden\s*;", field_text)),
            "normal_current_at_axial_ends": 0.0 if zero_current else math.nan,
        },
        "wall": {
            "model": "uniform thin conducting wall",
            "wall_conductance_ratio": _contract_scalar(
                wall_conductance_ratio(
                    wall_conductivity=wall_conductivity,
                    wall_thickness=outer - half,
                    fluid_conductivity=fluid["conductivity"],
                    length_scale=length,
                )
            ),
            "numerical_realization": "explicit volumetric shell preserving c_w",
            "thickness_over_L": _contract_scalar((outer - half) / length),
            "outer_electric_boundary": "zero normal current",
        },
        "boundary_drive": {
            "velocity_inlet": "integral flow rate with extrapolated profile",
            "velocity_outlet": "zero normal gradient" if "type zeroGradient" in sink else "unmatched",
            "velocity_walls": "no slip",
            "pressure_inlet": "zero normal gradient",
            "pressure_outlet": "fixed gauge",
            "pressure_outlet_gauge": scalar(block(block(pressure, "boundaryField"), "sink"), "value uniform"),
            "flow_constraint_scope": "inlet face only",
            "nondimensional_flow_rate": flow_rate,
            "electric_axial_ends": "zero normal current" if zero_current else "unmatched",
        },
        "observable": observable,
        "normalization": normalization,
        "mesh_coordinates": {
            "coordinate_system": "Cartesian x-y-z faces in duct-half-width units",
            "family": "uniform 5x5 fluid grid with one explicit wall cell per side",
            "exact_coordinate_arrays_required": True,
            "x_faces": _contract_array(x_faces),
            "y_faces": _contract_array(y_faces),
            "z_faces": _contract_array(z_faces),
            "field_source": quoted("lmxFieldSource"),
            "field_source_sha256": quoted("lmxFieldSourceSHA256"),
            "field_anchors_sha256": anchors_sha,
            "field_sample_x_over_L": _contract_array(sample_x),
            "field_sample_b_over_B0": _contract_array(sample_b),
        },
        "stopping_rules": {
            "dt": scalar(control, "deltaT"),
            "electric_iterations": int(scalar(e_solver, "maxIter")),
            "electric_tolerance": scalar(e_solver, "tolerance"),
            "projection_iterations": int(scalar(p_solver, "maxIter")),
            "projection_tolerance": scalar(p_solver, "tolerance"),
            "momentum_iterations": int(scalar(u_solver, "maxIter")),
            "momentum_tolerance": scalar(u_solver, "tolerance"),
            "executed_steps": round(scalar(control, "endTime") / scalar(control, "deltaT")),
            "steady_steps_required": int(scalar(control, "lmxSteadyStepsRequired")),
            "expected_stop_reason": "step_limit",
        },
    }


def _validate_b2_smoke_execution(
    lmx: dict[str, object], freemhd: dict[str, object], limits: dict[str, object]
) -> tuple[list[str], list[str], dict[str, float]]:
    execution_failed: list[str] = []
    expected_dt = 1.0 / 540000.0
    for name, observed in (("lmx", lmx), ("freemhd", freemhd)):

        def fail(gate: str) -> None:
            execution_failed.append(f"execution.{name}.{gate}")

        try:
            dt = np.asarray(observed["dt"], dtype=float)
            co_mean = np.asarray(observed["courant_mean"], dtype=float)
            co_max = np.asarray(observed["courant_max"], dtype=float)
            if observed["steps"] != limits["executed_steps"] or observed["stop_reason"] != "step_limit":
                fail("stopping")
            if (
                dt.shape != (2,)
                or not np.all(np.isfinite(dt))
                or np.any(np.abs(dt - expected_dt) > limits["dt_absolute_tolerance"])
            ):
                fail("dt")
            courant = np.concatenate((co_mean, co_max))
            if (
                co_mean.shape != (2,)
                or co_max.shape != (2,)
                or not np.all(np.isfinite(courant))
                or np.any(co_max > limits["courant_max"])
            ):
                fail("courant")
            for gate in (
                "mass_balance",
                "current_balance",
                "interface_current_balance",
            ):
                if not math.isfinite(float(observed[gate])) or float(observed[gate]) > limits[f"{gate}_max"]:
                    fail(gate)
            activity = float(observed["interface_current_activity"])
            if not math.isfinite(activity) or activity < limits["interface_current_activity_min"]:
                fail("interface_current_activity")
            if name == "lmx":
                restart = float(observed["restart_max_abs"])
                if not math.isfinite(restart) or restart > limits["restart_absolute_tolerance"]:
                    fail("restart")
        except (KeyError, TypeError, ValueError):
            fail("schema")

    comparison_failed: list[str] = []
    metrics: dict[str, float] = {}
    try:
        x_lmx, x_freemhd = (np.asarray(item["x_over_L"], dtype=float) for item in (lmx, freemhd))
        p_lmx, p_freemhd = (np.asarray(item["pressure_observable"], dtype=float) for item in (lmx, freemhd))
        if not np.array_equal(x_lmx, x_freemhd):
            comparison_failed.append("x")
        for key in ("courant_mean", "courant_max"):
            if not np.allclose(
                np.asarray(lmx[key]),
                np.asarray(freemhd[key]),
                rtol=limits["cross_code_courant_relative_tolerance"],
                atol=limits["cross_code_courant_absolute_tolerance"],
            ):
                comparison_failed.append(key)
        if (
            p_lmx.shape != x_lmx.shape
            or p_freemhd.shape != x_freemhd.shape
            or not np.all(np.isfinite(np.concatenate((x_lmx, x_freemhd, p_lmx, p_freemhd))))
        ):
            raise ValueError
        delta = p_lmx - p_freemhd
        metrics = {
            "pressure_rms": float(np.sqrt(np.mean(delta**2))),
            "pressure_linf": float(np.max(np.abs(delta))),
        }
        for key in metrics:
            if metrics[key] > limits[f"cross_code_{key}_max"]:
                comparison_failed.append(key)
    except (KeyError, TypeError, ValueError):
        comparison_failed.append("arrays")
    return execution_failed, comparison_failed, metrics


def validate_matched_b_record(
    record: dict[str, object],
    *,
    expected_case_id: str,
    artifact_root: str | Path | None = None,
) -> dict[str, object]:
    """Validate matched Benchmark-B semantics and recompute comparison gates."""

    from .validation import (
        _MATCHED_CONTRACT_SECTIONS,
        BENCHMARK_B_SPEC_FILES,
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
    schema_version = record.get("schema_version")
    role = record.get("acceptance_role")
    executed_smoke = (
        schema_version == 3 and expected_case_id == "B2-fringing-square" and role == "harness-smoke"
    )
    if (
        set(record) != required
        or schema_version not in {2, 3}
        or (schema_version == 3 and not executed_smoke)
    ):
        schema_failed.append("schema")
    if record.get("case_id") != expected_case_id:
        schema_failed.append("case_id")
    if role not in {"harness-smoke", "b1-production", "b2-production"}:
        schema_failed.append("acceptance_role")
    if "exact_case_match" in record:
        schema_failed.append("schema.exact_case_match")

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
    resolved_artifacts: dict[str, Path] = {}
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
                        expected_kind = (
                            "tree"
                            if executed_smoke and name in {"lmx_output", "freemhd_output"}
                            else _MATCHED_B_ARTIFACT_KINDS[name]
                        )
                        if kind != expected_kind:
                            raise ValueError("kind")
                        calculated = artifact_sha256(path, kind)
                    except (OSError, ValueError) as error:
                        artifact_failed.append(f"provenance.{name}.{error}")
                        continue
                    resolved_artifacts[name] = path
                    calculated_artifacts[name] = calculated
                    if calculated != expected_hash:
                        artifact_failed.append(f"provenance.{name}.sha256.current")
                paths = list(resolved_artifacts.values())
                identities = [(os.stat(path).st_dev, os.stat(path).st_ino) for path in paths]
                overlap = len(set(identities)) != len(identities) or any(
                    left in right.parents or right in left.parents
                    for index, left in enumerate(paths)
                    for right in paths[index + 1 :]
                )
                if overlap:
                    artifact_failed.append("provenance.artifacts.overlap")
    spec_path = BENCHMARK_A_SPEC_DIR / BENCHMARK_B_SPEC_FILES[expected_case_id]
    if provenance.get("benchmark_spec_sha256") != hashlib.sha256(spec_path.read_bytes()).hexdigest():
        artifact_failed.append("provenance.benchmark_spec_sha256.current")
    if role == "harness-smoke" and "freemhd_source" in resolved_artifacts:
        try:
            source_pin = json.loads((resolved_artifacts["freemhd_source"] / "source-pin.json").read_text())
            reference = spec["free_mhd_discretization_reference"]
            expected_files = {
                reference[key]: reference[f"{key}_sha256"] for key in reference if key.endswith("_source")
            }
            if (
                source_pin.get("commit") != reference["repository_commit"]
                or source_pin.get("openfoam_release") != reference["openfoam_release"]
                or source_pin.get("files") != dict(sorted(expected_files.items()))
            ):
                raise ValueError
        except (
            AttributeError,
            KeyError,
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ):
            artifact_failed.append("provenance.freemhd_source.pin")

    comparison = record.get("comparison")
    comparison = comparison if isinstance(comparison, dict) else {}
    metrics: dict[str, float] = {}
    comparison_failed: list[str] = []
    if executed_smoke:
        if comparison != {"source": "independent-output-observers"}:
            schema_failed.append("comparison.source")
    else:
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
            uncertainty = np.interp(
                x,
                reference_x,
                np.asarray(reference["pressure_uncertainty"], dtype=float),
            )
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
    observation_failed: list[str] = []
    observed_outputs: dict[str, dict[str, object]] = {}
    if (
        expected_case_id == "B2-fringing-square"
        and role == "harness-smoke"
        and all(
            name in resolved_artifacts
            for name in ("lmx_input", "freemhd_input", "freemhd_source", "evaluator")
        )
    ):
        try:
            observed_lmx = observe_lmx_b2_contract(
                resolved_artifacts["lmx_input"], resolved_artifacts["evaluator"]
            )
            observed_freemhd = observe_freemhd_b2_contract(
                resolved_artifacts["freemhd_input"],
                resolved_artifacts["freemhd_source"],
                resolved_artifacts["evaluator"],
            )

            def differences(left: object, right: object, prefix: str) -> list[str]:
                if isinstance(left, dict) and isinstance(right, dict):
                    keys = sorted(set(left) | set(right))
                    return [
                        item
                        for key in keys
                        for item in differences(left.get(key), right.get(key), f"{prefix}.{key}")
                    ]
                if isinstance(left, list) and isinstance(right, list):
                    return [] if left == right else [prefix]
                return [] if left == right else [prefix]

            observation_failed += [
                f"{path}.lmx_observed" for path in differences(lmx, observed_lmx, "contract")
            ]
            observation_failed += [
                f"{path}.freemhd_observed" for path in differences(freemhd, observed_freemhd, "contract")
            ]
            observation_failed += [
                f"{path}.observer_mismatch"
                for path in differences(observed_lmx, observed_freemhd, "contract")
            ]
            if executed_smoke:
                observed_outputs = {
                    "lmx": observe_lmx_b2_output(
                        resolved_artifacts["lmx_output"],
                        resolved_artifacts["lmx_input"],
                        resolved_artifacts["evaluator"],
                    ),
                    "freemhd": observe_freemhd_b2_output(
                        resolved_artifacts["freemhd_output"],
                        resolved_artifacts["freemhd_input"],
                        resolved_artifacts["evaluator"],
                    ),
                }
        except (
            KeyError,
            OSError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            observation_failed.append(f"contract.observers.error.{type(error).__name__}")
    else:
        observation_failed.append("contract.observers.unavailable")
    observation_pass = not observation_failed
    execution_failed: list[str] = []
    if executed_smoke:
        execution_failed, comparison_failed, metrics = _validate_b2_smoke_execution(
            observed_outputs.get("lmx", {}),
            observed_outputs.get("freemhd", {}),
            spec["harness_smoke_execution"],
        )
        comparison_pass = not comparison_failed
    role_allows_acceptance = role == expected_role
    all_failed = (
        schema_failed
        + contract_failed
        + artifact_failed
        + observation_failed
        + execution_failed
        + [f"comparison.{name}" for name in comparison_failed]
    )
    report = {
        "schema_complete": schema_complete,
        "artifact_pass": artifact_pass,
        "contract_pass": contract_pass,
        "observation_pass": observation_pass,
        "comparison_pass": comparison_pass,
        "role_allows_acceptance": role_allows_acceptance,
        "acceptance_pass": contract_pass
        and artifact_pass
        and observation_pass
        and comparison_pass
        and role_allows_acceptance,
        "failed_checks": all_failed,
        "metrics": metrics,
        "calculated_artifact_sha256": calculated_artifacts,
    }
    if executed_smoke:
        report["execution_pass"] = (
            schema_complete and artifact_pass and contract_pass and observation_pass and not execution_failed
        )
    return report


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
        if any(
            not math.isclose(float(case["hartmann_wall_conductance"]), expected_conductance)
            for case in subset
        ):
            raise ValueError(f"Incorrect {case_kind} wall conductance in {source}")
        if any(float(case["analytical_flow_rate"]) <= 0.0 for case in subset):
            raise ValueError(f"Non-positive {case_kind} flow-rate reference in {source}")
    payload["path"] = source.relative_to(source.parents[2]).as_posix()
    payload["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    return payload
