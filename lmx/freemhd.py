from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from pathlib import Path, PurePosixPath
import unicodedata

import numpy as np

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

from .units import (
    dynamic_to_kinematic_viscosity,
    hartmann_number,
    interaction_parameter,
    reynolds_number,
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


def _candidate_u_paths(case_dir: str | Path):
    root = Path(case_dir)
    return (
        root / base / region / "U"
        for base in ("case/0", "0", "latestTime")
        for region in ("liquid", "fluid", "")
    )


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


def _infer_inlet_value(case_dir: str | Path, pattern: str) -> str | None:
    expression = re.compile(pattern)
    for path in _candidate_u_paths(case_dir):
        if not path.exists():
            continue
        inlet_block = _extract_inlet_block(path.read_text())
        match = expression.search(inlet_block) if inlet_block is not None else None
        if match is not None:
            return match.group(1)
    return None


def infer_inlet_flow_rate(case_dir: str | Path) -> float | None:
    value = _infer_inlet_value(case_dir, r"volumetricFlowRate\s+(?:constant\s+)?([0-9eE+.\-]+)\s*;")
    return None if value is None else float(value)


def infer_inlet_drive_mode(case_dir: str | Path) -> str | None:
    inlet_type = _infer_inlet_value(case_dir, r"type\s+(\S+)\s*;")
    return None if inlet_type is None else (
        "inlet_flow_rate" if inlet_type == "flowRateInletVelocity" else "inlet_velocity"
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
            case_dir, f"case/constant/{region}/thermophysicalProperties",
            f"constant/{region}/thermophysicalProperties",
        )
        return None if path is None else _extract_first_scalar(
            path.read_text(), r"\belcond\s+(?:\[[^\]]*\])?\s*([0-9eE+.\-]+)\s*;"
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


def _decode_matched_b2_lmx_input(path: str | Path):
    from dataclasses import fields

    from ._fringing_types import ExtrudedInductionlessProblem, FringingProfile
    from .fringing import _cross_section_mesh
    from .specs import (
        BoundaryCondition, CaseSpec, GeometrySpec, MagneticFieldSpec, OutputSpec,
        RegionSpec, SolverConfig, TimeStepperConfig,
    )

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_top = {
        "schema_version", "kind", "case_id", "case", "scaling", "mesh",
        "field_profile", "effective_controls",
    }
    if not isinstance(payload, dict) or set(payload) != expected_top or (
        payload.get("schema_version"), payload.get("kind"), payload.get("case_id")
    ) != (1, "lmx-matched-b2-input", "B2-fringing-square"):
        raise ValueError("Invalid matched B2 LMX input schema")

    def checked(cls, value, name):
        if not isinstance(value, dict) or set(value) != {item.name for item in fields(cls)}:
            raise ValueError(f"Invalid matched B2 {name} schema")
        return dict(value)

    raw = checked(CaseSpec, payload["case"], "case")
    geometry = checked(GeometrySpec, raw.pop("geometry"), "geometry")
    geometry["wall_thickness"], geometry["wall_cells"] = (
        tuple(geometry["wall_thickness"]), tuple(geometry["wall_cells"])
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
        **raw, geometry=GeometrySpec(**geometry), regions=regions,
        magnetic_field=MagneticFieldSpec(**magnetic), boundary_conditions=tuple(boundaries),
        time_stepper=time_stepper, solver=solver, output=output,
    )
    if case.name != "alex_b2-fringing-square_harness-smoke" or case.geometry.kind != "layered_duct":
        raise ValueError("Matched B2 LMX input does not select the canonical solver path")
    mesh = _cross_section_mesh(case)
    mesh_payload = payload["mesh"]
    if not isinstance(mesh_payload, dict) or set(mesh_payload) != {
        "coordinate_system", "x_faces", "y_faces", "z_faces"
    } or mesh_payload["coordinate_system"] != "Cartesian x-y-z faces in duct-half-width units":
        raise ValueError("Invalid matched B2 mesh schema")
    if any(
        not np.array_equal(np.asarray(mesh_payload[f"{axis}_faces"], dtype=float), np.asarray(getattr(mesh, f"{axis}_faces")))
        for axis in "xyz"
    ):
        raise ValueError("Matched B2 stored mesh faces do not reproduce the case")

    profile = payload["field_profile"]
    profile_keys = {
        "axis", "interpolation", "extrapolation", "source_name", "source_sha256",
        "anchors_sha256", "anchor_x_over_L", "anchor_b_over_B0",
        "sample_x_over_L", "sample_b_over_B0",
    }
    if not isinstance(profile, dict) or set(profile) != profile_keys or (
        profile["axis"], profile["interpolation"], profile["extrapolation"]
    ) != ("y", "linear", "forbidden"):
        raise ValueError("Invalid matched B2 field-profile schema")
    anchors_x = np.asarray(profile["anchor_x_over_L"], dtype=float)
    anchors_b = np.asarray(profile["anchor_b_over_B0"], dtype=float)
    sample_x = np.asarray(mesh.x_centers, dtype=float)
    sample_b = np.asarray(profile["sample_b_over_B0"], dtype=float)
    encoded = json.dumps(
        {"x_over_L": anchors_x.tolist(), "b_over_B0": anchors_b.tolist()},
        sort_keys=True, separators=(",", ":"),
    ).encode()
    if (
        anchors_x.ndim != 1 or anchors_x.shape != anchors_b.shape or anchors_x.size < 2
        or np.any(~np.isfinite(anchors_x)) or np.any(~np.isfinite(anchors_b))
        or np.any(np.diff(anchors_x) <= 0.0) or np.any(np.diff(anchors_b) > 1.0e-12)
        or sample_x[0] < anchors_x[0] or sample_x[-1] > anchors_x[-1]
        or not np.array_equal(np.asarray(profile["sample_x_over_L"], dtype=float), sample_x)
        or not np.array_equal(sample_b, np.interp(sample_x, anchors_x, anchors_b))
        or hashlib.sha256(encoded).hexdigest() != profile["anchors_sha256"]
        or re.fullmatch(r"[0-9a-f]{64}", str(profile["source_sha256"])) is None
    ):
        raise ValueError("Matched B2 field samples do not reproduce their anchors and mesh")

    scaling, controls = payload["scaling"], payload["effective_controls"]
    if not isinstance(scaling, dict) or set(scaling) != {
        "length_scale", "half_width_m", "nondimensional_length", "velocity", "density", "conductivity"
    } or scaling["length_scale"] != "duct half-width":
        raise ValueError("Invalid matched B2 scaling schema")
    fluid = [region for region in regions if region.kind == "fluid"]
    wall = [region for region in regions if region.kind == "solid"]
    inlet = [bc for bc in boundaries if bc.kind == "inlet_flow_rate"]
    outlet = [bc for bc in boundaries if bc.kind == "outlet_pressure"]
    if len(fluid) != 1 or len(wall) != 1 or len(inlet) != 1 or len(outlet) != 1:
        raise ValueError("Matched B2 input requires one fluid, wall, and inlet-flow region")
    length, velocity = float(scaling["nondimensional_length"]), float(scaling["velocity"])
    field_vector = np.asarray(case.magnetic_field.value, dtype=float)
    base_b = float(np.linalg.norm(field_vector))
    mean_velocity = float(inlet[0].value) / (case.geometry.width * case.geometry.height)
    ha = hartmann_number(
        magnetic_field=base_b, length_scale=length, conductivity=float(fluid[0].conductivity),
        density=float(fluid[0].density), kinematic_viscosity=float(fluid[0].viscosity),
    )
    interaction = interaction_parameter(
        magnetic_field=base_b, length_scale=length, conductivity=float(fluid[0].conductivity),
        density=float(fluid[0].density), velocity=velocity,
    )
    reynolds = reynolds_number(
        velocity=velocity, length_scale=length, kinematic_viscosity=float(fluid[0].viscosity)
    )
    conductance = wall_conductance_ratio(
        wall_conductivity=wall[0].conductivity, wall_thickness=float(wall[0].wall_thickness),
        fluid_conductivity=fluid[0].conductivity, length_scale=length,
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
        "expected_stop_reason": "step_limit" if case.time_stepper.max_steps < 3 else "in_progress",
    }
    if controls != expected_controls or not math.isclose(
        float(case.time_stepper.dt), float(controls.get("dt", math.nan))
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


def observe_lmx_b2_contract(path: str | Path) -> dict[str, object]:
    """Derive the matched-B2 contract from a real LMX input, never its expected spec."""

    problem, mesh, payload = _decode_matched_b2_lmx_input(path)
    case, scaling, profile, controls = (
        problem.case, payload["scaling"], payload["field_profile"], payload["effective_controls"]
    )
    fluid = next(region for region in case.regions if region.kind == "fluid")
    wall = next(region for region in case.regions if region.kind == "solid")
    inlet = next(bc for bc in case.boundary_conditions if bc.kind == "inlet_flow_rate")
    length, velocity = float(scaling["nondimensional_length"]), float(scaling["velocity"])
    magnetic_field = float(np.linalg.norm(np.asarray(case.magnetic_field.value, dtype=float)))
    ha = hartmann_number(
        magnetic_field=magnetic_field, length_scale=length, conductivity=fluid.conductivity,
        density=fluid.density, kinematic_viscosity=fluid.viscosity,
    )
    interaction = interaction_parameter(
        magnetic_field=magnetic_field, length_scale=length, conductivity=fluid.conductivity,
        density=fluid.density, velocity=velocity,
    )
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
                velocity=velocity, length_scale=length, kinematic_viscosity=fluid.viscosity
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
                wall_conductivity=wall.conductivity, wall_thickness=wall.wall_thickness,
                fluid_conductivity=fluid.conductivity, length_scale=length,
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
        "observable": {
            "primary": "excess transverse pressure difference between published A/B taps",
            "tap_geometry": "top and side wall midpoints at each axial station",
            "signed_orientation": "side (+z) minus top (+y)",
        },
        "normalization": {
            "field": "B_y / B0",
            "pressure": "Delta p_AB / (sigma * U * B0^2 * half-width) minus plateau",
            "coordinate": "x / half-width",
        },
        "mesh_coordinates": {
            "coordinate_system": payload["mesh"]["coordinate_system"],
            "family": "uniform 5x5 fluid grid with one explicit wall cell per side",
            "exact_coordinate_arrays_required": True,
            **{f"{axis}_faces": np.asarray(getattr(mesh, f"{axis}_faces")).tolist() for axis in "xyz"},
            "field_source": profile["source_name"],
            "field_source_sha256": profile["source_sha256"],
            "field_anchors_sha256": profile["anchors_sha256"],
            "field_sample_x_over_L": profile["sample_x_over_L"],
            "field_sample_b_over_B0": profile["sample_b_over_B0"],
        },
        "stopping_rules": dict(controls),
    }
    return contract


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
