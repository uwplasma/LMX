#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lmx.benchmarks import build_benchmark_b_problem, load_benchmark_b_reference, load_benchmark_b_spec
from lmx.freemhd import (
    artifact_sha256,
    infer_inlet_drive_mode,
    infer_inlet_flow_rate,
    infer_liquid_material_properties,
    infer_rectangular_geometry,
    infer_solid_conductivities,
    infer_uniform_b0,
    load_benchmark_a_spec,
    load_matched_b2_lmx_input,
    observe_freemhd_b2_contract,
    observe_freemhd_b2_output,
    observe_lmx_b2_contract,
    observe_lmx_b2_output,
    validate_matched_b_record,
)
from lmx.reference_data import default_closed_channel_reference_root
from lmx.units import hartmann_number


DEFAULT_FREEMHD_INSTALL_DIR = Path("/Users/rogerio/local/tests/freemhd_install")
DEFAULT_FREEMHD_SOURCE_REPO = Path("/Users/rogerio/local/tests/lmx_external_codes/FreeMHD")
DEFAULT_PROCESSED_ROOT = default_closed_channel_reference_root()

_FREEMHD_SOURCE_NAMES = (
    "momentum", "electric", "limiter", "scheme_macro", "limiter_registration", "nvd", "vector_transform",
)


def audit_freemhd_case_against_spec(
    case_dir: str | Path,
    *,
    case_kind: str,
    spec_dir: str | Path | None = None,
) -> dict[str, object]:
    """Audit one Benchmark-A case without fitting its physical parameters."""

    def check(name: str, expected: object, observed: object) -> dict[str, object]:
        passed = (
            math.isclose(float(observed), float(expected), rel_tol=1.0e-9, abs_tol=1.0e-14)
            if isinstance(expected, (int, float)) and isinstance(observed, (int, float))
            else observed == expected
        )
        return {"name": name, "expected": expected, "observed": observed, "pass": passed}

    root = Path(case_dir)
    mesh = next((path for path in (root / "case/system/blockMeshDict", root / "system/blockMeshDict") if path.is_file()), None)
    declared_ha = None
    if mesh is not None and (match := re.search(r"\bHa\s+([0-9eE+.\-]+)\s*;", mesh.read_text())):
        declared_ha = float(match.group(1))
    spec = load_benchmark_a_spec(case_kind, spec_dir)
    geometry = infer_rectangular_geometry(root)
    fluid = infer_liquid_material_properties(root)
    b0 = infer_uniform_b0(root)
    solid, insulator = infer_solid_conductivities(root)
    checks: list[dict[str, object]] = []
    expected_geometry = spec["geometry"]
    if geometry is None:
        checks.append(check("geometry.available", True, False))
    else:
        width, height, thickness, wall_cells = geometry
        checks += [
            check("geometry.width", expected_geometry["width"], width),
            check("geometry.height", expected_geometry["height"], height),
            check("geometry.wall_thickness", expected_geometry["wall_thickness"], thickness),
            check("geometry.wall_cells", expected_geometry["wall_cells"], wall_cells),
        ]
    expected_fluid = spec["fluid"]
    if fluid is None:
        checks.append(check("fluid.available", True, False))
    else:
        checks += [check(f"fluid.{key}", expected_fluid[key], fluid[key]) for key in (
            "conductivity", "density", "dynamic_viscosity", "kinematic_viscosity"
        )]
    expected_field = tuple(map(float, spec["magnetic_field"]["vector"]))
    physical_ha = None
    if fluid is not None and b0 is not None and geometry is not None:
        physical_ha = hartmann_number(
            magnetic_field=math.sqrt(sum(value * value for value in b0)),
            length_scale=float(expected_geometry["length_scale"]),
            conductivity=fluid["conductivity"], density=fluid["density"],
            kinematic_viscosity=fluid["kinematic_viscosity"],
        )
    expected_wall = spec["wall"]
    checks += [
        check("magnetic_field.vector", expected_field, b0),
        check("mesh.declared_hartmann", spec["magnetic_field"]["hartmann_number"], declared_ha),
        check("physics.hartmann", spec["magnetic_field"]["hartmann_number"], physical_ha),
        check("wall.conducting_wall_conductivity", expected_wall["conducting_wall_conductivity"], solid),
        check("wall.insulating_wall_conductivity", expected_wall["insulating_wall_conductivity"], insulator),
        check("drive.mode", spec["drive"]["mode"], infer_inlet_drive_mode(root)),
        check("drive.target_flow_rate", spec["drive"].get("target_flow_rate"), infer_inlet_flow_rate(root)),
    ]
    failed = [item for item in checks if not item["pass"]]
    return {
        "case_kind": case_kind, "spec_id": spec["id"], "spec_path": spec["path"],
        "spec_sha256": spec["sha256"], "reference_case_dir": str(root),
        "matched": not failed, "failed_check_count": len(failed), "checks": checks,
        "physical_hartmann_number": physical_ha,
        "declared_mesh_hartmann_number": declared_ha,
    }


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


def materialize_lmx_source_snapshot(output_dir: str | Path) -> dict[str, object]:
    """Copy the clean tracked LMX package and parity driver used by the smoke."""

    repo, destination = Path(__file__).resolve().parents[1], Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing LMX source snapshot {destination}")
    scope = ("lmx", "pyproject.toml", "scripts/run_freemhd_parity_suite.py")
    status = subprocess.run(
        ("git", "-C", str(repo), "status", "--porcelain", "--", *scope),
        check=True, capture_output=True, text=True,
    ).stdout
    if status:
        raise ValueError("LMX source scope has staged or unstaged changes")
    files = subprocess.run(
        ("git", "-C", str(repo), "ls-files", "--", *scope),
        check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    destination.mkdir(parents=True)
    hashes = {}
    for relative in files:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / relative, target)
        hashes[relative] = artifact_sha256(target, "file")
    manifest = {
        "schema_version": 1,
        "commit": subprocess.run(
            ("git", "-C", str(repo), "rev-parse", "HEAD"),
            check=True, capture_output=True, text=True,
        ).stdout.strip(),
        "files": dict(sorted(hashes.items())),
    }
    _write_json(destination / "source-pin.json", manifest)
    return manifest


def _tiny_b2_problem(
    spec_root: str | Path | None = None,
    *,
    solver_shape: tuple[int, int, int] = (8, 7, 7),
) -> tuple[object, dict[str, object]]:
    from lmx._fringing_types import ExtrudedInductionlessProblem, FringingProfile
    from lmx.fringing import _cross_section_mesh

    problem = build_benchmark_b_problem("B2-fringing-square", mesh_level="coarse", root=spec_root)
    dt = 1.0 / 540000.0
    nx, ny, nz = solver_shape
    wall_cells = (1, 1, 1, 1)
    if nx < 2 or ny < 5 or nz < 5:
        raise ValueError("Matched B2 shape requires nx >= 2 and three transverse fluid cells")
    case = replace(
        problem.case,
        name="alex_b2-fringing-square_harness-smoke",
        geometry=replace(
            problem.case.geometry,
            nx=nx,
            ny=ny - sum(wall_cells[:2]),
            nz=nz - sum(wall_cells[2:]),
            wall_cells=wall_cells,
        ),
        time_stepper=replace(problem.case.time_stepper, dt=dt, t_final=2.0 * dt, max_steps=2),
    )
    mesh = _cross_section_mesh(case)
    generated_shape = tuple(len(getattr(mesh, f"{axis}_centers")) for axis in "xyz")
    if generated_shape != solver_shape:
        raise RuntimeError(f"Matched B2 mesh shape is {generated_shape}, expected {solver_shape}")
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
    output_file: str | Path,
    *,
    spec_root: str | Path | None = None,
    solver_shape: tuple[int, int, int] = (8, 7, 7),
) -> dict[str, object]:
    """Write a deterministic matched-B2 input; the default is the exact smoke."""

    destination = Path(output_file)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing LMX B2 input {destination}")
    _, payload = _tiny_b2_problem(spec_root, solver_shape=solver_shape)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def materialize_matched_b2_evaluator(output_file: str | Path) -> dict[str, object]:
    """Write the shared pressure-tap and normalization contract used by both solvers."""

    destination = Path(output_file)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing matched B2 evaluator {destination}")
    payload = {
        "schema_version": 1,
        "case_id": "B2-fringing-square",
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
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload


_B2_SKELETON_FILES = (
    "0/B0", "0/JxB", "0/T", "0/U", "0/alpha.liquidMetal", "0/p", "0/p_rgh", "0/potE",
    "constant/g", "constant/regionProperties", "constant/liquid/thermophysicalProperties.liquidMetal",
    "constant/solidWalls/thermophysicalProperties", "system/blockMeshDict", "system/controlDict",
    "system/liquid/changeDictionaryDict", "system/liquid/fvSchemes", "system/liquid/fvSolution",
    "system/solidWalls/changeDictionaryDict", "system/solidWalls/fvSchemes", "system/solidWalls/fvSolution",
)


def _foam_text(object_name: str, body: str, *, class_name: str = "dictionary", location: str = "") -> str:
    location_entry = f'    location    "{location}";\n' if location else ""
    return (
        "FoamFile\n{\n    version 2.0;\n    format ascii;\n"
        f"    class {class_name};\n{location_entry}    object {object_name};\n}}\n\n{body.strip()}\n"
    )


def _write_foam(path: Path, body: str, *, class_name: str = "dictionary") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        _foam_text(path.name, body, class_name=class_name),
        encoding="utf-8",
    )


def _b2_field_expression(anchor_count: int) -> str:
    labels = [chr(ord("a") + index) for index in range(anchor_count)]
    slopes = [f"(b{right}-b{left})/(x{right}-x{left})" for left, right in zip(labels, labels[1:])]
    terms = [f"b{labels[0]}", f"{slopes[0]}*(x-x{labels[0]})"]
    terms += [
        f"({slopes[index]}-{slopes[index - 1]})*pos(x-x{labels[index]})*(x-x{labels[index]})"
        for index in range(1, anchor_count - 1)
    ]
    return "+".join(terms)


def _b2_block_mesh() -> str:
    return """
scale 1;
xMin -15; xMax 10; Ly 1; Ly_wall 1.02; physicalHalfWidth 0.0439;
Nx 8; Ny 5; Nz 5; N_wall 1;
vertices
(
 ($xMin -$Ly -$Ly) ($xMax -$Ly -$Ly) ($xMax $Ly -$Ly) ($xMin $Ly -$Ly)
 ($xMin -$Ly $Ly) ($xMax -$Ly $Ly) ($xMax $Ly $Ly) ($xMin $Ly $Ly)
 ($xMin -$Ly -$Ly_wall) ($xMax -$Ly -$Ly_wall) ($xMax $Ly -$Ly_wall) ($xMin $Ly -$Ly_wall)
 ($xMin -$Ly_wall -$Ly_wall) ($xMax -$Ly_wall -$Ly_wall) ($xMax $Ly_wall -$Ly_wall) ($xMin $Ly_wall -$Ly_wall)
 ($xMin -$Ly_wall -$Ly) ($xMax -$Ly_wall -$Ly) ($xMax $Ly_wall -$Ly) ($xMin $Ly_wall -$Ly)
 ($xMin -$Ly $Ly_wall) ($xMax -$Ly $Ly_wall) ($xMax $Ly $Ly_wall) ($xMin $Ly $Ly_wall)
 ($xMin -$Ly_wall $Ly_wall) ($xMax -$Ly_wall $Ly_wall) ($xMax $Ly_wall $Ly_wall) ($xMin $Ly_wall $Ly_wall)
 ($xMin -$Ly_wall $Ly) ($xMax -$Ly_wall $Ly) ($xMax $Ly_wall $Ly) ($xMin $Ly_wall $Ly)
);
blocks
(
 hex (0 1 2 3 4 5 6 7) liquid ($Nx $Ny $Nz) simpleGrading (1 1 1)
 hex (12 13 9 8 16 17 1 0) solidWalls ($Nx $N_wall $N_wall) simpleGrading (1 1 1)
 hex (11 10 14 15 3 2 18 19) solidWalls ($Nx $N_wall $N_wall) simpleGrading (1 1 1)
 hex (8 9 10 11 0 1 2 3) solidWalls ($Nx $Ny $N_wall) simpleGrading (1 1 1)
 hex (28 29 5 4 24 25 21 20) solidWalls ($Nx $N_wall $N_wall) simpleGrading (1 1 1)
 hex (7 6 30 31 23 22 26 27) solidWalls ($Nx $N_wall $N_wall) simpleGrading (1 1 1)
 hex (4 5 6 7 20 21 22 23) solidWalls ($Nx $Ny $N_wall) simpleGrading (1 1 1)
 hex (16 17 1 0 28 29 5 4) solidWalls ($Nx $N_wall $Nz) simpleGrading (1 1 1)
 hex (3 2 18 19 7 6 30 31) solidWalls ($Nx $N_wall $Nz) simpleGrading (1 1 1)
);
edges ();
boundary
(
 inlet { type patch; faces ((0 4 7 3)); }
 sink { type patch; faces ((2 6 5 1)); }
 outerWalls
 {
  type wall;
  faces
  (
   (12 16 0 8) (8 0 3 11) (11 3 19 15) (3 7 31 19)
   (7 23 27 31) (4 20 23 7) (28 24 20 4) (16 28 4 0)
   (13 17 16 12) (17 29 28 16) (29 25 24 28) (25 21 20 24)
   (21 22 23 20) (22 26 27 23) (30 26 27 31) (18 30 31 19)
   (14 18 19 15) (10 14 15 11) (9 10 11 8) (13 9 8 12)
   (13 17 1 9) (9 1 2 10) (10 2 18 14) (17 29 5 1)
   (2 6 30 18) (29 25 21 5) (5 21 22 6) (6 22 26 30)
  );
 }
);
mergePatchPairs ();
"""


def _b2_set_expr(reference: dict[str, object]) -> str:
    x_values = [float(value) for value in reference["x_over_L"]]
    b_values = [float(value) for value in reference["b_over_B0"]]
    variables = ["\"x=pos().x()\"", "\"Bscale=sqrt(540)\""]
    variables += [f'"x{chr(97 + i)}={value:.17g}"' for i, value in enumerate(x_values)]
    variables += [f'"b{chr(97 + i)}={value:.17g}"' for i, value in enumerate(b_values)]
    anchors = {"x_over_L": x_values, "b_over_B0": b_values}
    anchors_sha = hashlib.sha256(
        json.dumps(anchors, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return f"""
lmxFieldSource \"alex-b2-square.csv\";
lmxFieldSourceSHA256 \"6778defc3a4b94c2d6f47b480b951d5d780a7f526cb0b661d7a424b02a83f483\";
lmxFieldAnchorsSHA256 \"{anchors_sha}\";
lmxInterpolation linear;
lmxExtrapolation forbidden;
expressions
(
 B0
 {{
  field B0;
  dimensions [1 0 -2 0 0 -1 0];
  variables ({' '.join(variables)});
  expression #{{ vector(0,Bscale*({_b2_field_expression(len(x_values))}),0) #}};
 }}
);
"""


def _b2_function_objects(sample_x: list[float]) -> str:
    """Return compact, replayable B2 pressure and boundary-flux observations."""

    points = "\n  ".join(
        f"({x:.17g} {y:.17g} {z:.17g})"
        for y, z in ((0.8, 0.0), (0.0, 0.8))
        for x in sample_x
    )

    def surface(name: str, patch: str, field: str, operation: str = "sum") -> str:
        return f"""{name}
 {{
  type surfaceFieldValue; libs (fieldFunctionObjects); region liquid;
  executeControl timeStep; executeInterval 1; writeControl timeStep; writeInterval 1;
  regionType patch; name {patch}; operation {operation}; fields ({field}); writeFields false;
 }}"""

    blocks = [
        f"""b2PressureTaps
 {{
  type probes; libs (sampling); region liquid;
  executeControl timeStep; executeInterval 1; writeControl timeStep; writeInterval 1;
  fixedLocations true; interpolationScheme cell; fields (p);
  probeLocations\n (\n  {points}\n );
 }}""",
        surface("massIn", "inlet", "rhoPhi"),
        surface("massOut", "sink", "rhoPhi"),
        surface("currentIn", "inlet", "jn"),
        surface("currentOut", "sink", "jn"),
        surface("currentIntoSolid", "liquid_to_solidWalls", "jn"),
        surface("currentIntoSolidMagnitude", "liquid_to_solidWalls", "jn", "sumMag"),
    ]
    return "functions\n{\n " + "\n ".join(blocks) + "\n}"


def materialize_matched_b2_freemhd_input(
    template_dir: str | Path,
    output_dir: str | Path,
    *,
    spec_root: str | Path | None = None,
) -> dict[str, object]:
    """Write a compact, deterministic two-step FreeMHD B2 input without generated data."""

    source, destination = Path(template_dir).resolve(), Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing FreeMHD B2 input {destination}")
    missing = [relative for relative in _B2_SKELETON_FILES if not (source / relative).is_file()]
    if missing:
        raise ValueError(f"FreeMHD hunt_demo skeleton is incomplete: {missing}")
    skeleton_hashes = {
        relative: artifact_sha256(source / relative, "file") for relative in _B2_SKELETON_FILES
    }
    destination.mkdir(parents=True)

    field_specs = {
        "B0": ("volVectorField", "[1 0 -2 0 0 -1 0]", "uniform (0 23.2379000772445 0)"),
        "JxB": ("volVectorField", "[1 -2 -2 0 0 0 0]", "uniform (0 0 0)"),
        "T": ("volScalarField", "[0 0 0 1 0 0 0]", "uniform 300"),
        "U": ("volVectorField", "[0 1 -1 0 0 0 0]", "uniform (1 0 0)"),
        "alpha.liquidMetal": ("volScalarField", "[0 0 0 0 0 0 0]", "uniform 1"),
        "p": ("volScalarField", "[1 -1 -2 0 0 0 0]", "uniform 0"),
        "p_rgh": ("volScalarField", "[1 -1 -2 0 0 0 0]", "uniform 0"),
        "potE": ("volScalarField", "[1 2 -3 0 0 -1 0]", "uniform 0"),
    }
    for name, (class_name, dimensions, internal) in field_specs.items():
        _write_foam(
            destination / "0" / name,
            f'dimensions {dimensions};\ninternalField {internal};\nboundaryField {{ ".*" {{ type calculated; value $internalField; }} }}',
            class_name=class_name,
        )

    mu = 540.0 / 2900.0**2
    _write_foam(destination / "constant/g", "dimensions [0 1 -2 0 0 0 0];\nvalue (0 0 0);", class_name="uniformDimensionedVectorField")
    _write_foam(destination / "constant/regionProperties", "regions ( fluid (liquid) solid (solidWalls) );")
    _write_foam(destination / "constant/liquid/fvOptions", "{}")
    _write_foam(destination / "constant/liquid/turbulenceProperties", "simulationType laminar;")
    _write_foam(destination / "constant/liquid/thermophysicalProperties", "phases (liquidMetal air);\npMin 0;\nsigma [1 0 -2 0 0 0 0] 0;")
    fluid_body = f"""
thermoType {{ type heRhoThermo; mixture pureMixture; transport const; thermo hConst; equationOfState rhoConst; specie specie; energy sensibleInternalEnergy; }}
mixture {{ specie {{ molWeight 1; }} equationOfState {{ rho 1; }} thermodynamics {{ Cp 1; Cv 1; Hf 0; }} transport {{ mu {mu:.17g}; Pr 1; }} }}
elcond [-1 -3 3 0 0 2 0] 1;
"""
    _write_foam(destination / "constant/liquid/thermophysicalProperties.liquidMetal", fluid_body)
    _write_foam(destination / "constant/liquid/thermophysicalProperties.air", fluid_body.replace("elcond [-1 -3 3 0 0 2 0] 1;", "elcond [-1 -3 3 0 0 2 0] 0;"))
    solid_body = """
thermoType { type heSolidThermo; mixture pureMixture; transport constIso; thermo hConst; equationOfState rhoConst; specie specie; energy sensibleEnthalpy; }
mixture { specie { molWeight 1; } transport { kappa 1; } thermodynamics { Hf 0; Cp 1; } equationOfState { rho 1; } }
elcond 3.5;
"""
    _write_foam(destination / "constant/solidWalls/thermophysicalProperties", solid_body)

    reference = load_benchmark_b_reference("B2-fringing-square", spec_root)
    sample_x = np.linspace(-15.0 + 25.0 / 16.0, 10.0 - 25.0 / 16.0, 8).tolist()
    dt = 1.0 / 540000.0
    control = f"""
application epotMultiRegionInterFoam; startFrom startTime; startTime 0; stopAt endTime;
endTime {2 * dt:.17g}; deltaT {dt:.17g}; adjustTimeStep off; maxCo 0.4; maxAlphaCo 0.3; maxDeltaT {dt:.17g};
writeControl timeStep; writeInterval 2; purgeWrite 0; writeFormat ascii; writePrecision 16; timeFormat general; timePrecision 16;
runTimeModifiable false; BtStartTime 0; BtDuration 0; JConservativeForm true;
lmxSteadyStepsRequired 3; {_b2_function_objects(sample_x)}
"""
    _write_foam(destination / "system/controlDict", control)
    _write_foam(destination / "system/blockMeshDict", _b2_block_mesh())
    _write_foam(destination / "system/fvSchemes", "ddtSchemes {} gradSchemes {} divSchemes {} laplacianSchemes {} interpolationSchemes {} snGradSchemes {}")
    _write_foam(destination / "system/fvSolution", "PIMPLE { nOuterCorrectors 1; }")
    for region in ("", "liquid", "solidWalls"):
        _write_foam(destination / "system" / region / "decomposeParDict", "numberOfSubdomains 2;\nmethod scotch;")

    liquid_schemes = """
ddtSchemes { default Euler; }
gradSchemes { default cellLimited leastSquares 1.0; }
divSchemes { default Gauss linear; div(rhoPhi,U) Gauss limitedLinear 1.0; div(phi,alpha) Gauss vanLeer; div(phirb,alpha) Gauss interfaceCompression; div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear; }
laplacianSchemes { default Gauss linear uncorrected; }
interpolationSchemes { default linear; }
snGradSchemes { default uncorrected; }
"""
    _write_foam(destination / "system/liquid/fvSchemes", liquid_schemes)
    liquid_solution = """
solvers {
 "alpha.liquidMetal.*" { nAlphaCorr 1; nAlphaSubCycles 1; cAlpha 1; solver PBiCG; preconditioner DILU; tolerance 1e-12; relTol 0; }
 p_rgh { solver PCG; preconditioner DIC; tolerance 1e-12; relTol 0; maxIter 4000; }
 p_rghFinal { $p_rgh; }
 "(U).*" { solver PBiCG; preconditioner DILU; tolerance 1e-10; relTol 0; maxIter 400; }
 "(h|T).*" { solver PBiCG; preconditioner DILU; tolerance 1e-12; relTol 0; maxIter 0; }
 potE { solver PCG; preconditioner DIC; tolerance 1e-12; relTol 0; maxIter 600; }
 potEFinal { $potE; }
}
PIMPLE { correctPhi yes; momentumPredictor yes; nCorrectors 1; nOuterCorrectors 1; nNonOrthogonalCorrectors 0; }
potentialFlow { nNonOrthogonalCorrectors 0; PhiRefCell 0; PhiRefValue 0; }
potE { nCorrectors 0; nNonOrthogonalCorrectors 0; PotERefCell 0; PotERefValue 0; }
"""
    _write_foam(destination / "system/liquid/fvSolution", liquid_solution)
    solid_schemes = """
ddtSchemes { default Euler; } gradSchemes { default Gauss linear; } divSchemes { default Gauss linear; }
laplacianSchemes { default Gauss linear corrected; } interpolationSchemes { default linear; } snGradSchemes { default corrected; }
"""
    _write_foam(destination / "system/solidWalls/fvSchemes", solid_schemes)
    solid_solution = """
solvers { "(h|T).*" { solver PBiCG; preconditioner DILU; tolerance 1e-12; relTol 0; maxIter 0; } potE { solver PCG; preconditioner DIC; tolerance 1e-12; relTol 0; maxIter 600; } potEFinal { $potE; } }
PIMPLE { nNonOrthogonalCorrectors 0; } potE { nCorrectors 0; nNonOrthogonalCorrectors 0; PotERefCell 0; PotERefValue 0; }
"""
    _write_foam(destination / "system/solidWalls/fvSolution", solid_solution)

    liquid_change = """
alpha.liquidMetal { internalField uniform 1; boundaryField { inlet { type fixedValue; value uniform 1; } ".*" { type zeroGradient; } } }
U { internalField uniform (1 0 0); boundaryField { inlet { type flowRateInletVelocity; volumetricFlowRate 4; extrapolateProfile yes; value uniform (1 0 0); } sink { type zeroGradient; value uniform (1 0 0); } ".*" { type noSlip; } } }
T { internalField uniform 300; boundaryField { ".*" { type fixedValue; value uniform 300; } "liquid_to_.*" { type compressible::turbulentTemperatureCoupledBaffleMixed; Tnbr T; kappaMethod fluidThermo; value uniform 300; } inlet { type fixedValue; value uniform 300; } sink { type fixedValue; value uniform 300; } } }
p_rgh { internalField uniform 0; boundaryField { sink { type fixedValue; value uniform 0; } inlet { type zeroGradient; } ".*" { type fixedFluxPressure; value uniform 0; } } }
p { internalField uniform 0; boundaryField { ".*" { type calculated; value uniform 0; } } }
B0 { internalField uniform (0 23.2379000772445 0); boundaryField { ".*" { type zeroGradient; value $internalField; } } }
potE { internalField uniform 0; boundaryField { inlet { type zeroGradient; } sink { type zeroGradient; } "liquid_to_.*" { type compressible::turbulentTemperatureCoupledBaffleMixed; Tnbr potE; kappaMethod lookup; kappa elcond; kappaName elcond; value uniform 0; } } }
"""
    _write_foam(destination / "system/liquid/changeDictionaryDict", liquid_change)
    solid_change = """
T { internalField uniform 300; boundaryField { outerWalls { type fixedValue; value uniform 300; } "solidWalls_to_.*" { type compressible::turbulentTemperatureCoupledBaffleMixed; Tnbr T; kappaMethod solidThermo; value uniform 300; } } }
B0 { internalField uniform (0 23.2379000772445 0); boundaryField { ".*" { type zeroGradient; value $internalField; } } }
potE { internalField uniform 0; boundaryField { outerWalls { type zeroGradient; value uniform 0; } "solidWalls_to_.*" { type compressible::turbulentTemperatureCoupledBaffleMixed; Tnbr potE; kappaMethod lookup; kappa elcond; kappaName elcond; value uniform 0; } } }
"""
    _write_foam(destination / "system/solidWalls/changeDictionaryDict", solid_change)
    field_dict = _b2_set_expr(reference)
    for region in ("liquid", "solidWalls"):
        _write_foam(destination / "system" / region / "setExprFieldsDict", field_dict)

    manifest: dict[str, object] = {
        "schema_version": 1,
        "kind": "freemhd-matched-b2-input",
        "case_id": "B2-fringing-square",
        "source_template": source.name,
        "source_skeleton_sha256": dict(sorted(skeleton_hashes.items())),
        "excluded_generated_data": True,
        "case_tree_sha256_before_manifest": artifact_sha256(destination, "tree"),
    }
    (destination / "lmx-benchmark-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def materialize_matched_b2_preflight(
    template_dir: str | Path,
    source_repo: str | Path,
    output_dir: str | Path,
) -> dict[str, object]:
    """Materialize and independently observe the complete solver-free B2 bundle."""

    from lmx.freemhd import observe_freemhd_b2_contract, observe_lmx_b2_contract

    destination = Path(output_dir)
    destination.mkdir(parents=True)
    paths = {
        "freemhd_input": destination / "freemhd_input",
        "freemhd_source": destination / "freemhd_source",
        "lmx_input": destination / "lmx_input.json",
        "evaluator": destination / "evaluator.json",
    }
    materialize_matched_b2_freemhd_input(template_dir, paths["freemhd_input"])
    materialize_freemhd_source_snapshot(source_repo, paths["freemhd_source"])
    materialize_matched_b2_lmx_input(paths["lmx_input"])
    materialize_matched_b2_evaluator(paths["evaluator"])
    lmx_contract = observe_lmx_b2_contract(paths["lmx_input"], paths["evaluator"])
    freemhd_contract = observe_freemhd_b2_contract(
        paths["freemhd_input"], paths["freemhd_source"], paths["evaluator"]
    )
    if lmx_contract != freemhd_contract:
        raise ValueError("Independent matched-B2 input observers disagree")
    encoded = json.dumps(lmx_contract, sort_keys=True, separators=(",", ":")).encode()
    artifacts = {}
    for name, path in paths.items():
        kind = "tree" if path.is_dir() else "file"
        artifacts[name] = {"kind": kind, "sha256": artifact_sha256(path, kind)}
    summary: dict[str, object] = {
        "schema_version": 1,
        "case_id": "B2-fringing-square",
        "status": "preflight-pass",
        "contract_sha256": hashlib.sha256(encoded).hexdigest(),
        "artifacts": artifacts,
    }
    _write_json(destination / "preflight.json", summary)
    return summary


def _run_matched_b2_lmx_direct(problem, *, num_devices: int):
    """Run the exact two-step path and return its step-one checkpoint."""

    import jax
    from lmx.fringing import solve_extruded_inductionless

    progress = []
    direct = solve_extruded_inductionless(
        problem,
        num_devices=num_devices,
        progress_callback=progress.append,
        checkpoint_interval=1,
    )
    if len(progress) != 2 or progress[0].checkpoint is None:
        raise ValueError("LMX B2 direct path did not emit its step-one checkpoint")
    jax.block_until_ready((direct.bundle.u, direct.bundle.p, direct.bundle.phi))
    if direct.bundle.u.dtype != np.float64:
        raise ValueError("LMX B2 smoke requires float64 execution")
    return progress[0].checkpoint, direct.bundle


def _resume_matched_b2_lmx(problem, checkpoint, *, num_devices: int):
    """Replay the second update from an exact matched-B2 checkpoint."""

    import jax
    from lmx.fringing import solve_extruded_inductionless

    replay = replace(problem, case=replace(problem.case, time_stepper=replace(
        problem.case.time_stepper, t_final=problem.case.time_stepper.dt, max_steps=1)))
    resumed = solve_extruded_inductionless(
        replay, initial_bundle=checkpoint, num_devices=num_devices
    ).bundle
    jax.block_until_ready((resumed.u, resumed.p, resumed.phi))
    return resumed


def _write_matched_b2_lmx_output(
    input_path, evaluator, output_dir, bundles, *, num_devices: int, wall_seconds: float
):
    """Serialize already-executed matched-B2 bundles outside timing regions."""

    from lmx.io import write_extruded_bundle_restart_npz

    destination = Path(output_dir)
    destination.mkdir(parents=True)
    problem = load_matched_b2_lmx_input(input_path)
    for name, bundle in (
        ("checkpoint.npz", bundles[0]),
        ("direct.npz", bundles[1]),
        ("resumed.npz", bundles[2]),
    ):
        write_extruded_bundle_restart_npz(bundle, problem.case, destination / name)
    _write_json(
        destination / "run.json",
        {
            "schema_version": 1, "code": "LMX", "case_id": "B2-fringing-square",
            "input_sha256": artifact_sha256(input_path, "file"),
            "evaluator_sha256": artifact_sha256(evaluator, "file"),
            "wall_seconds": wall_seconds,
            "num_devices": num_devices, "float_precision": "float64",
        },
    )


def run_matched_b2_lmx_smoke(
    input_path: str | Path,
    evaluator: str | Path,
    output_dir: str | Path,
    *,
    num_devices: int = 1,
) -> dict[str, object]:
    """Run direct and checkpoint-resumed LMX paths and write replayable evidence."""

    problem = load_matched_b2_lmx_input(input_path)
    started = time.perf_counter()
    checkpoint, direct = _run_matched_b2_lmx_direct(
        problem, num_devices=num_devices
    )
    resumed = _resume_matched_b2_lmx(
        problem, checkpoint, num_devices=num_devices
    )
    wall_seconds = time.perf_counter() - started
    _write_matched_b2_lmx_output(
        input_path, evaluator, output_dir, (checkpoint, direct, resumed),
        num_devices=num_devices, wall_seconds=wall_seconds,
    )
    destination = Path(output_dir)
    return observe_lmx_b2_output(destination, input_path, evaluator)


def run_matched_b2_freemhd_smoke(
    input_dir: str | Path,
    evaluator: str | Path,
    output_dir: str | Path,
    *,
    image: str = "freemhd-install:latest",
    nproc: int = 2,
    timeout_seconds: float = 300.0,
) -> dict[str, object]:
    """Run the exact FreeMHD smoke without reconstruction, VTK, or plotting."""

    source, destination = Path(input_dir).resolve(), Path(output_dir).resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing FreeMHD output {destination}")
    if nproc < 1 or timeout_seconds <= 0.0 or not image.strip():
        raise ValueError("FreeMHD smoke resources are invalid")
    input_sha = artifact_sha256(source, "tree")
    destination.mkdir(parents=True)
    cidfile = destination.parent / f".{destination.name}.cid"
    container = f"lmx-b2-{os.getpid()}-{time.time_ns()}"
    objects = "b2PressureTaps massIn massOut currentIn currentOut currentIntoSolid currentIntoSolidMagnitude"
    shell = f"""
source /usr/lib/openfoam/openfoam2206/etc/bashrc
set -euo pipefail
work=/tmp/lmx-b2-case
rm -rf "$work" && mkdir -p "$work"
rsync -a /input/ "$work/" && cd "$work"
blockMesh -fileHandler collated
splitMeshRegions -cellZonesOnly -overwrite -fileHandler collated
for region in liquid solidWalls; do
  changeDictionary -region "$region" -fileHandler collated
  setExprFields -region "$region" -fileHandler collated
done
decomposePar -allRegions -force -fileHandler collated
cp system/controlDict /output/controlDict.used
export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
mpirun -np {nproc} epotMultiRegionInterFoam -parallel 2>&1 | tee /output/run.log
mkdir /output/postProcessing
for name in {objects}; do
  path="$(find postProcessing -type d -name "$name" -print -quit)"
  test -n "$path" && cp -a "$path" "/output/postProcessing/$name"
done
"""
    command = [
        "docker", "run", "--rm", "--name", container, "--cidfile", str(cidfile),
        "--mount", f"type=bind,src={source},dst=/input,readonly",
        "--mount", f"type=bind,src={destination},dst=/output",
        "--entrypoint", "/bin/bash", image, "-lc", shell,
    ]
    started = time.perf_counter()
    try:
        try:
            subprocess.run(command, check=True, timeout=timeout_seconds, text=True, capture_output=True)
        except subprocess.TimeoutExpired as error:
            raise TimeoutError(f"FreeMHD B2 smoke exceeded {timeout_seconds:g} seconds") from error
        except subprocess.CalledProcessError as error:
            output = "\n".join(part.strip() for part in (error.stdout, error.stderr) if part)
            raise RuntimeError(
                f"FreeMHD B2 smoke exited {error.returncode}:\n{output[-4000:]}"
            ) from error
    finally:
        try:
            subprocess.run(["docker", "rm", "-f", container], check=False, capture_output=True, text=True)
        except OSError:
            pass
        cidfile.unlink(missing_ok=True)
    if artifact_sha256(source, "tree") != input_sha:
        raise ValueError("FreeMHD B2 input changed during execution")
    _write_json(destination / "run.json", {
        "schema_version": 1, "code": "FreeMHD", "case_id": "B2-fringing-square",
        "input_sha256": input_sha, "evaluator_sha256": artifact_sha256(evaluator, "file"),
        "wall_seconds": time.perf_counter() - started, "nproc": nproc,
        "image": image, "float_precision": "float64",
    })
    return observe_freemhd_b2_output(destination, source, evaluator)


def run_matched_b2_smoke_bundle(
    template_dir: str | Path,
    source_repo: str | Path,
    output_dir: str | Path,
    *,
    image: str = "freemhd-install:latest",
    nproc: int = 2,
    total_timeout_seconds: float = 600.0,
) -> dict[str, object]:
    """Run LMX then FreeMHD inside one budget and validate the independent record."""

    destination = Path(output_dir)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite existing matched B2 bundle {destination}")
    if total_timeout_seconds <= 0.0:
        raise ValueError("Matched B2 smoke budget must be positive")
    started = time.perf_counter()
    materialize_matched_b2_preflight(template_dir, source_repo, destination)
    materialize_lmx_source_snapshot(destination / "lmx_source")
    paths = {
        "lmx_source": destination / "lmx_source",
        "freemhd_source": destination / "freemhd_source",
        "lmx_input": destination / "lmx_input.json",
        "freemhd_input": destination / "freemhd_input",
        "evaluator": destination / "evaluator.json",
        "lmx_output": destination / "lmx_output",
        "freemhd_output": destination / "freemhd_output",
    }
    lmx = run_matched_b2_lmx_smoke(paths["lmx_input"], paths["evaluator"], paths["lmx_output"])
    limits = load_benchmark_b_spec("B2-fringing-square")["harness_smoke_execution"]
    lmx_failed = (
        lmx["steps"] != 2 or lmx["stop_reason"] != "step_limit"
        or any(abs(value - 1.0 / 540000.0) > limits["dt_absolute_tolerance"] for value in lmx["dt"])
        or max(lmx["courant_max"]) > limits["courant_max"]
        or any(lmx[name] > limits[f"{name}_max"] for name in (
            "mass_balance", "current_balance", "interface_current_balance"
        ))
        or lmx["interface_current_activity"] < limits["interface_current_activity_min"]
        or lmx["restart_max_abs"] > limits["restart_absolute_tolerance"]
    )
    if lmx_failed:
        raise ValueError("LMX B2 smoke failed its frozen execution gate; FreeMHD was not started")
    remaining = total_timeout_seconds - (time.perf_counter() - started)
    if remaining <= 0.0:
        raise TimeoutError("LMX B2 smoke exhausted the shared execution budget")
    run_matched_b2_freemhd_smoke(
        paths["freemhd_input"], paths["evaluator"], paths["freemhd_output"],
        image=image, nproc=nproc, timeout_seconds=remaining,
    )
    artifacts = {}
    for name, path in paths.items():
        kind = "tree" if path.is_dir() else "file"
        artifacts[name] = {
            "path": path.relative_to(destination).as_posix(),
            "kind": kind,
            "sha256": artifact_sha256(path, kind),
        }
    spec_path = Path(__file__).resolve().parents[1] / "lmx/data/benchmarks/specs/alex-b2-square.toml"
    record = {
        "schema_version": 3, "case_id": "B2-fringing-square", "acceptance_role": "harness-smoke",
        "contract": {
            "lmx": observe_lmx_b2_contract(paths["lmx_input"], paths["evaluator"]),
            "freemhd": observe_freemhd_b2_contract(
                paths["freemhd_input"], paths["freemhd_source"], paths["evaluator"]
            ),
        },
        "comparison": {"source": "independent-output-observers"},
        "provenance": {
            "benchmark_spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
            "artifacts": artifacts,
        },
    }
    _write_json(destination / "record.json", record)
    report = validate_matched_b_record(
        record, expected_case_id="B2-fringing-square", artifact_root=destination
    )
    _write_json(destination / "report.json", report)
    return report


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
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--materialize",
        choices=("shercliff", "hunt"),
        help="materialize an audited smoke case at --output and exit without running a solver",
    )
    mode.add_argument(
        "--matched-b2-preflight",
        action="store_true",
        help="materialize and observe the matched-B2 inputs at --output without running a solver",
    )
    mode.add_argument(
        "--matched-b2-smoke",
        action="store_true",
        help="run and independently validate the exact two-update LMX/FreeMHD smoke",
    )
    parser.add_argument("--freemhd-image", default=os.environ.get("LMX_FREEMHD_IMAGE", "freemhd-install:latest"))
    parser.add_argument("--nproc", type=int, default=2)
    parser.add_argument("--smoke-timeout", type=float, default=600.0)
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
    parser.add_argument(
        "--freemhd-source-repo",
        type=Path,
        default=Path(os.environ.get("LMX_FREEMHD_SOURCE_REPO", DEFAULT_FREEMHD_SOURCE_REPO)),
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
    if args.matched_b2_preflight:
        summary = materialize_matched_b2_preflight(
            args.freemhd_install_dir / "cases/hunt_demo",
            args.freemhd_source_repo,
            args.output,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.matched_b2_smoke:
        report = run_matched_b2_smoke_bundle(
            args.freemhd_install_dir / "cases/hunt_demo",
            args.freemhd_source_repo,
            args.output,
            image=args.freemhd_image,
            nproc=args.nproc,
            total_timeout_seconds=args.smoke_timeout,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0 if report.get("execution_pass") and report.get("comparison_pass") else 2

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
