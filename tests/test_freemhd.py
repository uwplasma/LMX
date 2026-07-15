from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

import examples
import lmx.benchmarks as benchmarks
from lmx._fringing_types import ExtrudedFieldBundle
from lmx.benchmarks import (
    BENCHMARK_B_SPEC_FILES,
    canonical_matched_b_contract,
    load_benchmark_b_reference,
    load_benchmark_b_spec,
)
from lmx.freemhd import (
    artifact_sha256,
    infer_inlet_drive_mode,
    infer_inlet_flow_rate,
    infer_liquid_material_properties,
    infer_rectangular_geometry,
    infer_solid_conductivities,
    infer_uniform_b0,
    load_matched_b2_lmx_input,
    load_benchmark_a_spec,
    load_samper_table_i,
    observe_freemhd_b2_contract,
    observe_freemhd_b2_output,
    observe_lmx_b2_contract,
    observe_lmx_b2_output,
    validate_matched_b_record,
)
from lmx.io import write_extruded_bundle_restart_npz
from scripts import run_freemhd_parity_suite


pytestmark = pytest.mark.unit


def _test_artifact_sha256(path: Path, kind: str) -> str:
    if kind == "file":
        return hashlib.sha256(path.read_bytes()).hexdigest()
    digest = hashlib.sha256(b"LMX-ARTIFACT-TREE-v1\0")
    for child in sorted(path.rglob("*"), key=lambda item: item.relative_to(path).as_posix()):
        entry_kind = b"d" if child.is_dir() else b"f"
        relative = child.relative_to(path).as_posix().encode()
        for value in (entry_kind, relative):
            digest.update(len(value).to_bytes(8, "big") + value)
        if entry_kind == b"f":
            value = hashlib.sha256(child.read_bytes()).digest()
            digest.update(len(value).to_bytes(8, "big") + value)
    return digest.hexdigest()


def _matched_b_record(root: Path, case_id: str, *, role: str | None = None) -> dict[str, object]:
    role = role or ("b1-production" if case_id.startswith("B1") else "b2-production")
    manifest = canonical_matched_b_contract(load_benchmark_b_spec(case_id), role)
    reference = load_benchmark_b_reference(case_id)
    spec_path = Path("benchmarks/specs") / BENCHMARK_B_SPEC_FILES[case_id]
    artifacts = {}
    for name in ("lmx_source", "freemhd_source", "lmx_input", "freemhd_input", "evaluator", "lmx_output", "freemhd_output"):
        kind = "tree" if name in {"lmx_source", "freemhd_source", "freemhd_input"} else "file"
        path = root / name
        if kind == "tree":
            path.mkdir(parents=True)
            (path / "evidence.txt").write_text(name)
        else:
            path.write_text(name)
        artifacts[name] = {"path": name, "kind": kind, "sha256": _test_artifact_sha256(path, kind)}
    return {
        "schema_version": 2,
        "case_id": case_id,
        "acceptance_role": role,
        "contract": {"lmx": deepcopy(manifest), "freemhd": deepcopy(manifest)},
        "comparison": {
            "x_over_L": list(reference["x_over_L"]),
            "lmx_observable": list(reference["pressure_observable"]),
            "freemhd_observable": list(reference["pressure_observable"]),
        },
        "provenance": {
            "benchmark_spec_sha256": hashlib.sha256(spec_path.read_bytes()).hexdigest(),
            "artifacts": artifacts,
        },
    }


def _write_u(root: Path, boundary: str) -> None:
    (root / "U").write_text(f"internalField uniform ( 0.2 0 0 );\nboundaryField {{ {boundary} }}\n")


def _write_reference_inputs(root: Path, *, b0: str = "internalField uniform ( 0 0.2 0 );\n") -> None:
    liquid = root / "constant/liquid"
    liquid.mkdir(parents=True)
    (liquid / "thermophysicalProperties.liquidMetal").write_text(
        "mixture { equationOfState { rho 1000; } transport { mu 0.001; } }\nelcond [-1 -3 3 0 0 2 0] 1e6;\n"
    )
    for region, conductivity in (("solidWalls", "5e6"), ("insulator", "1e-6")):
        directory = root / "constant" / region
        directory.mkdir(parents=True)
        (directory / "thermophysicalProperties").write_text(f"elcond {conductivity};\n")
    initial = root / "0/liquid"
    initial.mkdir(parents=True)
    (initial / "B0").write_text(b0)
    system = root / "system"
    system.mkdir()
    (system / "blockMeshDict").write_text("Ly 0.1;\nLy_wall 0.101;\nN_wall 2;\n")


def _write_demo_template(root: Path) -> None:
    liquid_zero = root / "0" / "liquid"
    liquid_zero.mkdir(parents=True)
    (liquid_zero / "B0").write_text("internalField uniform ( 0 10 0 );\n")
    (liquid_zero / "U").write_text(
        "internalField uniform ( 0.9725 0 0 );\n"
        "boundaryField { inlet { type flowRateInletVelocity; volumetricFlowRate 0.0389; "
        "value uniform ( 0.9725 0 0 ); } }\n"
    )
    for region in ("solidWalls", "insulator"):
        zero = root / "0" / region
        zero.mkdir(parents=True)
        (zero / "B0").write_text("internalField uniform ( 0 10 0 );\n")
        constant = root / "constant" / region
        constant.mkdir(parents=True)
        (constant / "thermophysicalProperties").write_text("elcond 1e-6;\n")
        system = root / "system" / region
        system.mkdir(parents=True)
        (system / "changeDictionaryDict").write_text("B0 { internalField uniform (0 10 0); value uniform (0 10 0); }\n")
    liquid_constant = root / "constant" / "liquid"
    liquid_constant.mkdir(parents=True)
    (liquid_constant / "thermophysicalProperties.liquidMetal").write_text("rho 1000;\nmu 1;\nelcond [-1 -3 3 0 0 2 0] 1e6;\n")
    liquid_system = root / "system" / "liquid"
    liquid_system.mkdir(parents=True)
    (liquid_system / "changeDictionaryDict").write_text(
        "U { internalField uniform ( 0.9725 0 0 ); volumetricFlowRate 0.0389; }\nB0 { internalField uniform (0 10 0); value uniform (0 10 0); }\n"
    )
    (root / "system" / "blockMeshDict").write_text("Ly 0.1;\nLy_wall 0.101;\nHa 20;\nN_wall 2;\n")


def _write_b2_skeleton(root: Path) -> None:
    for relative in run_freemhd_parity_suite._B2_SKELETON_FILES:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"skeleton {relative}\n")


def _write_observer_source_snapshot(root: Path) -> dict[str, object]:
    snippets = {
        "mhdUEqn.H": "fvm::ddt(rho, U) + fvm::div(rhoPhi, U); turbulence.divDevRhoReff(U);",
        "ePotEqn.H": "fvm::laplacian(elcond,potE); fvc::div(psiub); JConservativeForm;",
        "limitedLinear.H": "limitedLinearLimiter NVDTVD",
        "limitedLinear.C": "makeLimitedSurfaceInterpolationScheme(limitedLinear, limitedLinearLimiter)",
        "LimitedScheme.H": "makeLimitedSurfaceInterpolationTypeScheme(SS,LIMITER,NVDTVD,magSqr,vector)",
        "NVDTVD.H": "class NVDTVD {};",
        "LimitFuncs.C": "return Foam::magSqr(phi);",
    }
    files = {}
    for name, content in snippets.items():
        path = root / "src" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        files[f"src/{name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"commit": "fixture", "openfoam_release": "v2206", "files": files}
    (root / "source-pin.json").write_text(json.dumps(manifest, sort_keys=True))
    return manifest


def _write_lmx_b2_output(root: Path, input_path: Path, evaluator: Path) -> None:
    payload = json.loads(input_path.read_text())
    x = benchmarks.jnp.asarray(payload["field_profile"]["sample_x_over_L"])
    faces = benchmarks.jnp.asarray(payload["mesh"]["y_faces"])
    y = z = 0.5 * (faces[1:] + faces[:-1])
    shape, dt = (8, 7, 7), payload["effective_controls"]["dt"]

    def bundle(steps: int) -> ExtrudedFieldBundle:
        zeros = benchmarks.jnp.zeros(shape)
        return ExtrudedFieldBundle(
            x=x, y=y, z=z, field_scale=benchmarks.jnp.asarray(payload["field_profile"]["sample_b_over_B0"]),
            u=zeros, v=zeros, w=zeros, p=zeros, phi=zeros,
            geometry_kind="layered_duct", solver_kind="extruded_inductionless",
            rho_phi_plus=benchmarks.jnp.zeros((3, 8, 5, 5)), rho_phi_inlet=benchmarks.jnp.zeros((5, 5)),
            aitken_state=(benchmarks.jnp.zeros((4, *shape)), 1.0, 0),
            stopping_state=(steps, 0, "step_limit" if steps == 2 else "in_progress"),
            jx=benchmarks.jnp.ones(shape), jz=benchmarks.jnp.ones(shape),
            volumetric_flow_rate=benchmarks.jnp.full(8, 4.0),
            charge_balance_residual=benchmarks.jnp.full(8, 1.0e-5),
            boundary_current_residual=benchmarks.jnp.full(8, 1.0e-5),
            transverse_pressure_difference=benchmarks.jnp.asarray([0, 0, 1, 2, 3, 2, 0, 0]),
            iteration_residual_history=benchmarks.jnp.zeros(steps),
            iteration_component_residual_history=benchmarks.jnp.zeros((steps, 6)),
            iteration_pressure_residual_history=benchmarks.jnp.zeros(steps),
            iteration_electric_linear_history=benchmarks.jnp.zeros((steps, 6)),
            iteration_potential_residual_history=benchmarks.jnp.zeros(steps),
            iteration_courant_history=benchmarks.jnp.tile(benchmarks.jnp.asarray([dt, 1e-6, 2e-6]), (steps, 1)),
        )

    root.mkdir()
    case = load_matched_b2_lmx_input(input_path).case
    for name, value in (("checkpoint.npz", bundle(1)), ("direct.npz", bundle(2)), ("resumed.npz", bundle(2))):
        write_extruded_bundle_restart_npz(value, case, root / name)
    (root / "run.json").write_text(json.dumps({
        "schema_version": 1, "code": "LMX", "case_id": "B2-fringing-square",
        "input_sha256": artifact_sha256(input_path, "file"),
        "evaluator_sha256": artifact_sha256(evaluator, "file"),
        "wall_seconds": 1.0, "num_devices": 1, "float_precision": "float64",
    }))


def _write_freemhd_b2_output(root: Path, input_dir: Path, evaluator: Path) -> None:
    dt = 1.0 / 540000.0
    times = (dt, 2.0 * dt)
    root.mkdir()
    (root / "controlDict.used").write_bytes((input_dir / "system/controlDict").read_bytes())
    (root / "run.log").write_text(
        "Region: liquid Courant Number mean: 0 max: 0\n"
        + "".join(
            f"Time = {time:.17g}\nRegion: liquid Courant Number mean: 1e-6 max: 2e-6\n"
            "PCG: Solving for potE, Initial residual = 1e-4, Final residual = 1e-8, No Iterations 2\n"
            for time in times
        )
        + "End\n"
    )
    post = root / "postProcessing"
    x = np.linspace(-15.0 + 25.0 / 16.0, 10.0 - 25.0 / 16.0, 8)
    probes = post / "b2PressureTaps/0/p"
    probes.parent.mkdir(parents=True)
    headers = [f"# Probe {i} ({value:.15g} 0.8 0)" for i, value in enumerate(x)]
    headers += [f"# Probe {i + 8} ({value:.15g} 0 0.8)" for i, value in enumerate(x)]
    delta = np.asarray([0, 0, 1, 2, 3, 2, 0, 0], dtype=float)
    rows = [" ".join([f"{time:.17g}", *map(str, np.zeros(8)), *map(str, delta)]) for time in times]
    probes.write_text("\n".join([*headers, "# Time 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15", *rows]) + "\n")
    values = {
        "massIn": (-4.0, "sum(rhoPhi)"), "massOut": (4.0, "sum(rhoPhi)"),
        "currentIn": (-0.1, "sum(jn)"), "currentOut": (0.1, "sum(jn)"),
        "currentIntoSolid": (1.0e-5, "sum(jn)"),
        "currentIntoSolidMagnitude": (1.0, "sumMag(jn)"),
    }
    for name, (value, header) in values.items():
        path = post / name / "0/surfaceFieldValue.dat"
        path.parent.mkdir(parents=True)
        path.write_text(f"# Time {header}\n" + "".join(f"{time:.17g} {value}\n" for time in times))
    (root / "run.json").write_text(json.dumps({
        "schema_version": 1, "code": "FreeMHD", "case_id": "B2-fringing-square",
        "input_sha256": artifact_sha256(input_dir, "tree"),
        "evaluator_sha256": artifact_sha256(evaluator, "file"),
        "wall_seconds": 1.0, "nproc": 2, "image": "freemhd:test", "float_precision": "float64",
    }))


def _matched_b2_smoke_record(
    root: Path, monkeypatch: pytest.MonkeyPatch, *, executed: bool = False
) -> dict[str, object]:
    template, source = root / "hunt_demo", root / "freemhd_source"
    _write_b2_skeleton(template)
    manifest = _write_observer_source_snapshot(source)
    run_freemhd_parity_suite.materialize_matched_b2_freemhd_input(template, root / "freemhd_input")
    run_freemhd_parity_suite.materialize_matched_b2_lmx_input(root / "lmx_input")
    run_freemhd_parity_suite.materialize_matched_b2_evaluator(root / "evaluator")
    (root / "lmx_source").mkdir()
    (root / "lmx_source/evidence.txt").write_text("LMX source fixture\n")
    if executed:
        _write_lmx_b2_output(root / "lmx_output", root / "lmx_input", root / "evaluator")
        _write_freemhd_b2_output(root / "freemhd_output", root / "freemhd_input", root / "evaluator")
    else:
        for name in ("lmx_output", "freemhd_output"):
            (root / name).write_text("not executed\n")
    spec = deepcopy(load_benchmark_b_spec("B2-fringing-square"))
    reference = spec["free_mhd_discretization_reference"]
    reference["repository_commit"] = manifest["commit"]
    for name in run_freemhd_parity_suite._FREEMHD_SOURCE_NAMES:
        key = f"{name}_source"
        relative = f"src/{Path(reference[key]).name}"
        reference[key], reference[f"{key}_sha256"] = relative, manifest["files"][relative]
    monkeypatch.setattr(benchmarks, "load_benchmark_b_spec", lambda *_: spec)
    artifacts = {}
    for name in ("lmx_source", "freemhd_source", "lmx_input", "freemhd_input", "evaluator", "lmx_output", "freemhd_output"):
        kind = "tree" if name in {"lmx_source", "freemhd_source", "freemhd_input"} or executed and name.endswith("_output") else "file"
        artifacts[name] = {"path": name, "kind": kind, "sha256": artifact_sha256(root / name, kind)}
    contract = canonical_matched_b_contract(spec, "harness-smoke")
    return {
        "schema_version": 3 if executed else 2, "case_id": "B2-fringing-square", "acceptance_role": "harness-smoke",
        "contract": {"lmx": deepcopy(contract), "freemhd": deepcopy(contract)},
        "comparison": {"source": "independent-output-observers"} if executed else {"x_over_L": [], "lmx_observable": [], "freemhd_observable": []},
        "provenance": {
            "benchmark_spec_sha256": hashlib.sha256(Path("benchmarks/specs/alex-b2-square.toml").read_bytes()).hexdigest(),
            "artifacts": artifacts,
        },
    }


def _materialize_matched_case(root: Path, case_kind: str) -> Path:
    _write_demo_template(root / "template")
    run_freemhd_parity_suite.materialize_matched_freemhd_case(
        root / "template", root / "case", case_kind=case_kind
    )
    return root / "case"


@pytest.mark.parametrize("case_kind", ["shercliff", "hunt"])
def test_matched_benchmark_a_specs_are_dimensionally_consistent(case_kind: str):
    spec = load_benchmark_a_spec(case_kind)

    assert spec["magnetic_field"]["hartmann_number"] == pytest.approx(20.0)
    assert spec["fluid"]["kinematic_viscosity"] == pytest.approx(1.0e-3)
    assert spec["normalization"]["per_profile_peak_fitting"] is False
    assert len(spec["mesh"]["levels"]) == 4
    assert len(spec["sha256"]) == 64


def test_freemhd_case_audit_exposes_mislabeled_ha_and_hunt_wall(tmp_path: Path):
    case = _materialize_matched_case(tmp_path, "hunt")
    (case / "0/liquid/B0").write_text("internalField uniform ( 0 10 0 );\n")
    (case / "constant/solidWalls/thermophysicalProperties").write_text("elcond 1e-6;\n")
    report = run_freemhd_parity_suite.audit_freemhd_case_against_spec(case, case_kind="hunt")
    failed_names = {check["name"] for check in report["checks"] if not check["pass"]}

    assert report["matched"] is False
    assert report["physical_hartmann_number"] == pytest.approx(1000.0)
    assert "magnetic_field.vector" in failed_names
    assert "physics.hartmann" in failed_names
    assert "wall.conducting_wall_conductivity" in failed_names


@pytest.mark.parametrize("case_id", ["B1-fringing-pipe", "B2-fringing-square"])
def test_matched_b_schema2_verifies_contract_comparison_and_real_artifacts(tmp_path: Path, case_id: str):
    record = _matched_b_record(tmp_path, case_id)
    report = validate_matched_b_record(record, expected_case_id=case_id, artifact_root=tmp_path)

    assert report["schema_complete"] and report["artifact_pass"] and report["contract_pass"] and report["comparison_pass"]
    assert report["observation_pass"] is report["acceptance_pass"] is False
    assert set(report["calculated_artifact_sha256"]) == set(record["provenance"]["artifacts"])
    assert "contract.observers.unavailable" in report["failed_checks"]
    assert report["metrics"]["weighted_rms"] == pytest.approx(0.0)
    assert validate_matched_b_record(record, expected_case_id=case_id)["artifact_pass"] is False


def test_matched_b_schema2_rejects_contract_and_record_forgery(tmp_path: Path):
    record = _matched_b_record(tmp_path, "B2-fringing-square")
    mismatch = deepcopy(record)
    mismatch["contract"]["freemhd"]["equations"] = {"semantic_contract": "different"}
    mismatch["exact_case_match"] = mismatch["pass"] = True
    rejected = validate_matched_b_record(mismatch, expected_case_id="B2-fringing-square", artifact_root=tmp_path)
    assert "contract.equations.mismatch" in rejected["failed_checks"]
    assert "legacy.exact_case_match" in rejected["failed_checks"]

    mutations = (
        (("schema_version",), 1, "schema"),
        (("case_id",), "wrong-case", "case_id"),
        (("acceptance_role",), "self-promoted", "acceptance_role"),
        (("contract", "lmx", "wall"), {}, "contract.wall.missing"),
        (
            ("provenance", "benchmark_spec_sha256"),
            "b" * 64,
            "provenance.benchmark_spec_sha256.current",
        ),
        (("provenance", "artifacts"), {}, "provenance.artifacts"),
        (("comparison", "x_over_L"), [0.0], "comparison.arrays"),
    )
    for path, value, expected_check in mutations:
        malformed = deepcopy(record)
        target = malformed
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        report = validate_matched_b_record(malformed, expected_case_id="B2-fringing-square", artifact_root=tmp_path)
        assert expected_check in report["failed_checks"]

    for section, key, wrong in (
        ("equations", "inertia", "omitted"),
        ("boundary_drive", "flow_constraint_scope", "stationwise"),
    ):
        self_consistent = deepcopy(record)
        self_consistent["contract"]["lmx"][section][key] = wrong
        self_consistent["contract"]["freemhd"][section][key] = wrong
        rejected = validate_matched_b_record(self_consistent, expected_case_id="B2-fringing-square", artifact_root=tmp_path)
        assert f"contract.{section}.canonical" in rejected["failed_checks"]

    smoke = deepcopy(record)
    smoke["acceptance_role"] = "harness-smoke"
    report = validate_matched_b_record(smoke, expected_case_id="B2-fringing-square", artifact_root=tmp_path)
    assert report["contract_pass"] is report["role_allows_acceptance"] is report["acceptance_pass"] is False
    assert "contract.mesh_coordinates.canonical" in report["failed_checks"]

    poor = deepcopy(record)
    poor["comparison"]["freemhd_observable"] = [value + 1.0 for value in poor["comparison"]["freemhd_observable"]]
    report = validate_matched_b_record(poor, expected_case_id="B2-fringing-square", artifact_root=tmp_path)
    assert report["comparison_pass"] is report["acceptance_pass"] is False


@pytest.mark.parametrize(
    ("hazard", "check"),
    [
        ("file_mutation", "provenance.lmx_output.sha256.current"),
        ("tree_mutation", "provenance.lmx_source.sha256.current"),
        ("noncanonical", "provenance.lmx_input.path.noncanonical"),
        ("absolute", "provenance.lmx_input.path.absolute"),
        ("symlink", "provenance.evaluator.path.symlink"),
        ("tree_symlink", "provenance.lmx_source.tree.symlink"),
        ("missing", "provenance.lmx_input.path.missing_or_escape"),
        ("wrong_kind", "provenance.evaluator.kind"),
        ("overlap", "provenance.artifacts.overlap"),
        ("hardlink_tree", "provenance.lmx_source.tree.hardlink"),
        ("empty_tree", "provenance.lmx_source.content.empty"),
    ],
)
def test_matched_b_schema2_rejects_unsafe_or_changed_artifacts(tmp_path: Path, hazard: str, check: str):
    record = _matched_b_record(tmp_path, "B2-fringing-square")
    artifacts = record["provenance"]["artifacts"]
    if hazard == "file_mutation":
        (tmp_path / "lmx_output").write_text("changed")
    elif hazard == "tree_mutation":
        (tmp_path / "lmx_source" / "evidence.txt").write_text("changed")
    elif hazard == "noncanonical":
        artifacts["lmx_input"]["path"] = "../lmx_input"
    elif hazard == "absolute":
        artifacts["lmx_input"]["path"] = str((tmp_path / "lmx_input").resolve())
    elif hazard == "symlink":
        (tmp_path / "evaluator").unlink()
        (tmp_path / "evaluator").symlink_to("lmx_output")
    elif hazard == "tree_symlink":
        (tmp_path / "lmx_source" / "alias.txt").symlink_to("evidence.txt")
    elif hazard == "missing":
        artifacts["lmx_input"]["path"] = "missing"
    elif hazard == "wrong_kind":
        artifacts["evaluator"]["kind"] = "tree"
    elif hazard == "overlap":
        artifacts["freemhd_output"] = deepcopy(artifacts["lmx_output"])
    elif hazard == "hardlink_tree":
        (tmp_path / "lmx_source" / "alias.txt").hardlink_to(tmp_path / "lmx_source" / "evidence.txt")
    else:
        empty = tmp_path / "empty"
        empty.mkdir()
        artifacts["lmx_source"].update(path="empty", sha256="0" * 64)
    report = validate_matched_b_record(record, expected_case_id="B2-fringing-square", artifact_root=tmp_path)
    assert report["artifact_pass"] is False and check in report["failed_checks"]


def test_artifact_tree_hash_is_portable_and_content_sensitive(tmp_path: Path):
    left, right = tmp_path / "left", tmp_path / "right"
    for root, order in ((left, ("a", "b")), (right, ("b", "a"))):
        root.mkdir()
        for name in order:
            (root / name).write_text(name)
    expected = _test_artifact_sha256(left, "tree")
    assert artifact_sha256(left, "tree") == expected == artifact_sha256(right, "tree")
    (right / "empty").mkdir()
    assert artifact_sha256(right, "tree") != expected


@pytest.mark.parametrize("case_kind", ["shercliff", "hunt"])
def test_materialize_matched_freemhd_case_is_audited_and_refuses_overwrite(tmp_path: Path, case_kind: str):
    template = tmp_path / "template"
    output = tmp_path / "output"
    second_output = tmp_path / "second-output"
    _write_demo_template(template)

    manifest = run_freemhd_parity_suite.materialize_matched_freemhd_case(template, output, case_kind=case_kind)

    assert manifest["run_profile"] == "docker_smoke_only"
    assert manifest["audit"]["matched"] is True
    assert manifest["audit"]["physical_hartmann_number"] == pytest.approx(20.0)
    assert len(manifest["source_template_sha256"]) == 64
    assert (output / "lmx-benchmark-manifest.json").is_file()
    assert infer_uniform_b0(output) == pytest.approx((0.0, 0.2, 0.0))
    assert infer_inlet_drive_mode(output) == "inlet_flow_rate"
    assert infer_inlet_flow_rate(output) == pytest.approx(load_benchmark_a_spec(case_kind)["drive"]["target_flow_rate"])
    assert run_freemhd_parity_suite.materialize_matched_freemhd_case(template, second_output, case_kind=case_kind) == manifest
    assert (second_output / "lmx-benchmark-manifest.json").read_bytes() == (output / "lmx-benchmark-manifest.json").read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_freemhd_parity_suite.materialize_matched_freemhd_case(template, output, case_kind=case_kind)


def _frozen_source_repo(root: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, dict, Path]:
    repo = root / "FreeMHD"
    repo.mkdir()
    reference = {"openfoam_release": "v2206"}
    for index, name in enumerate(run_freemhd_parity_suite._FREEMHD_SOURCE_NAMES):
        source_key, relative = f"{name}_source", f"src/{name}.C"
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"source-{index}\n")
        reference[source_key] = relative
        reference[f"{source_key}_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (repo / "unrelated.txt").write_text("clean\n")
    for args in (("init", "-q"), ("config", "user.email", "test@example.com"), ("config", "user.name", "Test"), ("add", "."), ("commit", "-qm", "fixture")):
        subprocess.run(("git", "-C", str(repo), *args), check=True)
    reference["repository_commit"] = subprocess.check_output(
        ("git", "-C", str(repo), "rev-parse", "HEAD"), text=True
    ).strip()
    spec = {"free_mhd_discretization_reference": reference}
    monkeypatch.setattr(run_freemhd_parity_suite, "load_benchmark_b_spec", lambda case_id, spec_root=None: spec)
    return repo, reference, repo / reference["momentum_source"]


def test_freemhd_source_snapshot_is_deterministic_and_allows_unrelated_dirt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    repo, reference, _ = _frozen_source_repo(tmp_path, monkeypatch)
    (repo / "unrelated.txt").write_text("dirty but out of scope\n")
    first, second = tmp_path / "first", tmp_path / "second"

    manifest = run_freemhd_parity_suite.materialize_freemhd_source_snapshot(repo, first)
    assert manifest == json.loads((first / "source-pin.json").read_text())
    assert manifest == run_freemhd_parity_suite.materialize_freemhd_source_snapshot(repo, second)
    assert artifact_sha256(first, "tree") == artifact_sha256(second, "tree")
    assert len([path for path in first.rglob("*") if path.is_file()]) == 8
    assert set(manifest) == {"schema_version", "project", "commit", "openfoam_release", "files"}
    assert manifest["commit"] == reference["repository_commit"]
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_freemhd_parity_suite.materialize_freemhd_source_snapshot(repo, first)


@pytest.mark.parametrize("hazard", ["head", "unstaged", "staged", "hash", "missing", "symlink", "noncanonical"])
def test_freemhd_source_snapshot_rejects_unfrozen_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hazard: str
):
    repo, reference, path = _frozen_source_repo(tmp_path, monkeypatch)
    source_key, relative = "momentum_source", reference["momentum_source"]
    if hazard == "head":
        (repo / "new.txt").write_text("new commit\n")
        subprocess.run(("git", "-C", str(repo), "add", "new.txt"), check=True)
        subprocess.run(("git", "-C", str(repo), "commit", "-qm", "advance"), check=True)
    elif hazard in {"unstaged", "staged"}:
        path.write_text("changed\n")
        if hazard == "staged":
            subprocess.run(("git", "-C", str(repo), "add", relative), check=True)
    elif hazard == "hash":
        reference[f"{source_key}_sha256"] = "0" * 64
    elif hazard == "missing":
        path.unlink()
    elif hazard == "symlink":
        path.unlink()
        path.symlink_to(repo / "unrelated.txt")
    else:
        reference[source_key] = f"../{relative}"
    with pytest.raises(ValueError):
        run_freemhd_parity_suite.materialize_freemhd_source_snapshot(repo, tmp_path / "snapshot")


def test_matched_b2_lmx_input_is_deterministic_real_and_observed(tmp_path: Path):
    first, second = tmp_path / "first.json", tmp_path / "second.json"
    payload = run_freemhd_parity_suite.materialize_matched_b2_lmx_input(first)
    run_freemhd_parity_suite.materialize_matched_b2_lmx_input(second)
    problem, contract = load_matched_b2_lmx_input(first), observe_lmx_b2_contract(first)

    assert first.read_bytes() == second.read_bytes()
    assert payload["mesh"]["x_faces"] == [-15.0, -11.875, -8.75, -5.625, -2.5, 0.625, 3.75, 6.875, 10.0]
    assert payload["field_profile"]["sample_b_over_B0"] == [1.0, 1.0, 0.991875, 0.9371875, 0.69, 0.16125, 0.00875, 0.0]
    assert problem.case.name == "alex_b2-fringing-square_harness-smoke"
    groups = contract["nondimensional_groups"]
    assert {name: groups[name] for name in ("hartmann_number", "interaction_parameter", "reynolds_number")} == pytest.approx(
        {"hartmann_number": 2900.0, "interaction_parameter": 540.0, "reynolds_number": 2900.0**2 / 540.0}
    )
    assert groups["magnetic_reynolds_number_assumption"] == "Rm << 1"
    assert contract["stopping_rules"]["expected_stop_reason"] == "step_limit"
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_freemhd_parity_suite.materialize_matched_b2_lmx_input(first)


def test_matched_b2_freemhd_input_is_deterministic_slim_and_solver_free(tmp_path: Path):
    template, first, second = tmp_path / "hunt_demo", tmp_path / "first", tmp_path / "second"
    _write_b2_skeleton(template)
    (template / "0/cellToRegion").write_text("generated junk must not be copied\n")
    (template / "0/insulator").mkdir()
    manifest = run_freemhd_parity_suite.materialize_matched_b2_freemhd_input(template, first)

    assert manifest == run_freemhd_parity_suite.materialize_matched_b2_freemhd_input(template, second)
    assert artifact_sha256(first, "tree") == artifact_sha256(second, "tree")
    assert manifest["excluded_generated_data"] is True
    assert len([path for path in first.rglob("*") if path.is_file()]) == 32
    assert sum(path.stat().st_size for path in first.rglob("*") if path.is_file()) < 30_000
    assert not any("insulator" in path.parts or path.name == "cellToRegion" for path in first.rglob("*"))
    assert "hex (0 1 2 3 4 5 6 7) liquid ($Nx $Ny $Nz)" in (first / "system/blockMeshDict").read_text()
    assert "xa=-15" in (first / "system/liquid/setExprFieldsDict").read_text()
    functions = (first / "system/controlDict").read_text()
    assert functions.count("type probes;") == 1 and functions.count("type surfaceFieldValue;") == 6
    assert "currentIntoSolidMagnitude" in functions and "operation sumMag;" in functions
    assert "solidGradPotE" not in functions and "region solidWalls;" not in functions
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_freemhd_parity_suite.materialize_matched_b2_freemhd_input(template, first)


def test_matched_b2_freemhd_input_requires_complete_skeleton(tmp_path: Path):
    template = tmp_path / "hunt_demo"
    _write_b2_skeleton(template)
    (template / "system/blockMeshDict").unlink()
    with pytest.raises(ValueError, match="skeleton is incomplete"):
        run_freemhd_parity_suite.materialize_matched_b2_freemhd_input(template, tmp_path / "case")


def test_independent_matched_b2_input_observers_agree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    template, source = tmp_path / "hunt_demo", tmp_path / "source"
    _write_b2_skeleton(template)
    monkeypatch.setattr(
        run_freemhd_parity_suite,
        "materialize_freemhd_source_snapshot",
        lambda _, path: _write_observer_source_snapshot(Path(path)),
    )
    summary = run_freemhd_parity_suite.materialize_matched_b2_preflight(template, source, tmp_path / "bundle")
    case, source = tmp_path / "bundle/freemhd_input", tmp_path / "bundle/freemhd_source"
    lmx_input, evaluator = tmp_path / "bundle/lmx_input.json", tmp_path / "bundle/evaluator.json"

    assert summary["status"] == "preflight-pass"
    assert observe_freemhd_b2_contract(case, source, evaluator) == observe_lmx_b2_contract(lmx_input, evaluator)


def test_lmx_b2_output_observer_replays_restart_evidence(tmp_path: Path):
    input_path, evaluator = tmp_path / "lmx.json", tmp_path / "evaluator.json"
    run_freemhd_parity_suite.materialize_matched_b2_lmx_input(input_path)
    run_freemhd_parity_suite.materialize_matched_b2_evaluator(evaluator)
    _write_lmx_b2_output(tmp_path / "output", input_path, evaluator)

    observed = observe_lmx_b2_output(tmp_path / "output", input_path, evaluator)

    assert observed["steps"] == 2 and observed["stop_reason"] == "step_limit"
    assert observed["restart_max_abs"] == observed["mass_balance"] == 0.0
    assert observed["current_balance"] == observed["interface_current_balance"] == pytest.approx(1.0e-5)
    assert observed["pressure_observable"][4] == pytest.approx(3.0 / 540.0)


def test_freemhd_b2_output_observer_reads_only_native_tables(tmp_path: Path):
    template, case, evaluator = tmp_path / "template", tmp_path / "case", tmp_path / "evaluator"
    _write_b2_skeleton(template)
    run_freemhd_parity_suite.materialize_matched_b2_freemhd_input(template, case)
    run_freemhd_parity_suite.materialize_matched_b2_evaluator(evaluator)
    _write_freemhd_b2_output(tmp_path / "output", case, evaluator)

    observed = observe_freemhd_b2_output(tmp_path / "output", case, evaluator)

    assert observed["steps"] == 2 and observed["stop_reason"] == "step_limit"
    assert observed["mass_balance"] == observed["current_balance"] == 0.0
    assert observed["interface_current_balance"] == pytest.approx(1.0e-5)
    assert observed["pressure_observable"][4] == pytest.approx(3.0 / 540.0)
    assert observed["residual_max"] == {"potE": pytest.approx(1.0e-8)}


@pytest.mark.parametrize("mutation", ["fatal", "probe", "magnitude"])
def test_freemhd_b2_output_observer_rejects_corrupt_evidence(
    tmp_path: Path, mutation: str
):
    template, case, evaluator = tmp_path / "template", tmp_path / "case", tmp_path / "evaluator"
    _write_b2_skeleton(template)
    run_freemhd_parity_suite.materialize_matched_b2_freemhd_input(template, case)
    run_freemhd_parity_suite.materialize_matched_b2_evaluator(evaluator)
    output = tmp_path / "output"
    _write_freemhd_b2_output(output, case, evaluator)
    path, old, new = {
        "fatal": (output / "run.log", "End", "FOAM FATAL ERROR\nEnd"),
        "probe": (output / "postProcessing/b2PressureTaps/0/p", "0.8 0)", "0.7 0)"),
        "magnitude": (output / "postProcessing/currentIntoSolidMagnitude/0/surfaceFieldValue.dat", " 1.0", " -1.0"),
    }[mutation]
    path.write_text(path.read_text().replace(old, new, 1))

    with pytest.raises(ValueError, match="FreeMHD B2"):
        observe_freemhd_b2_output(output, case, evaluator)


def test_matched_b2_smoke_observation_passes_without_claiming_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    record = _matched_b2_smoke_record(tmp_path, monkeypatch)
    report = validate_matched_b_record(record, expected_case_id="B2-fringing-square", artifact_root=tmp_path)

    assert report["schema_complete"] and report["artifact_pass"] and report["contract_pass"]
    assert report["observation_pass"] is True
    assert report["comparison_pass"] is report["role_allows_acceptance"] is report["acceptance_pass"] is False
    assert report["failed_checks"] == ["comparison.arrays"]


def test_matched_b2_executed_smoke_passes_without_claiming_acceptance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    record = _matched_b2_smoke_record(tmp_path, monkeypatch, executed=True)
    report = validate_matched_b_record(record, expected_case_id="B2-fringing-square", artifact_root=tmp_path)

    assert report["schema_complete"] and report["artifact_pass"] and report["contract_pass"]
    assert report["observation_pass"] and report["execution_pass"] and report["comparison_pass"]
    assert report["role_allows_acceptance"] is report["acceptance_pass"] is False
    assert report["metrics"] == {"pressure_rms": 0.0, "pressure_linf": 0.0}
    assert report["failed_checks"] == []


@pytest.mark.parametrize(
    ("mutation", "failed"),
    [("mass", "execution.freemhd.mass_balance"), ("source", "comparison.source")],
)
def test_matched_b2_executed_smoke_attributes_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str, failed: str
):
    record = _matched_b2_smoke_record(tmp_path, monkeypatch, executed=True)
    if mutation == "mass":
        path = tmp_path / "freemhd_output/postProcessing/massOut/0/surfaceFieldValue.dat"
        path.write_text(path.read_text().replace(" 4.0", " 5.0"))
        record["provenance"]["artifacts"]["freemhd_output"]["sha256"] = artifact_sha256(
            tmp_path / "freemhd_output", "tree"
        )
    else:
        record["comparison"] = {"source": "claimed"}

    report = validate_matched_b_record(record, expected_case_id="B2-fringing-square", artifact_root=tmp_path)

    assert failed in report["failed_checks"]
    assert report.get("execution_pass", False) is False
    assert report["acceptance_pass"] is False


@pytest.mark.parametrize(
    ("mutation", "failed_path"),
    [
        ("mesh", "contract.geometry.half_width_m.freemhd_observed"),
        ("field", "contract.mesh_coordinates.field_anchors_sha256.freemhd_observed"),
        ("fluid", "contract.nondimensional_groups.hartmann_number.freemhd_observed"),
        ("wall", "contract.wall.wall_conductance_ratio.freemhd_observed"),
        ("velocity", "contract.boundary_drive.velocity_outlet.freemhd_observed"),
        ("pressure", "contract.boundary_drive.pressure_outlet_gauge.freemhd_observed"),
        ("electric", "contract.boundary_drive.electric_axial_ends.freemhd_observed"),
        ("scheme", "contract.equations.advection_discretization.freemhd_observed"),
        ("iterations", "contract.stopping_rules.electric_iterations.freemhd_observed"),
        ("source", "provenance.freemhd_source.pin"),
    ],
)
def test_matched_b2_smoke_attributes_one_sided_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str, failed_path: str
):
    record = _matched_b2_smoke_record(tmp_path, monkeypatch)
    lmx_before = observe_lmx_b2_contract(tmp_path / "lmx_input", tmp_path / "evaluator")
    case = tmp_path / "freemhd_input"

    def replace(relative: str, old: str, new: str) -> None:
        path = case / relative
        text = path.read_text()
        assert old in text
        path.write_text(text.replace(old, new))

    if mutation == "mesh":
        replace("system/blockMeshDict", "physicalHalfWidth 0.0439", "physicalHalfWidth 0.05")
    elif mutation == "field":
        path = case / "system/liquid/setExprFieldsDict"
        text = path.read_text().replace('"bc=1"', '"bc=0.99"')
        values = dict(re.findall(r'"([xb][a-z])=([^\"]+)"', text))
        labels = sorted(name[1:] for name in values if name.startswith("x"))
        anchors = {"x_over_L": [float(values[f"x{label}"]) for label in labels], "b_over_B0": [float(values[f"b{label}"]) for label in labels]}
        digest = hashlib.sha256(json.dumps(anchors, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        text = re.sub(r'(lmxFieldAnchorsSHA256 ")[0-9a-f]{64}', rf'\g<1>{digest}', text)
        for region in ("liquid", "solidWalls"):
            (case / f"system/{region}/setExprFieldsDict").write_text(text)
    elif mutation == "fluid":
        path = case / "constant/liquid/thermophysicalProperties.liquidMetal"
        path.write_text(re.sub(r"(\bmu\s+)[^;]+", r"\g<1>0.0001", path.read_text(), count=1))
    elif mutation == "wall":
        replace("constant/solidWalls/thermophysicalProperties", "elcond 3.5", "elcond 3.6")
    elif mutation == "velocity":
        replace("system/liquid/changeDictionaryDict", "sink { type zeroGradient", "sink { type fixedValue")
    elif mutation == "pressure":
        replace("system/liquid/changeDictionaryDict", "sink { type fixedValue; value uniform 0; }", "sink { type fixedValue; value uniform 1; }")
    elif mutation == "electric":
        replace("system/liquid/changeDictionaryDict", "inlet { type zeroGradient; } sink", "inlet { type fixedValue; value uniform 0; } sink")
    elif mutation == "scheme":
        replace("system/liquid/fvSchemes", "div(rhoPhi,U) Gauss limitedLinear 1.0", "div(rhoPhi,U) Gauss limitedLinear 0.5")
    elif mutation == "iterations":
        for region in ("liquid", "solidWalls"):
            replace(f"system/{region}/fvSolution", "maxIter 600", "maxIter 601")
    else:
        path = tmp_path / "freemhd_source/src/LimitFuncs.C"
        path.write_text(path.read_text() + "\n// byte mutation\n")
        pin = json.loads((tmp_path / "freemhd_source/source-pin.json").read_text())
        pin["files"]["src/LimitFuncs.C"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (tmp_path / "freemhd_source/source-pin.json").write_text(json.dumps(pin, sort_keys=True))

    artifact = "freemhd_source" if mutation == "source" else "freemhd_input"
    record["provenance"]["artifacts"][artifact]["sha256"] = artifact_sha256(tmp_path / artifact, "tree")
    report = validate_matched_b_record(record, expected_case_id="B2-fringing-square", artifact_root=tmp_path)

    assert failed_path in report["failed_checks"]
    assert observe_lmx_b2_contract(tmp_path / "lmx_input", tmp_path / "evaluator") == lmx_before


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("mesh", "y_faces", 0), -2.0),
        (("field_profile", "anchor_b_over_B0", 2), 0.0),
        (("field_profile", "sample_b_over_B0", 2), 0.0),
        (("case", "regions", 0, "viscosity"), 1.0),
        (("case", "boundary_conditions", 1, "value"), 5.0),
        (("case", "time_stepper", "dt"), 1.0),
        (("effective_controls", "dt"), 1.0),
    ],
)
def test_matched_b2_lmx_input_rejects_mutated_facts(tmp_path: Path, path, value):
    source = tmp_path / "source.json"
    payload = run_freemhd_parity_suite.materialize_matched_b2_lmx_input(source)
    target = payload
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    source.write_text(json.dumps(payload, sort_keys=True))
    with pytest.raises(ValueError, match="Matched B2|Invalid matched B2"):
        load_matched_b2_lmx_input(source)


def test_parity_command_materializes_without_running_suite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    install, output = tmp_path / "freemhd", tmp_path / "case"
    called = {}

    def materialize(template, destination, *, case_kind):
        called.update(template=template, destination=destination, case_kind=case_kind)
        return {"case_kind": case_kind, "audit": {"matched": True}}

    monkeypatch.setattr(run_freemhd_parity_suite, "materialize_matched_freemhd_case", materialize)
    monkeypatch.setattr(run_freemhd_parity_suite, "run_suite", lambda **_: pytest.fail("parity must not run"))
    assert run_freemhd_parity_suite.main(
        ["--output", str(output), "--freemhd-install-dir", str(install), "--materialize", "hunt"]
    ) == 0
    assert called == {
        "template": install / "cases/hunt_demo",
        "destination": output,
        "case_kind": "hunt",
    }
    monkeypatch.setattr(
        run_freemhd_parity_suite,
        "materialize_matched_b2_preflight",
        lambda template, source, destination: {
            "template": str(template), "source": str(source), "output": str(destination)
        },
    )
    assert run_freemhd_parity_suite.main(
        ["--output", str(output), "--freemhd-install-dir", str(install), "--matched-b2-preflight"]
    ) == 0


def test_parity_command_portably_skips_missing_references(tmp_path: Path):
    output = tmp_path / "parity"
    assert run_freemhd_parity_suite.main(
        [
            "--output",
            str(output),
            "--freemhd-install-dir",
            str(tmp_path / "missing-freemhd"),
            "--processed-root",
            str(tmp_path / "missing-processed"),
        ]
    ) == 0
    assert json.loads((output / "summary.json").read_text())["status"] == "skipped"
    assert (output / "summary.md").is_file()


@pytest.mark.parametrize("matched", [True, False])
def test_parity_suite_gates_profile_comparison_on_audited_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, matched: bool
):
    install = tmp_path / "freemhd"
    for case_kind in ("shercliff", "hunt"):
        (install / "freemhd_output" / case_kind).mkdir(parents=True)
    transient = SimpleNamespace(OUTPUT_DIR=tmp_path / "unset", FREEMHD_INSTALL_DIR=tmp_path / "unset")
    transient.run_freemhd_closed_channel_parity = lambda: {
        "records": [{"y_l2_error": 0.03, "z_l2_error": 0.02, "u_max_abs_diff": 0.01}]
    }
    monkeypatch.setattr(examples, "freemhd_closed_channel_parity", transient, raising=False)
    processed = tmp_path / ("processed" if matched else "missing")
    if matched:
        processed.mkdir()
        observable = SimpleNamespace(OUTPUT_DIR=tmp_path / "unset", REFERENCE_ROOT=tmp_path / "unset")
        observable.run_freemhd_closed_channel_observable_parity = lambda: {
            "records": [
                {
                    "observables": {
                        "velocity": {"y": {"l2_error": 0.04}, "z": {"l2_error": 0.05}},
                        "current": {"y": {"l2_error": 0.06}, "z": {"l2_error": 0.01}},
                    }
                }
            ],
            "observable_gate": {"research_grade_validation_pass": True},
        }
        monkeypatch.setattr(examples, "freemhd_closed_channel_observable_parity", observable, raising=False)
    monkeypatch.setattr(
        run_freemhd_parity_suite,
        "audit_freemhd_case_against_spec",
        lambda *_args, case_kind, **_kwargs: {
            "case_kind": case_kind,
            "matched": matched,
            "failed_check_count": 0 if matched else 1,
        },
    )

    summary = run_freemhd_parity_suite.run_suite(
        output=tmp_path / "out",
        freemhd_install_dir=install,
        processed_root=processed,
    )

    assert summary["status"] == ("completed" if matched else "invalid_reference")
    assert summary["matched_case_gate"] is matched
    assert ("closed_channel_parity" in summary["runs"]) is matched
    metrics = summary["parity_report"]["metrics"]
    assert metrics["reference_sample_y_l2_error"] == (0.06 if matched else None)
    assert metrics["reference_sample_z_l2_error"] == (0.05 if matched else None)
    if matched:
        markdown = tmp_path / "summary.md"
        run_freemhd_parity_suite._write_markdown(markdown, summary)
        rendered = markdown.read_text()
        assert "Research-grade pass: `True`" in rendered
        assert "shercliff: matched=`True`" in rendered
    else:
        monkeypatch.setattr(run_freemhd_parity_suite, "run_suite", lambda **_: summary)
        assert run_freemhd_parity_suite.main(["--output", str(tmp_path / "invalid")]) == 2


def test_benchmark_a_spec_loader_rejects_unsupported_case():
    with pytest.raises(ValueError, match="Unsupported matched Benchmark-A"):
        load_benchmark_a_spec("hartmann")


@pytest.mark.parametrize(
    ("case_kind", "old", "new", "message"),
    [
        (
            "shercliff",
            "schema_version = 1",
            "schema_version = 2",
            "Invalid matched benchmark identity",
        ),
        (
            "shercliff",
            "kinematic_viscosity = 1.0e-3",
            "kinematic_viscosity = 2.0e-3",
            "Inconsistent dynamic",
        ),
        (
            "shercliff",
            "vector = [0.0, 0.2, 0.0]",
            "vector = [0.0, 0.3, 0.0]",
            "do not reproduce Ha",
        ),
        ("shercliff", "[65, 49]", "[57, 43]", "refinement ratios are too uneven"),
        (
            "hunt",
            "conductance_ratio = 0.05",
            "conductance_ratio = 0.06",
            "do not reproduce the conductance ratio",
        ),
    ],
)
def test_benchmark_a_spec_loader_rejects_inconsistent_inputs(tmp_path: Path, case_kind: str, old: str, new: str, message: str):
    source = Path("benchmarks/specs") / f"{case_kind}-ha20.toml"
    (tmp_path / source.name).write_text(source.read_text().replace(old, new, 1))
    with pytest.raises(ValueError, match=message):
        load_benchmark_a_spec(case_kind, tmp_path)


def test_samper_table_i_reference_is_complete_and_exact(tmp_path: Path):
    table = load_samper_table_i()
    rows = {(row["case_kind"], row["hartmann_number"]): row for row in table["cases"]}

    assert len(table["sha256"]) == 64
    assert rows[("shercliff", 500)]["analytical_flow_rate"] == pytest.approx(7.680e-3)
    assert rows[("shercliff", 15000)]["published_numerical_flow_rate"] == pytest.approx(2.648e-4)
    assert rows[("hunt", 500)]["hartmann_wall_conductance"] == pytest.approx(0.01)
    assert rows[("hunt", 15000)]["analytical_flow_rate"] == pytest.approx(2.425e-6)

    source = Path("benchmarks/references/samper-table-i.toml")
    invalid = tmp_path / "invalid.toml"
    invalid.write_text(source.read_text().replace('case_kind = "hunt"', 'case_kind = "other"', 1))
    with pytest.raises(ValueError, match="Incomplete hunt Hartmann ladder"):
        load_samper_table_i(invalid)


def test_freemhd_audit_reports_missing_inputs_and_explicit_nu(tmp_path: Path):
    report = run_freemhd_parity_suite.audit_freemhd_case_against_spec(tmp_path, case_kind="shercliff")
    failed_names = {check["name"] for check in report["checks"] if not check["pass"]}
    assert "geometry.available" in failed_names
    assert "fluid.available" in failed_names
    assert report["physical_hartmann_number"] is None

    liquid = tmp_path / "constant" / "liquid"
    liquid.mkdir(parents=True)
    (liquid / "thermophysicalProperties").write_text("sigma 3;\nrho 1000;\nnu 2e-6;\n")
    properties = infer_liquid_material_properties(tmp_path)
    assert properties == pytest.approx(
        {
            "conductivity": 3.0,
            "density": 1000.0,
            "dynamic_viscosity": 2.0e-3,
            "kinematic_viscosity": 2.0e-6,
        }
    )


def test_freemhd_inference_helpers_recover_geometry_materials_and_b0(tmp_path: Path):
    _write_reference_inputs(tmp_path)

    assert infer_liquid_material_properties(tmp_path) == pytest.approx(
        {
            "conductivity": 1.0e6,
            "density": 1000.0,
            "dynamic_viscosity": 1.0e-3,
            "kinematic_viscosity": 1.0e-6,
        }
    )
    assert infer_uniform_b0(tmp_path) == pytest.approx((0.0, 0.2, 0.0))
    width, height, wall_thickness, wall_cells = infer_rectangular_geometry(tmp_path)
    assert width == pytest.approx(0.2)
    assert height == pytest.approx(0.2)
    assert wall_thickness == pytest.approx(0.001)
    assert wall_cells == 2
    assert infer_solid_conductivities(tmp_path) == pytest.approx((5.0e6, 1.0e-6))


def test_freemhd_inference_helpers_cover_missing_and_fallback_paths(tmp_path: Path):
    assert infer_uniform_b0(tmp_path) is None
    assert infer_rectangular_geometry(tmp_path) is None
    assert infer_solid_conductivities(tmp_path) == (None, None)

    incomplete_liquid = tmp_path / "case" / "constant" / "liquid"
    incomplete_liquid.mkdir(parents=True)
    (incomplete_liquid / "thermophysicalProperties").write_text("sigma 3.0;\nrho 1000;\nmu 0.002;\n")
    assert infer_liquid_material_properties(tmp_path)["kinematic_viscosity"] == pytest.approx(2.0e-6)

    b0_dir = tmp_path / "case" / "0" / "liquid"
    b0_dir.mkdir(parents=True, exist_ok=True)
    (b0_dir / "B0").write_text("internalField nonuniform List<vector> 0();\n")
    assert infer_uniform_b0(tmp_path) is None

    system = tmp_path / "case" / "system"
    system.mkdir(parents=True)
    (system / "blockMeshDict").write_text("Ly_wall 0.1;\n")
    assert infer_rectangular_geometry(tmp_path) is None
    (system / "blockMeshDict").write_text("Ly 0.1;\nLy_wall 0.09;\n")
    width, height, wall_thickness, wall_cells = infer_rectangular_geometry(tmp_path)
    assert width == pytest.approx(0.2)
    assert height == pytest.approx(0.2)
    assert wall_thickness is None
    assert wall_cells is None


def test_inlet_flow_rate_helpers_cover_malformed_and_fallback_cases(tmp_path: Path):
    u_dir = tmp_path / "0"
    u_dir.mkdir()
    _write_u(u_dir, "outlet { type zeroGradient; }")
    assert infer_inlet_drive_mode(tmp_path) is None
    assert infer_inlet_flow_rate(tmp_path) is None

    _write_u(u_dir, "inlet { value uniform (0.2 0 0); }")
    assert infer_inlet_drive_mode(tmp_path) is None
    assert infer_inlet_flow_rate(tmp_path) is None

    _write_u(u_dir, "inlet { type flowRateInletVelocity; volumetricFlowRate 0.0; }")
    assert infer_inlet_flow_rate(tmp_path) == pytest.approx(0.0)

    _write_u(
        u_dir,
        "inlet { type flowRateInletVelocity; volumetricFlowRate constant 0.125; }",
    )
    assert infer_inlet_flow_rate(tmp_path) == pytest.approx(0.125)
