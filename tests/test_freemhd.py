import hashlib
import json
import re
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import lmx.validation as benchmarks
from lmx.io import write_extruded_bundle_restart_npz
from lmx.specs import ExtrudedFieldBundle
from lmx.validation import (
    BENCHMARK_B_SPEC_FILES,
    canonical_matched_b_contract,
    load_benchmark_a_spec,
    load_benchmark_b_reference,
    load_benchmark_b_spec,
    load_samper_table_i,
)
from scripts import run_freemhd_parity_suite
from validation.freemhd import (
    _validate_b2_smoke_execution,
    artifact_sha256,
    infer_inlet_drive_mode,
    infer_inlet_flow_rate,
    infer_liquid_material_properties,
    infer_rectangular_geometry,
    infer_solid_conductivities,
    infer_uniform_b0,
    load_matched_b2_lmx_input,
    observe_freemhd_b2_contract,
    observe_freemhd_b2_output,
    observe_lmx_b2_contract,
    observe_lmx_b2_output,
    validate_matched_b_record,
)

pytestmark = pytest.mark.unit


def test_repository_parity_script_bootstraps_current_source_tree():
    completed = subprocess.run(
        [sys.executable, "scripts/run_freemhd_parity_suite.py", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--matched-b2-smoke" in completed.stdout


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


def test_matched_b2_direct_can_exclude_optional_checkpoint_callback(monkeypatch):
    calls = []
    bundle = SimpleNamespace(**{name: np.zeros(1) for name in ("u", "p", "phi")})

    def solve(problem, **options):
        calls.append(options)
        return SimpleNamespace(bundle=bundle)

    monkeypatch.setattr("lmx.fringing.solve_extruded_inductionless", solve)
    problem = SimpleNamespace(case=SimpleNamespace(time_stepper=SimpleNamespace(max_steps=4)))
    checkpoint, _ = run_freemhd_parity_suite._run_matched_b2_lmx_direct(
        problem, num_devices=1, capture_checkpoint=False
    )
    assert checkpoint is None and calls == [{"num_devices": 1}]


def _matched_b_record(root: Path, case_id: str, *, role: str | None = None) -> dict[str, object]:
    role = role or ("b1-production" if case_id.startswith("B1") else "b2-production")
    manifest = canonical_matched_b_contract(load_benchmark_b_spec(case_id), role)
    reference = load_benchmark_b_reference(case_id)
    spec_path = Path("src/lmx/data/benchmarks/specs") / BENCHMARK_B_SPEC_FILES[case_id]
    artifacts = {}
    for name in (
        "lmx_source",
        "freemhd_source",
        "lmx_input",
        "freemhd_input",
        "evaluator",
        "lmx_output",
        "freemhd_output",
    ):
        kind = "tree" if name in {"lmx_source", "freemhd_source", "freemhd_input"} else "file"
        path = root / name
        if kind == "tree":
            path.mkdir(parents=True)
            (path / "evidence.txt").write_text(name)
        else:
            path.write_text(name)
        artifacts[name] = {
            "path": name,
            "kind": kind,
            "sha256": _test_artifact_sha256(path, kind),
        }
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
        (system / "changeDictionaryDict").write_text(
            "B0 { internalField uniform (0 10 0); value uniform (0 10 0); }\n"
        )
    liquid_constant = root / "constant" / "liquid"
    liquid_constant.mkdir(parents=True)
    (liquid_constant / "thermophysicalProperties.liquidMetal").write_text(
        "rho 1000;\nmu 1;\nelcond [-1 -3 3 0 0 2 0] 1e6;\n"
    )
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


def _write_lmx_b2_output(
    root: Path,
    input_path: Path,
    evaluator: Path,
    *,
    direct_u: float = 0.0,
    resumed_u_delta: float = 0.0,
) -> None:
    payload = json.loads(input_path.read_text())
    x = benchmarks.jnp.asarray(payload["field_profile"]["sample_x_over_L"])
    faces = benchmarks.jnp.asarray(payload["mesh"]["y_faces"])
    y = z = 0.5 * (faces[1:] + faces[:-1])
    shape, dt = (8, 7, 7), payload["effective_controls"]["dt"]
    executed_steps = payload["effective_controls"]["executed_steps"]
    checkpoint_step = (executed_steps + 1) // 2
    case = load_matched_b2_lmx_input(input_path).case

    def bundle(steps: int, *, u_value: float = 0.0) -> ExtrudedFieldBundle:
        zeros = benchmarks.jnp.zeros(shape)
        compact_flux = benchmarks.jnp.zeros((3, 8, 5, 5))
        inlet_flux = benchmarks.jnp.zeros((5, 5))
        anderson_state = (
            (
                benchmarks.jnp.zeros((3, *shape)),
                benchmarks.jnp.zeros((3, *shape)),
                compact_flux,
                inlet_flux,
            )
            if case.solver.coupling_acceleration == "anderson"
            else None
        )
        return ExtrudedFieldBundle(
            x=x,
            y=y,
            z=z,
            field_scale=benchmarks.jnp.asarray(payload["field_profile"]["sample_b_over_B0"]),
            u=benchmarks.jnp.full(shape, u_value),
            v=zeros,
            w=zeros,
            p=zeros,
            phi=zeros,
            geometry_kind="layered_duct",
            solver_kind="extruded_inductionless",
            rho_phi_plus=compact_flux,
            rho_phi_inlet=inlet_flux,
            aitken_state=((None, 1.0, 0) if case.solver.coupling_acceleration == "aitken" else None),
            anderson_state=anderson_state,
            stopping_state=(
                steps,
                0,
                "step_limit" if steps == executed_steps else "in_progress",
            ),
            jx=benchmarks.jnp.ones(shape),
            jz=benchmarks.jnp.ones(shape),
            volumetric_flow_rate=benchmarks.jnp.full(8, 4.0),
            charge_balance_residual=benchmarks.jnp.full(8, 1.0e-5),
            boundary_current_residual=benchmarks.jnp.full(8, 1.0e-5),
            transverse_pressure_difference=benchmarks.jnp.asarray([0, 0, 1, 2, 3, 2, 0, 0]),
            iteration_residual_history=benchmarks.jnp.zeros(steps),
            iteration_momentum_defect_history=benchmarks.jnp.zeros(steps),
            iteration_component_residual_history=benchmarks.jnp.zeros((steps, 6)),
            iteration_pressure_residual_history=benchmarks.jnp.zeros(steps),
            iteration_pressure_linear_history=benchmarks.jnp.tile(
                benchmarks.jnp.asarray([1.0e-12, 1.0e-13, 4.0, 1.0, 1.0]),
                (steps, 1),
            ),
            iteration_electric_linear_history=benchmarks.jnp.zeros((steps, 6)),
            iteration_potential_residual_history=benchmarks.jnp.zeros(steps),
            iteration_courant_history=benchmarks.jnp.tile(
                benchmarks.jnp.asarray([dt, 1e-6, 2e-6]), (steps, 1)
            ),
        )

    root.mkdir()
    for name, value in (
        ("checkpoint.npz", bundle(checkpoint_step)),
        ("direct.npz", bundle(executed_steps, u_value=direct_u)),
        ("resumed.npz", bundle(executed_steps, u_value=direct_u + resumed_u_delta)),
    ):
        write_extruded_bundle_restart_npz(value, case, root / name)
    (root / "run.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "code": "LMX",
                "case_id": "B2-fringing-square",
                "input_sha256": artifact_sha256(input_path, "file"),
                "evaluator_sha256": artifact_sha256(evaluator, "file"),
                "wall_seconds": 1.0,
                "num_devices": 1,
                "float_precision": "float64",
            }
        )
    )


def _write_freemhd_b2_output(root: Path, input_dir: Path, evaluator: Path) -> None:
    dt = 1.0 / 540000.0
    times = (dt, 2.0 * dt)
    root.mkdir()
    (root / "controlDict.used").write_bytes((input_dir / "system/controlDict").read_bytes())
    (root / "run.log").write_text(
        "trapFpe: Floating point exception trapping enabled (FOAM_SIGFPE).\n"
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
    probes.write_text("\n".join([*headers, "# Time", *rows]) + "\n")
    values = {
        "massIn": (-4.0, "sum(rhoPhi)"),
        "massOut": (4.0, "sum(rhoPhi)"),
        "currentIn": (-0.1, "sum(jn)"),
        "currentOut": (0.1, "sum(jn)"),
        "currentIntoSolid": (1.0e-5, "sum(jn)"),
        "currentIntoSolidMagnitude": (1.0, "sumMag(jn)"),
    }
    for name, (value, header) in values.items():
        path = post / name / "0/surfaceFieldValue.dat"
        path.parent.mkdir(parents=True)
        path.write_text(f"# Time {header}\n" + "".join(f"{time:.17g} {value}\n" for time in times))
    (root / "run.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "code": "FreeMHD",
                "case_id": "B2-fringing-square",
                "input_sha256": artifact_sha256(input_dir, "tree"),
                "evaluator_sha256": artifact_sha256(evaluator, "file"),
                "wall_seconds": 1.0,
                "nproc": 2,
                "image": "freemhd:test",
                "float_precision": "float64",
            }
        )
    )


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
        reference[key], reference[f"{key}_sha256"] = (
            relative,
            manifest["files"][relative],
        )
    monkeypatch.setattr(benchmarks, "load_benchmark_b_spec", lambda *_: spec)
    artifacts = {}
    for name in (
        "lmx_source",
        "freemhd_source",
        "lmx_input",
        "freemhd_input",
        "evaluator",
        "lmx_output",
        "freemhd_output",
    ):
        kind = (
            "tree"
            if name in {"lmx_source", "freemhd_source", "freemhd_input"}
            or executed
            and name.endswith("_output")
            else "file"
        )
        artifacts[name] = {
            "path": name,
            "kind": kind,
            "sha256": artifact_sha256(root / name, kind),
        }
    contract = canonical_matched_b_contract(spec, "harness-smoke")
    return {
        "schema_version": 3 if executed else 2,
        "case_id": "B2-fringing-square",
        "acceptance_role": "harness-smoke",
        "contract": {"lmx": deepcopy(contract), "freemhd": deepcopy(contract)},
        "comparison": {"source": "independent-output-observers"}
        if executed
        else {"x_over_L": [], "lmx_observable": [], "freemhd_observable": []},
        "provenance": {
            "benchmark_spec_sha256": hashlib.sha256(
                Path("src/lmx/data/benchmarks/specs/alex-b2-square.toml").read_bytes()
            ).hexdigest(),
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

    assert (
        report["schema_complete"]
        and report["artifact_pass"]
        and report["contract_pass"]
        and report["comparison_pass"]
    )
    assert report["observation_pass"] is report["acceptance_pass"] is False
    assert set(report["calculated_artifact_sha256"]) == set(record["provenance"]["artifacts"])
    assert "contract.observers.unavailable" in report["failed_checks"]
    assert report["metrics"]["weighted_rms"] == pytest.approx(0.0)
    assert validate_matched_b_record(record, expected_case_id=case_id)["artifact_pass"] is False
    (tmp_path / "not-directory").write_text("not a directory\n")
    assert not validate_matched_b_record(
        record, expected_case_id=case_id, artifact_root=tmp_path / "not-directory"
    )["artifact_pass"]


def test_matched_b_schema2_rejects_contract_and_record_forgery(tmp_path: Path):
    record = _matched_b_record(tmp_path, "B2-fringing-square")
    mismatch = deepcopy(record)
    mismatch["contract"]["freemhd"]["equations"] = {"semantic_contract": "different"}
    mismatch["exact_case_match"] = mismatch["pass"] = True
    rejected = validate_matched_b_record(
        mismatch, expected_case_id="B2-fringing-square", artifact_root=tmp_path
    )
    assert "contract.equations.mismatch" in rejected["failed_checks"]
    assert "schema.exact_case_match" in rejected["failed_checks"]

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
        report = validate_matched_b_record(
            malformed, expected_case_id="B2-fringing-square", artifact_root=tmp_path
        )
        assert expected_check in report["failed_checks"]

    for section, key, wrong in (
        ("equations", "inertia", "omitted"),
        ("boundary_drive", "flow_constraint_scope", "stationwise"),
    ):
        self_consistent = deepcopy(record)
        self_consistent["contract"]["lmx"][section][key] = wrong
        self_consistent["contract"]["freemhd"][section][key] = wrong
        rejected = validate_matched_b_record(
            self_consistent,
            expected_case_id="B2-fringing-square",
            artifact_root=tmp_path,
        )
        assert f"contract.{section}.canonical" in rejected["failed_checks"]

    smoke = deepcopy(record)
    smoke["acceptance_role"] = "harness-smoke"
    report = validate_matched_b_record(smoke, expected_case_id="B2-fringing-square", artifact_root=tmp_path)
    assert report["contract_pass"] is report["role_allows_acceptance"] is report["acceptance_pass"] is False
    assert "contract.mesh_coordinates.canonical" in report["failed_checks"]

    poor = deepcopy(record)
    poor["comparison"]["freemhd_observable"] = [
        value + 1.0 for value in poor["comparison"]["freemhd_observable"]
    ]
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

    manifest = run_freemhd_parity_suite.materialize_matched_freemhd_case(
        template, output, case_kind=case_kind
    )

    assert manifest["run_profile"] == "docker_smoke_only"
    assert manifest["audit"]["matched"] is True
    assert manifest["audit"]["physical_hartmann_number"] == pytest.approx(20.0)
    assert len(manifest["source_template_sha256"]) == 64
    assert (output / "lmx-benchmark-manifest.json").is_file()
    assert infer_uniform_b0(output) == pytest.approx((0.0, 0.2, 0.0))
    assert infer_inlet_drive_mode(output) == "inlet_flow_rate"
    assert infer_inlet_flow_rate(output) == pytest.approx(
        load_benchmark_a_spec(case_kind)["drive"]["target_flow_rate"]
    )
    assert (
        run_freemhd_parity_suite.materialize_matched_freemhd_case(
            template, second_output, case_kind=case_kind
        )
        == manifest
    )
    assert (second_output / "lmx-benchmark-manifest.json").read_bytes() == (
        output / "lmx-benchmark-manifest.json"
    ).read_bytes()
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
    for args in (
        ("init", "-q"),
        ("config", "user.email", "test@example.com"),
        ("config", "user.name", "Test"),
        ("add", "."),
        ("commit", "-qm", "fixture"),
    ):
        subprocess.run(("git", "-C", str(repo), *args), check=True)
    reference["repository_commit"] = subprocess.check_output(
        ("git", "-C", str(repo), "rev-parse", "HEAD"), text=True
    ).strip()
    spec = {"free_mhd_discretization_reference": reference}
    monkeypatch.setattr(
        run_freemhd_parity_suite,
        "load_benchmark_b_spec",
        lambda case_id, spec_root=None: spec,
    )
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
    assert set(manifest) == {
        "schema_version",
        "project",
        "commit",
        "openfoam_release",
        "files",
    }
    assert manifest["commit"] == reference["repository_commit"]
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_freemhd_parity_suite.materialize_freemhd_source_snapshot(repo, first)


@pytest.mark.parametrize(
    "hazard",
    ["head", "unstaged", "staged", "hash", "missing", "symlink", "noncanonical"],
)
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
    first, second, scaled, sustained, sustained_copy = (
        tmp_path / name
        for name in (
            "first.json",
            "second.json",
            "scaled.json",
            "sustained.json",
            "sustained-copy.json",
        )
    )
    payload = run_freemhd_parity_suite.materialize_matched_b2_lmx_input(first)
    run_freemhd_parity_suite.materialize_matched_b2_lmx_input(second)
    problem, contract = load_matched_b2_lmx_input(first), observe_lmx_b2_contract(first)

    assert first.read_bytes() == second.read_bytes()
    assert payload["mesh"]["x_faces"] == [
        -15.0,
        -11.875,
        -8.75,
        -5.625,
        -2.5,
        0.625,
        3.75,
        6.875,
        10.0,
    ]
    assert payload["field_profile"]["sample_b_over_B0"] == [
        1.0,
        1.0,
        0.991875,
        0.9371875,
        0.69,
        0.16125,
        0.00875,
        0.0,
    ]
    assert problem.case.name == "alex_b2-fringing-square_harness-smoke"
    payload["mesh"]["y_faces"][2] += np.finfo(float).eps
    first.write_text(json.dumps(payload))
    assert load_matched_b2_lmx_input(first).case == problem.case
    groups = contract["nondimensional_groups"]
    assert {
        name: groups[name] for name in ("hartmann_number", "interaction_parameter", "reynolds_number")
    } == pytest.approx(
        {
            "hartmann_number": 2900.0,
            "interaction_parameter": 540.0,
            "reynolds_number": 2900.0**2 / 540.0,
        }
    )
    assert groups["magnetic_reynolds_number_assumption"] == "Rm << 1"
    assert contract["stopping_rules"]["expected_stop_reason"] == "step_limit"
    scaled_payload = run_freemhd_parity_suite.materialize_matched_b2_lmx_input(
        scaled, solver_shape=(16, 7, 7)
    )
    scaled_problem = load_matched_b2_lmx_input(scaled)
    geometry = scaled_problem.case.geometry
    assert (geometry.nx, geometry.ny, geometry.nz) == (16, 5, 5)
    assert len(scaled_payload["field_profile"]["sample_x_over_L"]) == 16
    sustained_payload = run_freemhd_parity_suite.materialize_matched_b2_lmx_input(sustained, executed_steps=6)
    run_freemhd_parity_suite.materialize_matched_b2_lmx_input(sustained_copy, executed_steps=6)
    sustained_problem = load_matched_b2_lmx_input(sustained)
    assert sustained.read_bytes() == sustained_copy.read_bytes()
    assert sustained_payload["effective_controls"]["executed_steps"] == 6
    assert sustained_problem.case.name.endswith("_scaling-calibration")
    assert sustained_problem.case.time_stepper.max_steps == 6
    assert sustained_problem.case.time_stepper.t_final == pytest.approx(
        6 * sustained_problem.case.time_stepper.dt
    )
    assert observe_lmx_b2_contract(sustained)["stopping_rules"] == (sustained_payload["effective_controls"])
    for invalid in (True, 1, 2.5):
        with pytest.raises((TypeError, ValueError), match="executed_steps"):
            run_freemhd_parity_suite.materialize_matched_b2_lmx_input(
                tmp_path / f"invalid-{invalid}.json", executed_steps=invalid
            )
    with pytest.raises(ValueError, match="requires nx"):
        run_freemhd_parity_suite.materialize_matched_b2_lmx_input(tmp_path / "bad", solver_shape=(8, 2, 7))
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_freemhd_parity_suite.materialize_matched_b2_lmx_input(first)


def test_matched_b2_freemhd_input_is_deterministic_slim_and_solver_free(tmp_path: Path):
    template, first, second = (
        tmp_path / "hunt_demo",
        tmp_path / "first",
        tmp_path / "second",
    )
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
    assert "maxCo 0.4; maxAlphaCo 0.3;" in functions and "timePrecision 16;" in functions
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
    lmx_input, evaluator = (
        tmp_path / "bundle/lmx_input.json",
        tmp_path / "bundle/evaluator.json",
    )

    assert summary["status"] == "preflight-pass"
    assert observe_freemhd_b2_contract(case, source, evaluator) == observe_lmx_b2_contract(
        lmx_input, evaluator
    )


@pytest.mark.parametrize("acceleration", ("anderson", "aitken"))
@pytest.mark.parametrize("executed_steps", (2, 6))
def test_lmx_b2_output_observer_replays_restart_evidence(
    tmp_path: Path, acceleration: str, executed_steps: int
):
    input_path, evaluator = tmp_path / "lmx.json", tmp_path / "evaluator.json"
    run_freemhd_parity_suite.materialize_matched_b2_lmx_input(input_path, executed_steps=executed_steps)
    if acceleration == "aitken":
        payload = json.loads(input_path.read_text())
        payload["case"]["solver"]["coupling_acceleration"] = acceleration
        input_path.write_text(json.dumps(payload, sort_keys=True))
    run_freemhd_parity_suite.materialize_matched_b2_evaluator(evaluator)
    _write_lmx_b2_output(tmp_path / "output", input_path, evaluator)

    observed = observe_lmx_b2_output(tmp_path / "output", input_path, evaluator)

    assert observed["steps"] == executed_steps
    assert observed["stop_reason"] == "step_limit"
    assert len(observed["dt"]) == len(observed["courant_mean"]) == executed_steps
    assert all(
        observed[name] == 0.0
        for name in (
            "restart_max_abs",
            "restart_state_max_abs",
            "restart_flux_max_abs",
            "restart_state_relative_l2",
            "restart_state_tolerance_ratio",
            "restart_flux_relative_l2",
            "restart_derived_max_abs",
            "restart_history_max_abs",
            "mass_balance",
        )
    )
    assert observed["current_balance"] == observed["interface_current_balance"] == pytest.approx(1.0e-5)
    assert observed["pressure_observable"][4] == pytest.approx(3.0 / 540.0)


def test_lmx_b2_output_observer_measures_relative_restart_state_corruption(
    tmp_path: Path,
):
    input_path, evaluator = tmp_path / "lmx.json", tmp_path / "evaluator.json"
    output = tmp_path / "output"
    run_freemhd_parity_suite.materialize_matched_b2_lmx_input(input_path)
    run_freemhd_parity_suite.materialize_matched_b2_evaluator(evaluator)
    _write_lmx_b2_output(
        output,
        input_path,
        evaluator,
        direct_u=1.0e8,
        resumed_u_delta=1.0e-3,
    )

    observed = observe_lmx_b2_output(output, input_path, evaluator)

    assert observed["restart_state_max_abs"] == pytest.approx(1.0e-3, rel=1.0e-5)
    assert 0.0 < observed["restart_state_relative_l2"] <= 1.0e-10
    assert observed["restart_state_tolerance_ratio"] < 1.0


@pytest.mark.parametrize("mutation", ("root", "metadata", "provenance"))
def test_lmx_b2_output_observer_rejects_corrupt_evidence(tmp_path: Path, mutation: str):
    input_path, evaluator, output = (
        tmp_path / "lmx.json",
        tmp_path / "evaluator.json",
        tmp_path / "output",
    )
    run_freemhd_parity_suite.materialize_matched_b2_lmx_input(input_path)
    run_freemhd_parity_suite.materialize_matched_b2_evaluator(evaluator)
    _write_lmx_b2_output(output, input_path, evaluator)
    path, old, new = {
        "root": (output / "unexpected", None, "unexpected\n"),
        "metadata": (output / "run.json", '"code": "LMX"', '"code": "bad"'),
        "provenance": (
            output / "run.json",
            '"float_precision": "float64"',
            '"float_precision": "bad"',
        ),
    }[mutation]
    path.write_text(new if old is None else path.read_text().replace(old, new, 1))
    with pytest.raises(ValueError, match="LMX B2"):
        observe_lmx_b2_output(output, input_path, evaluator)


def test_freemhd_b2_output_observer_reads_only_native_tables(tmp_path: Path):
    template, case, evaluator = (
        tmp_path / "template",
        tmp_path / "case",
        tmp_path / "evaluator",
    )
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


@pytest.mark.parametrize(
    "mutation",
    (
        "root",
        "metadata",
        "provenance",
        "controls",
        "fatal",
        "execution",
        "post",
        "width",
        "rows",
        "table_missing",
        "header",
        "time",
        "probe_missing",
        "probe_headers",
        "columns",
        "probe",
        "magnitude",
        "stations",
    ),
)
def test_freemhd_b2_output_observer_rejects_corrupt_evidence(tmp_path: Path, mutation: str):
    template, case, evaluator = (
        tmp_path / "template",
        tmp_path / "case",
        tmp_path / "evaluator",
    )
    _write_b2_skeleton(template)
    run_freemhd_parity_suite.materialize_matched_b2_freemhd_input(template, case)
    run_freemhd_parity_suite.materialize_matched_b2_evaluator(evaluator)
    output = tmp_path / "output"
    _write_freemhd_b2_output(output, case, evaluator)
    path, old, new = {
        "root": (output / "unexpected", None, "unexpected\n"),
        "metadata": (output / "run.json", '"code": "FreeMHD"', '"code": "bad"'),
        "provenance": (
            output / "run.json",
            '"float_precision": "float64"',
            '"float_precision": "bad"',
        ),
        "controls": (
            output / "controlDict.used",
            "adjustTimeStep off",
            "adjustTimeStep on",
        ),
        "fatal": (output / "run.log", "End", "FOAM FATAL ERROR\nEnd"),
        "execution": (output / "run.log", "End", "Stopped"),
        "post": (output / "postProcessing/unexpected", None, "unexpected\n"),
        "width": (
            output / "postProcessing/massIn/0/surfaceFieldValue.dat",
            " -4.0\n",
            " -4.0 0\n",
        ),
        "rows": (
            output / "postProcessing/massIn/0/surfaceFieldValue.dat",
            "3.7037037037037037e-06",
            "1.8518518518518519e-06",
        ),
        "table_missing": (
            output / "postProcessing/massIn/0/surfaceFieldValue.dat",
            None,
            None,
        ),
        "header": (
            output / "postProcessing/massIn/0/surfaceFieldValue.dat",
            "sum(rhoPhi)",
            "sum(phi)",
        ),
        "time": (
            output / "postProcessing/massIn/0/surfaceFieldValue.dat",
            "1.8518518518518519e-06",
            "1e-6",
        ),
        "probe_missing": (output / "postProcessing/b2PressureTaps/0/p", None, None),
        "probe_headers": (
            output / "postProcessing/b2PressureTaps/0/p",
            "# Probe 0",
            "# Probe 1",
        ),
        "columns": (
            output / "postProcessing/b2PressureTaps/0/p",
            "# Time\n",
            "# Bad\n",
        ),
        "probe": (output / "postProcessing/b2PressureTaps/0/p", "0.8 0)", "0.7 0)"),
        "magnitude": (
            output / "postProcessing/currentIntoSolidMagnitude/0/surfaceFieldValue.dat",
            " 1.0",
            " -1.0",
        ),
        "stations": (case / "system/blockMeshDict", "Nx 8;", "Nx 7;"),
    }[mutation]
    if old is None:
        path.unlink() if new is None else path.write_text(new)
    else:
        assert old in path.read_text()
        path.write_text(path.read_text().replace(old, new, 1))
    if mutation == "controls":
        control = case / "system/controlDict"
        control.write_text(control.read_text().replace(old, new, 1))
    if mutation in {"stations", "controls"}:
        metadata = json.loads((output / "run.json").read_text())
        metadata["input_sha256"] = artifact_sha256(case, "tree")
        (output / "run.json").write_text(json.dumps(metadata))

    with pytest.raises(ValueError, match="FreeMHD B2"):
        observe_freemhd_b2_output(output, case, evaluator)


@pytest.mark.parametrize(
    ("target", "key", "value", "failed"),
    (
        ("lmx", "steps", 1, "execution.lmx.stopping"),
        ("lmx", "dt", [0.0], "execution.lmx.dt"),
        ("lmx", "courant_max", [1.0], "execution.lmx.courant"),
        ("lmx", "mass_balance", 1.0, "execution.lmx.mass_balance"),
        (
            "lmx",
            "interface_current_activity",
            0.0,
            "execution.lmx.interface_current_activity",
        ),
        ("lmx", "restart_max_abs", 1.0, "execution.lmx.restart"),
        ("lmx", "dt", None, "execution.lmx.schema"),
        ("freemhd", "x_over_L", [0.0, 2.0], "x"),
        ("freemhd", "courant_mean", [0.1, 0.1], "courant_mean"),
        ("lmx", "pressure_observable", [0.0], "arrays"),
        ("freemhd", "pressure_observable", [1.0, 1.0], "pressure_rms"),
    ),
)
def test_b2_smoke_execution_attributes_each_gate(target, key, value, failed):
    dt = 1.0 / 540000.0
    observed = {
        "steps": 2,
        "stop_reason": "step_limit",
        "dt": [dt, dt],
        "courant_mean": [0.0, 0.0],
        "courant_max": [0.0, 0.0],
        "mass_balance": 0.0,
        "current_balance": 0.0,
        "interface_current_balance": 0.0,
        "interface_current_activity": 1.0,
        "restart_max_abs": 0.0,
        "x_over_L": [0.0, 1.0],
        "pressure_observable": [0.0, 0.0],
    }
    lmx, freemhd = deepcopy(observed), deepcopy(observed)
    selected = {"lmx": lmx, "freemhd": freemhd}[target]
    selected.pop(key) if value is None else selected.__setitem__(key, value)
    execution, comparison, _ = _validate_b2_smoke_execution(
        lmx,
        freemhd,
        load_benchmark_b_spec("B2-fringing-square")["harness_smoke_execution"],
    )
    assert failed in execution + comparison


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
        anchors = {
            "x_over_L": [float(values[f"x{label}"]) for label in labels],
            "b_over_B0": [float(values[f"b{label}"]) for label in labels],
        }
        digest = hashlib.sha256(
            json.dumps(anchors, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        text = re.sub(r'(lmxFieldAnchorsSHA256 ")[0-9a-f]{64}', rf"\g<1>{digest}", text)
        for region in ("liquid", "solidWalls"):
            (case / f"system/{region}/setExprFieldsDict").write_text(text)
    elif mutation == "fluid":
        path = case / "constant/liquid/thermophysicalProperties.liquidMetal"
        path.write_text(re.sub(r"(\bmu\s+)[^;]+", r"\g<1>0.0001", path.read_text(), count=1))
    elif mutation == "wall":
        replace("constant/solidWalls/thermophysicalProperties", "elcond 3.5", "elcond 3.6")
    elif mutation == "velocity":
        replace(
            "system/liquid/changeDictionaryDict",
            "sink { type zeroGradient",
            "sink { type fixedValue",
        )
    elif mutation == "pressure":
        replace(
            "system/liquid/changeDictionaryDict",
            "sink { type fixedValue; value uniform 0; }",
            "sink { type fixedValue; value uniform 1; }",
        )
    elif mutation == "electric":
        replace(
            "system/liquid/changeDictionaryDict",
            "inlet { type zeroGradient; } sink",
            "inlet { type fixedValue; value uniform 0; } sink",
        )
    elif mutation == "scheme":
        replace(
            "system/liquid/fvSchemes",
            "div(rhoPhi,U) Gauss limitedLinear 1.0",
            "div(rhoPhi,U) Gauss limitedLinear 0.5",
        )
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
        (("schema_version",), 2),
        (("case", "geometry"), None),
        (("case", "boundary_conditions"), {}),
        (("case", "name"), "not-the-canonical-case"),
        (("mesh", "coordinate_system"), "unknown"),
        (("field_profile", "axis"), "z"),
        (("scaling", "length_scale"), "full width"),
        (("case", "regions", 1, "kind"), "fluid"),
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
    assert (
        run_freemhd_parity_suite.main(
            [
                "--output",
                str(output),
                "--freemhd-install-dir",
                str(install),
                "--materialize",
                "hunt",
            ]
        )
        == 0
    )
    assert called == {
        "template": install / "cases/hunt_demo",
        "destination": output,
        "case_kind": "hunt",
    }
    monkeypatch.setattr(
        run_freemhd_parity_suite,
        "materialize_matched_b2_preflight",
        lambda template, source, destination: {
            "template": str(template),
            "source": str(source),
            "output": str(destination),
        },
    )
    assert (
        run_freemhd_parity_suite.main(
            [
                "--output",
                str(output),
                "--freemhd-install-dir",
                str(install),
                "--matched-b2-preflight",
            ]
        )
        == 0
    )
    monkeypatch.setattr(
        run_freemhd_parity_suite,
        "run_matched_b2_smoke_bundle",
        lambda template, source, destination, **options: {
            "execution_pass": template == install / "cases/hunt_demo",
            "comparison_pass": destination == output and options["nproc"] == 3,
        },
    )
    assert (
        run_freemhd_parity_suite.main(
            [
                "--output",
                str(output),
                "--freemhd-install-dir",
                str(install),
                "--freemhd-source-repo",
                str(tmp_path / "source"),
                "--matched-b2-smoke",
                "--nproc",
                "3",
                "--smoke-timeout",
                "9",
            ]
        )
        == 0
    )


@pytest.mark.parametrize("result", ["success", "timeout", "failure"])
def test_freemhd_docker_runners_enforce_pins_deadline_and_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, result: str
):
    source, evaluator, output = (
        tmp_path / "input",
        tmp_path / "evaluator",
        tmp_path / "output",
    )
    source.mkdir()
    (source / "input.txt").write_text("immutable\n")
    evaluator.write_text("evaluator\n")
    calls = []

    def run(command, **kwargs):
        calls.append((command, kwargs))
        if command[1:3] == ["image", "inspect"]:
            return SimpleNamespace(stdout=f"sha256:{'b' * 64}\n")
        if command[1] == "run" and result == "timeout":
            raise subprocess.TimeoutExpired(command, kwargs["timeout"])
        if command[1] == "run" and result == "failure":
            raise subprocess.CalledProcessError(1, command, output="setup failed")
        if command[1] == "rm" and result == "timeout":
            raise OSError("cleanup race")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(run_freemhd_parity_suite.subprocess, "run", run)
    monkeypatch.setattr(
        run_freemhd_parity_suite,
        "observe_freemhd_b2_output",
        lambda *args: {"observed": args},
    )

    def invoke():
        return run_freemhd_parity_suite.run_matched_b2_freemhd_smoke(
            source,
            evaluator,
            output,
            image="freemhd:test",
            nproc=2,
            timeout_seconds=7.0,
        )

    if result == "timeout":
        with pytest.raises(TimeoutError, match="exceeded 7 seconds"):
            invoke()
    elif result == "failure":
        with pytest.raises(RuntimeError, match="setup failed"):
            invoke()
    else:
        observed = invoke()
        assert "observed" in observed
        metadata = json.loads((output / "run.json").read_text())
        assert metadata["image"] == "freemhd:test"
    docker = next(call for call in calls if call[0][1] == "run")
    assert docker[1]["timeout"] == 7.0
    rendered = " ".join(docker[0])
    assert {"--rm", "--name", "--cidfile"} <= set(rendered.split()) and "readonly" in rendered
    assert calls[-1][0][1:3] == ["rm", "-f"]
    assert (source / "input.txt").read_text() == "immutable\n"


def test_matched_b2_bundle_runs_lmx_before_freemhd_and_builds_schema3(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    order, captured = [], {}

    def preflight(_template, _source, root):
        root.mkdir()
        for name in ("freemhd_input", "freemhd_source"):
            (root / name).mkdir()
            (root / name / "evidence").write_text(name)
        for name in ("lmx_input.json", "evaluator.json"):
            (root / name).write_text(name)

    def snapshot(root):
        root.mkdir()
        (root / "evidence").write_text("source")

    def lmx(_input, _evaluator, root):
        order.append("lmx")
        root.mkdir()
        (root / "evidence").write_text("lmx")
        return {
            "steps": 2,
            "stop_reason": "step_limit",
            "dt": [1 / 540000] * 2,
            "courant_max": [0.0, 0.0],
            "mass_balance": 0.0,
            "current_balance": 0.0,
            "interface_current_balance": 0.0,
            "interface_current_activity": 1.0,
            "restart_max_abs": 0.0,
        }

    def freemhd(_input, _evaluator, root, **options):
        order.append("freemhd")
        assert 0.0 < options["timeout_seconds"] <= 10.0
        root.mkdir()
        (root / "evidence").write_text("freemhd")

    monkeypatch.setattr(run_freemhd_parity_suite, "materialize_matched_b2_preflight", preflight)
    monkeypatch.setattr(run_freemhd_parity_suite, "materialize_lmx_source_snapshot", snapshot)
    monkeypatch.setattr(run_freemhd_parity_suite, "run_matched_b2_lmx_smoke", lmx)
    monkeypatch.setattr(run_freemhd_parity_suite, "run_matched_b2_freemhd_smoke", freemhd)
    monkeypatch.setattr(run_freemhd_parity_suite, "observe_lmx_b2_contract", lambda *_: {"code": "lmx"})
    monkeypatch.setattr(
        run_freemhd_parity_suite,
        "observe_freemhd_b2_contract",
        lambda *_: {"code": "freemhd"},
    )

    def validate(record, **_):
        captured["record"] = record
        return {"execution_pass": True, "comparison_pass": True}

    monkeypatch.setattr(run_freemhd_parity_suite, "validate_matched_b_record", validate)
    report = run_freemhd_parity_suite.run_matched_b2_smoke_bundle(
        tmp_path / "template",
        tmp_path / "source",
        tmp_path / "bundle",
        total_timeout_seconds=10.0,
    )

    assert order == ["lmx", "freemhd"] and report["execution_pass"]
    assert captured["record"]["comparison"] == {"source": "independent-output-observers"}
    assert all(
        item["kind"] == "tree"
        for name, item in captured["record"]["provenance"]["artifacts"].items()
        if name.endswith("output")
    )


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
        (
            "shercliff",
            "levels = [[37, 29], [49, 37], [65, 49], [85, 63]]",
            "levels = [[37, 29], [49, 37]]",
            "at least three 2D levels",
        ),
        (
            "shercliff",
            "levels = [[37, 29], [49, 37], [65, 49], [85, 63]]",
            "levels = [[37, 29], [37, 29], [65, 49], [85, 63]]",
            "not monotonically refined",
        ),
    ],
)
def test_benchmark_a_spec_loader_rejects_inconsistent_inputs(
    tmp_path: Path, case_kind: str, old: str, new: str, message: str
):
    source = Path("src/lmx/data/benchmarks/specs") / f"{case_kind}-ha20.toml"
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

    source = Path("src/lmx/data/benchmarks/references/samper-table-i.toml")
    invalid = tmp_path / "invalid.toml"
    invalid.write_text(source.read_text().replace('case_kind = "hunt"', 'case_kind = "other"', 1))
    with pytest.raises(ValueError, match="Incomplete hunt Hartmann ladder"):
        load_samper_table_i(invalid)
    invalid.write_text(source.read_text().replace("schema_version = 1", "schema_version = 0"))
    with pytest.raises(ValueError, match="Invalid Samper Table I"):
        load_samper_table_i(invalid)
    invalid.write_text(
        source.read_text().replace("hartmann_wall_conductance = 0.01", "hartmann_wall_conductance = 0.02", 1)
    )
    with pytest.raises(ValueError, match="Incorrect hunt wall conductance"):
        load_samper_table_i(invalid)
    invalid.write_text(
        source.read_text().replace("analytical_flow_rate = 7.680e-3", "analytical_flow_rate = -1.0", 1)
    )
    with pytest.raises(ValueError, match="Non-positive shercliff flow-rate"):
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
