from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

import examples
from lmx.benchmarks import (
    BENCHMARK_B_SPEC_FILES,
    canonical_matched_b_contract,
    load_benchmark_b_reference,
    load_benchmark_b_spec,
)
from lmx.freemhd import (
    artifact_sha256,
    audit_freemhd_case_against_spec,
    candidate_u_paths,
    compare_side_jet_profiles,
    infer_inlet_drive_mode,
    infer_inlet_flow_rate,
    infer_liquid_material_properties,
    infer_rectangular_geometry,
    infer_solid_conductivities,
    infer_uniform_b0,
    load_benchmark_a_spec,
    load_samper_table_i,
    side_jet_profile_metrics,
    summarize_observable_gate,
    summarize_observable_offenders,
    validate_matched_b_record,
)
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
    for index, name in enumerate(("lmx_source", "freemhd_source", "lmx_input", "freemhd_input", "evaluator", "lmx_output", "freemhd_output")):
        kind = "tree" if index < 4 else "file"
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


_FLOW_RATE_U = """internalField   uniform ( 0.9725 0 0 );

boundaryField
{
    inlet
    {
        type flowRateInletVelocity;
        volumetricFlowRate 0.0389;
    }
}
"""


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


@pytest.mark.parametrize("case_kind", ["shercliff", "hunt"])
def test_freemhd_case_audit_accepts_only_mechanically_matched_inputs(tmp_path: Path, case_kind: str):
    report = audit_freemhd_case_against_spec(_materialize_matched_case(tmp_path, case_kind), case_kind=case_kind)

    assert report["matched"] is True
    assert report["failed_check_count"] == 0
    assert report["physical_hartmann_number"] == pytest.approx(20.0)


def test_freemhd_case_audit_exposes_mislabeled_ha_and_hunt_wall(tmp_path: Path):
    case = _materialize_matched_case(tmp_path, "hunt")
    (case / "0/liquid/B0").write_text("internalField uniform ( 0 10 0 );\n")
    (case / "constant/solidWalls/thermophysicalProperties").write_text("elcond 1e-6;\n")
    report = audit_freemhd_case_against_spec(case, case_kind="hunt")
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
    assert "contract.acceptance_role.unavailable" in report["failed_checks"]

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
        ("empty_tree", "provenance.lmx_input.content.empty"),
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
        artifacts["lmx_input"].update(path="empty", sha256="0" * 64)
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
    assert len(manifest["source_template_sha256"]) == 64
    assert (output / "lmx-benchmark-manifest.json").is_file()
    assert infer_uniform_b0(output) == pytest.approx((0.0, 0.2, 0.0))
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
    assert len([path for path in first.rglob("*") if path.is_file()]) == 5
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
    report = audit_freemhd_case_against_spec(tmp_path, case_kind="shercliff")
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


def test_side_jet_profile_metrics_and_comparison_capture_peak_locations():
    coordinate = [-1.0, -0.7, 0.0, 0.7, 1.0]
    reference = [0.0, 1.4, 1.0, 1.4, 0.0]
    simulated = [0.0, 1.2, 1.0, 1.3, 0.0]

    metrics = side_jet_profile_metrics(coordinate, reference)
    assert metrics["negative_location"] == pytest.approx(-0.7)
    assert metrics["positive_location"] == pytest.approx(0.7)
    assert metrics["peak_to_center_ratio"] == pytest.approx(1.4)

    comparison = compare_side_jet_profiles(coordinate, simulated, coordinate, reference)
    assert comparison["normalized_location_error"] == pytest.approx(0.0)
    assert comparison["peak_value_relative_error"] == pytest.approx((1.4 - 1.3) / 1.4)


def test_inlet_drive_mode_reads_case_zero(tmp_path: Path):
    case_zero = tmp_path / "case" / "0" / "liquid"
    case_zero.mkdir(parents=True)
    (case_zero / "U").write_text(_FLOW_RATE_U)
    assert infer_inlet_drive_mode(tmp_path) == "inlet_flow_rate"


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
    assert len(candidate_u_paths(tmp_path)) >= 6
    assert infer_uniform_b0(tmp_path) is None
    assert infer_rectangular_geometry(tmp_path) is None
    assert infer_solid_conductivities(tmp_path) == (None, None)

    incomplete_liquid = tmp_path / "case" / "constant" / "liquid"
    incomplete_liquid.mkdir(parents=True)
    (incomplete_liquid / "thermophysicalProperties").write_text("sigma 3.0;\nrho 1000;\n")
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


def test_freemhd_observable_gate_ranks_accuracy_and_completeness():
    observable_records = [
        {
            "case_kind": "shercliff",
            "drive_mode": "forcing",
            "observables": {
                "velocity": {
                    "y": {"l2_error": 2.0e-2, "linf_error": 5.0e-2},
                    "z": {"l2_error": 4.0e-3, "linf_error": 1.0e-2},
                    "peak_ratio": 0.95,
                },
                "current": {
                    "y": {"l2_error": 8.0e-2, "linf_error": 2.0e-1, "peak_ratio": 1.4},
                    "z": {"l2_error": 1.0e-2, "linf_error": 2.0e-2},
                    "peak_ratio": 1.1,
                },
                "potential": {
                    "y": {
                        "l2_error": 1.0,
                        "linf_error": 1.0,
                        "reference_peak_abs": 1.0e-8,
                    },
                    "z": {
                        "l2_error": 2.0e-2,
                        "linf_error": 5.0e-2,
                        "reference_peak_abs": 1.0,
                    },
                },
            },
        }
    ]
    observable_offenders = summarize_observable_offenders(observable_records, l2_target=1.0e-2)
    assert observable_offenders[0]["observable"] == "current"
    assert observable_offenders[0]["axis"] == "y"
    assert observable_offenders[0]["status"] == "offender"
    assert observable_offenders[-1]["status"] == "low_signal"

    observable_gate = summarize_observable_gate(observable_records, l2_target=1.0e-2)
    assert observable_gate["research_grade_validation_pass"] is False
    assert observable_gate["observable_offender_count"] == 3
    assert observable_gate["low_signal_count"] == 1
    assert observable_gate["missing_observable_count"] == 1
    assert observable_gate["missing_observables"] == [{"case_kind": "shercliff", "observable": "lorentz", "axis": "*"}]
