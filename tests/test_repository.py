from __future__ import annotations

import ast
import copy
import hashlib
import json
from pathlib import Path
import zipfile

import lmx
import pytest

from scripts import manage_provenance as provenance
from scripts.audit_architecture import (
    _checkout_size,
    architecture_budget_errors,
    build_inventory,
    inspect_wheel,
    measure_import,
    write_inventory,
)
from scripts.manage_release_assets import (
    build_archive,
    build_manifest,
    check_manifest,
    verify_archive,
    write_manifest,
)


EXPECTED_ROOT_API = {
    "enable_compilation_cache",
    "make_hartmann_case",
    "make_shercliff_case",
    "make_hunt_case",
    "solve_steady",
    "solve_transient",
    "fully_developed_power_balance",
    "generate_rect_duct_mesh",
    "generate_rect_duct_mesh_from_faces",
    "generate_layered_duct_mesh",
    "generate_layered_duct_mesh_from_fluid_faces",
    "generate_multilayer_duct_mesh",
    "WallLayer",
    "dynamic_to_kinematic_viscosity",
    "kinematic_to_dynamic_viscosity",
    "hartmann_number",
    "reynolds_number",
    "interaction_parameter",
    "magnetic_reynolds_number",
    "magnetic_field_from_hartmann",
    "wall_conductance_ratio",
    "effective_pinhole_conductance_ratio",
    "tangential_stack_conductance_ratio",
    "normal_stack_leakage_ratio",
    "equivalent_single_layer",
    "nested_wall_layer_resolution_summary",
    "load_shercliff_analytical",
    "load_hunt_analytical",
    "load_closed_channel_analytical",
    "load_processed_slice",
}


def test_architecture_inventory_is_deterministic_without_timing(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_inventory(first)
    write_inventory(second)
    assert first.read_bytes() == second.read_bytes()


def test_stable_root_api_is_small_lazy_and_resolvable() -> None:
    assert set(lmx.__all__) == EXPECTED_ROOT_API
    assert EXPECTED_ROOT_API <= set(dir(lmx))
    assert all(callable(getattr(lmx, name)) for name in lmx.__all__)


def test_advanced_api_uses_owning_module() -> None:
    assert not hasattr(lmx, "solve_extruded_inductionless")
    from lmx.fringing import solve_extruded_inductionless

    assert callable(solve_extruded_inductionless)


def test_unknown_root_attribute_has_standard_error() -> None:
    with pytest.raises(AttributeError, match="not_an_api"):
        lmx.not_an_api


def test_architecture_inventory_ignores_generated_egg_info(tmp_path: Path) -> None:
    (tmp_path / "source.py").write_bytes(b"x")
    metadata = tmp_path / "package.egg-info"
    metadata.mkdir()
    (metadata / "PKG-INFO").write_bytes(b"generated")
    assert _checkout_size(tmp_path) == 1


def test_root_import_is_lazy_and_within_budget() -> None:
    payload = build_inventory()
    payload["import_measurement"] = measure_import(repeats=3)
    assert architecture_budget_errors(payload) == []


def test_wheel_audit_rejects_nonpackage_payload(tmp_path: Path) -> None:
    wheel = tmp_path / "lmx-test.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("lmx/__init__.py", "")
        archive.writestr("lmx-1.dist-info/METADATA", "")
        archive.writestr("benchmarks/raw.bin", b"large output")
    assert inspect_wheel(wheel)["forbidden_members"] == ["benchmarks/raw.bin"]
    assert "outside lmx/" in architecture_budget_errors(
        build_inventory(), wheel=wheel
    )[0]


def test_every_workflow_is_curated() -> None:
    inventory = build_inventory()["inventory"]
    curated = {item["path"] for item in inventory["curated_examples"]}
    assert curated == set(inventory["examples"])


def test_curated_examples_use_submodules_for_advanced_apis() -> None:
    inventory = build_inventory()["inventory"]
    stable = set(lmx.__all__)
    for item in inventory["curated_examples"]:
        path = Path(item["path"])
        if path.suffix != ".py":
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        root_imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "lmx"
            for alias in node.names
        }
        assert root_imports <= stable, (
            f"{path} imports legacy root APIs: {root_imports - stable}"
        )


def test_curated_examples_declare_user_facing_contracts() -> None:
    curated = build_inventory()["inventory"]["curated_examples"]
    assert len(curated) == 11
    for item in curated:
        assert item["command"]
        assert item["outputs"]
        assert item["runtime"] in {"portable", "external", "accelerator-optional"}
        assert Path(item["docs"]).is_file()


def test_benchmark_provenance_and_lock_are_current() -> None:
    assert provenance.check_manifests() == []
    assert provenance._check_uv_lock() == []
    payload = provenance._read_json(provenance.BENCHMARKS_PATH)
    assert provenance.validate_benchmark_manifest(payload) == []
    statuses = {item["id"]: item["status"] for item in payload["benchmarks"]}
    assert statuses == {
        "A1-hartmann": "verified-bounded",
        "A2-shercliff": "verified-bounded",
        "A2-hunt": "verified-bounded",
        "A3-high-ha": "verified-bounded",
        "B1-fringing-pipe": "specification-frozen",
        "B2-fringing-square": "specification-frozen",
    }

    missing = provenance._test_reference_error("tests/test_solver.py::test_not_present")
    malformed = provenance._test_reference_error("tests/test_solver.py")
    assert missing == "test function does not exist: tests/test_solver.py::test_not_present"
    assert malformed == "test reference must use path::function: tests/test_solver.py"
    assert provenance.validate_benchmark_manifest(
        {"schema_version": 0, "sources": [], "benchmarks": []}
    ) == [
        "benchmark manifest schema_version must be 1",
        "benchmark manifest requires at least one source",
    ]


def test_external_literature_verifier_reports_drift(tmp_path: Path) -> None:
    content = b"independent literature artifact"
    payload = {
        "sources": [
            {
                "filename": "present.pdf",
                "sha256": hashlib.sha256(content).hexdigest(),
            },
            {"filename": "missing.pdf", "sha256": "0" * 64},
        ]
    }
    (tmp_path / "present.pdf").write_bytes(content)
    assert provenance.verify_external_literature(payload, tmp_path) == [
        "external literature file is missing: missing.pdf"
    ]
    (tmp_path / "present.pdf").write_bytes(b"changed")
    assert "external literature checksum mismatch: present.pdf" in (
        provenance.verify_external_literature(payload, tmp_path)
    )


def test_tracked_release_asset_manifest_matches_sources() -> None:
    tracked = json.loads(Path("docs/release-assets.json").read_text())
    assert tracked["release"]["status"] == "uploaded"
    assert len(tracked["release"]["archive_sha256"]) == 64
    assert tracked["release"]["download_url"].startswith("https://github.com/")
    assert tracked["summary"]["logical_file_count"] > 0
    assert (
        tracked["summary"]["unique_content_count"]
        <= tracked["summary"]["logical_file_count"]
    )
    assert check_manifest() == tracked


def test_release_asset_archive_is_deterministic_and_verified(tmp_path: Path) -> None:
    root = tmp_path / "source"
    generated = root / "docs" / "_static" / "generated"
    generated.mkdir(parents=True)
    (generated / "large.bin").write_bytes(b"a" * (128 * 1024 + 1))
    manifest = write_manifest(tmp_path / "manifest.json", root)
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    assert build_archive(first, root) == build_archive(second, root)
    assert first.read_bytes() == second.read_bytes()
    verify_archive(first, manifest)


def test_release_asset_manifest_detects_drift(tmp_path: Path) -> None:
    root = tmp_path / "source"
    generated = root / "docs" / "_static" / "generated"
    generated.mkdir(parents=True)
    asset = generated / "large.bin"
    asset.write_bytes(b"a" * (128 * 1024 + 1))
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest_path, root)
    asset.write_bytes(b"b" * (128 * 1024 + 1))
    with pytest.raises(ValueError, match="source drift"):
        check_manifest(manifest_path, root)

    invalid = copy.deepcopy(build_manifest(root))
    invalid["release"]["status"] = "uploaded"
    manifest_path.write_text(json.dumps(invalid))
    with pytest.raises(ValueError, match="archive_sha256"):
        check_manifest(manifest_path, root)
