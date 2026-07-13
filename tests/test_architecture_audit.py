from __future__ import annotations

import json
import ast
from pathlib import Path
import zipfile

import lmx

from scripts.audit_architecture import (
    _checkout_size,
    architecture_budget_errors,
    build_inventory,
    inspect_wheel,
    measure_import,
    write_inventory,
)


def test_tracked_architecture_baseline_matches_repository() -> None:
    tracked = json.loads(Path("provenance/architecture-baseline.json").read_text())
    current = build_inventory()
    assert tracked["inventory"] == current["inventory"]
    assert tracked["targets"] == current["targets"]
    assert architecture_budget_errors(current) == []


def test_architecture_inventory_is_deterministic_without_timing(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_inventory(first)
    write_inventory(second)
    assert first.read_bytes() == second.read_bytes()


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
