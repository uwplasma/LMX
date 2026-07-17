from __future__ import annotations

import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile

import lmx
import pytest

from scripts import manage_provenance as provenance
from scripts.audit_architecture import (
    _checkout_size,
    architecture_budget_errors,
    build_inventory,
    inspect_sdist,
    inspect_wheel,
    measure_import,
    write_inventory,
)
from scripts.manage_release_assets import (
    build_archive,
    build_manifest,
    check_manifest,
    verify_archive,
    write_animated_webp,
    write_static_webp,
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


def _animated_webp_duration_ms(path: Path) -> int:
    """Read the exact animation duration from WebP ANMF chunks."""

    data, offset, duration = path.read_bytes(), 12, 0
    while offset + 8 <= len(data):
        size = int.from_bytes(data[offset + 4 : offset + 8], "little")
        if data[offset : offset + 4] == b"ANMF":
            duration += int.from_bytes(data[offset + 20 : offset + 23], "little")
        offset += 8 + size + size % 2
    return duration


def test_architecture_inventory_is_deterministic_without_timing(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    write_inventory(first)
    write_inventory(second)
    assert first.read_bytes() == second.read_bytes()


def test_stable_root_api_is_small_lazy_and_resolvable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert set(lmx.__all__) == EXPECTED_ROOT_API
    assert EXPECTED_ROOT_API <= set(dir(lmx))
    assert all(callable(getattr(lmx, name)) for name in lmx.__all__)
    assert all(inspect.getdoc(getattr(lmx, name)) for name in lmx.__all__)

    updates = []
    monkeypatch.setattr("lmx.io.jax.config.update", lambda *args: updates.append(args))
    cache = lmx.enable_compilation_cache(
        tmp_path / "jax-cache", min_compile_time_secs=2.0, min_entry_size_bytes=4096
    )
    assert cache.is_dir()
    assert updates == [
        ("jax_compilation_cache_dir", str(cache)),
        ("jax_persistent_cache_min_entry_size_bytes", 4096),
        ("jax_persistent_cache_min_compile_time_secs", 2.0),
    ]


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


def test_numerical_modules_do_not_import_optional_visualization() -> None:
    code = """
import sys
import lmx.blanket_flow, lmx.blanket_geometry, lmx.centerline_fields
import lmx.plotting, lmx.q2d, lmx.showcase
assert not any(name == 'matplotlib' or name.startswith('matplotlib.') for name in sys.modules)
assert not any(name == 'PIL' or name.startswith('PIL.') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


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


def test_sdist_audit_rejects_repository_tests(tmp_path: Path) -> None:
    source = tmp_path / "lmx-1" / "tests" / "test_solver.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"large output")
    sdist = tmp_path / "lmx-test.tar.gz"
    with tarfile.open(sdist, "w:gz") as archive:
        archive.add(source, arcname="lmx-1/tests/test_solver.py")
    assert inspect_sdist(sdist)["forbidden_members"] == ["tests/test_solver.py"]
    assert "outside its source payload" in architecture_budget_errors(
        build_inventory(), sdist=sdist
    )[0]


def test_curated_examples_use_submodules_and_linear_scripts_are_editable() -> None:
    inventory = build_inventory()["inventory"]
    stable = set(lmx.__all__)
    for item in inventory["curated_examples"]:
        path = Path(item["path"])
        if path.suffix != ".py":
            continue
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        imports = (
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "lmx"
        )
        root_imports = {alias.name for node in imports for alias in node.names}
        assert root_imports <= stable, (
            f"{path} imports legacy root APIs: {root_imports - stable}"
        )
        if path.name in {
            "autodiff_design_demo.py",
            "extruded_restart_demo.py",
            "fringing_benchmark_demo.py",
            "hartmann_example.py",
            "hunt_example.py",
            "operator_verification_demo.py",
            "pipe_reference_comparison_demo.py",
            "variable_field_extruded_demo.py",
        }:
            assert ast.get_docstring(tree)
            assert "# Inputs:" in source and "# Run" in source and len(source.splitlines()) <= 160
            functions = (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
            assert all(node.name != "main" and ast.get_docstring(node) for node in functions)
            assert "argparse" not in source and "__name__" not in source


def test_curated_examples_declare_user_facing_contracts(tmp_path: Path) -> None:
    inventory = build_inventory()["inventory"]
    curated = inventory["curated_examples"]
    assert {item["path"] for item in curated} == set(inventory["examples"])
    assert len(curated) == 11
    for item in curated:
        assert item["command"]
        assert item["outputs"]
        assert item["runtime"] in {"portable", "external", "accelerator-optional"}
        assert Path(item["docs"]).is_file()
    script = Path(__file__).resolve().parents[1] / "examples/operator_verification_demo.py"
    subprocess.run([sys.executable, script], cwd=tmp_path, timeout=30, check=True)
    summary = next((tmp_path / "artifacts").rglob("operator_verification_summary.json"))
    assert json.loads(summary.read_text())["observed_order"]["gradient_y"] > 1.8

    autodiff = Path(__file__).resolve().parents[1] / "examples/autodiff_design_demo.py"
    subprocess.run([sys.executable, autodiff], cwd=tmp_path, timeout=30, check=True)
    design_path = next((tmp_path / "artifacts").rglob("autodiff_summary.json"))
    design = json.loads(design_path.read_text())
    assert design["recovered"]["forcing"] == pytest.approx(1.0, abs=0.02)
    assert design["recovered"]["loss"] < design["optimization_history"][0]["loss"] * 1.0e-3
    assert all((design_path.parent / name).is_file() for name in design["plots"])


def test_benchmark_provenance_is_current() -> None:
    assert provenance.check_manifests() == []
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
    assert tracked["schema_version"] == 2
    assert tracked["release"]["status"] == "uploaded"
    assert len(tracked["release"]["archive_sha256"]) == 64
    assert tracked["release"]["download_url"].startswith("https://github.com/")
    assert tracked["summary"]["logical_file_count"] > 0
    assert (
        tracked["summary"]["unique_content_count"]
        <= tracked["summary"]["logical_file_count"]
    )
    showcase = tracked["showcase"]
    assert showcase["bytes"] == sum(item["bytes"] for item in showcase["files"])
    assert showcase["bytes"] < 1280 * 1024
    assert len(showcase["files"]) <= 20
    animations = {item["path"]: item for item in tracked["animations"]}
    assert set(animations) == {
        "docs/_static/readme-hunt-startup.webp",
        "docs/_static/readme-blanket-flow.webp",
        "docs/_static/readme-q2d-turbulence.webp",
    }
    for path, metadata in animations.items():
        assert _animated_webp_duration_ms(Path(path)) == metadata["duration_ms"]
    hunt = animations["docs/_static/readme-hunt-startup.webp"]
    q2d = animations["docs/_static/readme-q2d-turbulence.webp"]
    assert not hunt["accepted"] and hunt["status"] == "legacy_transient"
    assert not q2d["accepted"] and q2d["status"] == "legacy_non_statistically_steady"
    assert q2d["duration_ms"] == 7014
    blanket = animations["docs/_static/readme-blanket-flow.webp"]
    assert blanket["accepted"] and blanket["status"] == "steady_window_accepted"
    acceptance = blanket["animation_acceptance"]
    assert acceptance["accepted"]
    assert acceptance["media_terminal"] == {
        "source_frame": 21,
        "step": 58,
        "time_s": 2.9,
        "window_max_relative_update": 0.001676904288502,
    }
    assert acceptance["full_source_terminal"]["window_max_relative_update"] < 1e-13
    assert all(
        len(item["sha256"]) == 64
        for item in acceptance["historical_evidence"].values()
        if isinstance(item, dict)
    )
    assert check_manifest() == tracked


def test_animated_webp_is_bounded_and_preserves_motion(tmp_path: Path) -> None:
    from PIL import Image

    source = tmp_path / "source.gif"
    frames = [Image.new("RGB", (12, 8), (value, 0, 0)) for value in (0, 80, 160)]
    frames[0].save(
        source,
        save_all=True,
        append_images=frames[1:],
        duration=80,
        loop=0,
    )
    output = write_animated_webp(
        source, tmp_path / "motion.webp", width=24, fps=2, sampling_power=1.7
    )
    with Image.open(output) as animation:
        assert animation.size == (24, 16)
        assert animation.n_frames >= 3
    assert 0 < _animated_webp_duration_ms(output) <= 7000
    assert output.stat().st_size < 16_000

    trimmed = write_animated_webp(
        source, tmp_path / "trimmed.webp", seconds=1.0, fps=2, last_frame=1
    )
    with Image.open(trimmed) as animation:
        animation.seek(animation.n_frames - 1)
        assert animation.getpixel((0, 0))[0] < 120

    paired = write_animated_webp(
        source,
        tmp_path / "paired.webp",
        seconds=1.0,
        fps=2,
        width=24,
        right_source=source,
    )
    with Image.open(paired) as animation:
        assert animation.size == (24, 8)

    with pytest.raises(ValueError, match="duration"):
        write_animated_webp(source, tmp_path / "too-long.webp", seconds=7.01)

    still = write_static_webp(source, tmp_path / "still.webp", width=6, quality=70)
    with Image.open(still) as image:
        assert image.size == (6, 4)
        assert image.n_frames == 1


def test_release_asset_archive_is_deterministic_and_verified(tmp_path: Path) -> None:
    root = tmp_path / "source"
    generated = root / "docs" / "_static" / "generated"
    generated.mkdir(parents=True)
    (generated / "large.bin").write_bytes(b"a" * (128 * 1024 + 1))
    poster = root / "docs" / "_static" / "poster.webp"
    poster.write_bytes(b"poster")
    manifest = write_manifest(tmp_path / "manifest.json", root)
    assert check_manifest(tmp_path / "manifest.json", root) == manifest
    poster.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Tracked showcase media"):
        check_manifest(tmp_path / "manifest.json", root)
    poster.write_bytes(b"poster")
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    assert build_archive(first, root) == build_archive(second, root)
    assert first.read_bytes() == second.read_bytes()
    verify_archive(first, manifest)
    manifest["release"].update(status="uploaded", archive_sha256="a" * 64)
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    poster.write_bytes(b"refreshed")
    refreshed = write_manifest(tmp_path / "manifest.json", root)
    assert refreshed["release"]["status"] == "uploaded" and refreshed["showcase"]["bytes"] == 9


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
