#!/usr/bin/env python3
"""Generate and verify LMX environment, feature, and benchmark manifests."""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
PROVENANCE_DIR = ROOT / "provenance"
ENVIRONMENT_PATH = PROVENANCE_DIR / "environment.json"
FEATURES_PATH = PROVENANCE_DIR / "features.json"
BENCHMARKS_PATH = PROVENANCE_DIR / "benchmarks.json"
SCHEMA_DIR = PROVENANCE_DIR / "schemas"


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative_paths(root: Path, pattern: str) -> list[str]:
    return sorted(
        path.relative_to(root).as_posix()
        for path in root.glob(pattern)
        if path.is_file()
    )


def build_environment_manifest(root: Path = ROOT) -> dict[str, Any]:
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    lock_path = root / "uv.lock"
    lock = tomllib.loads(lock_path.read_text(encoding="utf-8"))
    project = pyproject["project"]
    policy = pyproject["tool"]["lmx"]["provenance"]
    optional = project.get("optional-dependencies", {})
    coverage = pyproject["tool"]["coverage"]["report"]

    return {
        "schema_version": 1,
        "generated_by": "scripts/manage_provenance.py",
        "project": {"name": project["name"], "version": project["version"]},
        "python": {
            "requires": project["requires-python"],
            "ci_tested": sorted(policy["tested-python"]),
        },
        "dependencies": {
            "runtime": sorted(project.get("dependencies", [])),
            **{
                group: sorted(requirements)
                for group, requirements in sorted(optional.items())
            },
        },
        "lock": {
            "format": "uv",
            "path": "uv.lock",
            "sha256": _sha256(lock_path),
            "version": int(lock["version"]),
            "revision": int(lock["revision"]),
        },
        "numerical_policy": {"jax_enable_x64": bool(policy["jax-enable-x64"])},
        "portable_gate": {
            "command": "python scripts/run_full_test_suite.py",
            "budget_seconds": int(policy["full-gate-budget-seconds"]),
            "warning_seconds": int(policy["warning-budget-seconds"]),
            "default_workers": int(policy["default-workers"]),
            "branch_coverage_percent": float(coverage["fail_under"]),
        },
        "repository_inventory": {
            "modules": _relative_paths(root, "lmx/*.py"),
            "tests": _relative_paths(root, "tests/test_*.py"),
            "examples": sorted(
                _relative_paths(root, "examples/**/*.py")
                + _relative_paths(root, "examples/**/*.toml")
            ),
            "scripts": _relative_paths(root, "scripts/*.py"),
            "benchmark_specs": _relative_paths(root, "benchmarks/specs/*.toml"),
        },
    }


def _canonical_benchmark_manifest(
    payload: Mapping[str, Any], root: Path = ROOT
) -> dict[str, Any]:
    canonical = copy.deepcopy(dict(payload))
    for source in canonical.get("sources", []):
        relative = source.get("path")
        if relative:
            path = root / str(relative)
            if path.is_file():
                source["sha256"] = _sha256(path)
    return canonical


def _schema_errors(
    payload: Mapping[str, Any], schema_name: str, root: Path = ROOT
) -> list[str]:
    schema = _read_json(root / "provenance" / "schemas" / schema_name)
    validator = Draft202012Validator(schema)
    errors = sorted(
        validator.iter_errors(payload), key=lambda item: list(item.absolute_path)
    )
    rendered = []
    for error in errors:
        location = ".".join(str(piece) for piece in error.absolute_path) or "<root>"
        rendered.append(f"{schema_name}:{location}: {error.message}")
    return rendered


def _test_reference_error(reference: str, root: Path = ROOT) -> str | None:
    path_text, separator, node = reference.partition("::")
    path = root / path_text
    if not separator or not node:
        return f"test reference must use path::function: {reference}"
    if not path.is_file():
        return f"test file does not exist: {path_text}"
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=path_text)
    except SyntaxError as exc:
        return f"cannot parse {path_text}: {exc}"
    functions = {
        item.name
        for item in ast.walk(tree)
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    function = node.split("[", maxsplit=1)[0]
    if function not in functions:
        return f"test function does not exist: {reference}"
    return None


def _path_error(relative: str, root: Path = ROOT) -> str | None:
    if not (root / relative).is_file():
        return f"referenced file does not exist: {relative}"
    return None


def validate_feature_manifest(
    payload: Mapping[str, Any], root: Path = ROOT
) -> list[str]:
    errors = _schema_errors(payload, "features.schema.json", root)
    features = payload.get("features", [])
    ids = [feature.get("id") for feature in features]
    if len(ids) != len(set(ids)):
        errors.append("feature ids must be unique")

    mapped_modules: set[str] = set()
    for feature in features:
        mapped_modules.update(str(path) for path in feature.get("modules", []))
        for path in feature.get("modules", []):
            error = _path_error(str(path), root)
            if error:
                errors.append(error)
        for references in feature.get("tests", {}).values():
            for reference in references:
                error = _test_reference_error(str(reference), root)
                if error:
                    errors.append(error)
        for path in feature.get("workflows", []):
            error = _path_error(str(path), root)
            if error:
                errors.append(error)

    actual_modules = set(_relative_paths(root, "lmx/*.py"))
    for path in sorted(actual_modules - mapped_modules):
        errors.append(f"module is missing from the feature manifest: {path}")
    for path in sorted(mapped_modules - actual_modules):
        errors.append(f"feature manifest maps an unknown module: {path}")
    return errors


def validate_benchmark_manifest(
    payload: Mapping[str, Any], root: Path = ROOT
) -> list[str]:
    errors = _schema_errors(payload, "benchmarks.schema.json", root)
    sources = payload.get("sources", [])
    benchmarks = payload.get("benchmarks", [])
    source_ids = [source.get("id") for source in sources]
    benchmark_ids = [benchmark.get("id") for benchmark in benchmarks]
    if len(source_ids) != len(set(source_ids)):
        errors.append("benchmark source ids must be unique")
    if len(benchmark_ids) != len(set(benchmark_ids)):
        errors.append("benchmark ids must be unique")

    known_sources = set(source_ids)
    used_sources: set[str] = set()
    for source in sources:
        relative = source.get("path")
        if relative:
            path = root / str(relative)
            if not path.is_file():
                errors.append(f"benchmark source path does not exist: {relative}")
            elif source.get("sha256") != _sha256(path):
                errors.append(f"benchmark source checksum is stale: {relative}")
    for benchmark in benchmarks:
        spec_path = benchmark.get("spec_path")
        if spec_path:
            error = _path_error(str(spec_path), root)
            if error:
                errors.append(error)
        for source_id in benchmark.get("source_ids", []):
            used_sources.add(str(source_id))
            if source_id not in known_sources:
                errors.append(
                    f"benchmark {benchmark.get('id')} uses unknown source: {source_id}"
                )
        for reference in benchmark.get("tests", []):
            error = _test_reference_error(str(reference), root)
            if error:
                errors.append(error)
        for path in benchmark.get("workflows", []):
            error = _path_error(str(path), root)
            if error:
                errors.append(error)
    for source_id in sorted(known_sources - used_sources):
        errors.append(f"benchmark source is not used by any benchmark: {source_id}")
    return errors


def verify_external_literature(
    payload: Mapping[str, Any], literature_root: Path
) -> list[str]:
    errors: list[str] = []
    for source in payload.get("sources", []):
        filename = source.get("filename")
        expected = source.get("sha256")
        if not filename or not expected:
            continue
        path = literature_root / str(filename)
        if not path.is_file():
            errors.append(f"external literature file is missing: {filename}")
        elif _sha256(path) != expected:
            errors.append(f"external literature checksum mismatch: {filename}")
    return errors


def _document_state(
    root: Path = ROOT,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    environment = build_environment_manifest(root)
    features = _read_json(root / "provenance" / "features.json")
    benchmarks = _canonical_benchmark_manifest(
        _read_json(root / "provenance" / "benchmarks.json"), root
    )
    return environment, features, benchmarks


def write_manifests(root: Path = ROOT) -> list[str]:
    environment, features, benchmarks = _document_state(root)
    (root / "provenance" / "environment.json").write_text(
        _canonical_json(environment), encoding="utf-8"
    )
    (root / "provenance" / "features.json").write_text(
        _canonical_json(features), encoding="utf-8"
    )
    (root / "provenance" / "benchmarks.json").write_text(
        _canonical_json(benchmarks), encoding="utf-8"
    )
    return validate_manifests(root)


def validate_manifests(root: Path = ROOT) -> list[str]:
    environment, features, benchmarks = _document_state(root)
    errors = _schema_errors(environment, "environment.schema.json", root)
    errors.extend(validate_feature_manifest(features, root))
    errors.extend(validate_benchmark_manifest(benchmarks, root))
    return errors


def check_manifests(root: Path = ROOT) -> list[str]:
    environment, features, benchmarks = _document_state(root)
    expected = {
        root / "provenance" / "environment.json": _canonical_json(environment),
        root / "provenance" / "features.json": _canonical_json(features),
        root / "provenance" / "benchmarks.json": _canonical_json(benchmarks),
    }
    errors: list[str] = []
    for path, content in expected.items():
        if not path.is_file() or path.read_text(encoding="utf-8") != content:
            errors.append(
                f"stale provenance manifest: {path.relative_to(root)}; "
                "run `python scripts/manage_provenance.py --write`"
            )
    errors.extend(validate_manifests(root))
    return errors


def _check_uv_lock(root: Path = ROOT) -> list[str]:
    executable = shutil.which("uv")
    if executable is None:
        return ["uv is required to verify that uv.lock matches pyproject.toml"]
    completed = subprocess.run(
        [executable, "lock", "--check"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        return []
    detail = (completed.stderr or completed.stdout).strip()
    return [f"uv.lock is stale: {detail}"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--write", action="store_true", help="regenerate canonical manifests"
    )
    mode.add_argument(
        "--check", action="store_true", help="verify manifests without changing files"
    )
    parser.add_argument(
        "--external-literature-root",
        type=Path,
        help="also verify checksums of locally held, non-redistributed literature PDFs",
    )
    args = parser.parse_args(argv)

    try:
        errors = write_manifests(ROOT) if args.write else check_manifests(ROOT)
        errors.extend(_check_uv_lock(ROOT))
        if args.external_literature_root is not None:
            if not args.external_literature_root.is_dir():
                errors.append(
                    f"external literature root is not a directory: {args.external_literature_root}"
                )
            else:
                benchmarks = _read_json(BENCHMARKS_PATH)
                errors.extend(
                    verify_external_literature(
                        benchmarks, args.external_literature_root
                    )
                )
    except (
        KeyError,
        OSError,
        ValueError,
        json.JSONDecodeError,
        tomllib.TOMLDecodeError,
    ) as exc:
        errors = [str(exc)]

    if errors:
        print("LMX provenance verification failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1
    action = "regenerated" if args.write else "verified"
    print(f"LMX provenance manifests {action} successfully")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
