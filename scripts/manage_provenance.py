#!/usr/bin/env python3
"""Generate and verify the compact LMX benchmark provenance manifest."""

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

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS_PATH = ROOT / "benchmarks" / "provenance.json"


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


def validate_benchmark_manifest(
    payload: Mapping[str, Any], root: Path = ROOT
) -> list[str]:
    errors: list[str] = []
    sources = payload.get("sources", [])
    benchmarks = payload.get("benchmarks", [])
    if payload.get("schema_version") != 1:
        errors.append("benchmark manifest schema_version must be 1")
    if not isinstance(sources, list) or not sources:
        return [*errors, "benchmark manifest requires at least one source"]
    if not isinstance(benchmarks, list) or not benchmarks:
        return [*errors, "benchmark manifest requires at least one benchmark"]
    if not all(isinstance(item, dict) for item in sources + benchmarks):
        return [*errors, "benchmark manifest entries must be JSON objects"]
    source_required = {"id", "kind", "title", "availability"}
    benchmark_required = {
        "id",
        "tier",
        "title",
        "status",
        "source_ids",
        "primary_observables",
        "acceptance",
        "tests",
        "workflows",
        "runtime_lane",
    }
    for index, source in enumerate(sources):
        missing = source_required - set(source)
        if missing:
            errors.append(f"source {index} lacks: {', '.join(sorted(missing))}")
        if source.get("availability") not in {"repository", "external", "missing"}:
            errors.append(f"source {index} has an invalid availability")
    for index, benchmark in enumerate(benchmarks):
        missing = benchmark_required - set(benchmark)
        if missing:
            errors.append(f"benchmark {index} lacks: {', '.join(sorted(missing))}")
        if benchmark.get("tier") not in {"A", "B", "C", "D", "E"}:
            errors.append(f"benchmark {index} has an invalid tier")
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


def _document_state(root: Path = ROOT) -> dict[str, Any]:
    return _canonical_benchmark_manifest(
        _read_json(root / "benchmarks" / "provenance.json"), root
    )


def write_manifests(root: Path = ROOT) -> list[str]:
    benchmarks = _document_state(root)
    (root / "benchmarks" / "provenance.json").write_text(
        _canonical_json(benchmarks), encoding="utf-8"
    )
    return validate_manifests(root)


def validate_manifests(root: Path = ROOT) -> list[str]:
    return validate_benchmark_manifest(_document_state(root), root)


def check_manifests(root: Path = ROOT) -> list[str]:
    benchmarks = _document_state(root)
    expected = {
        root / "benchmarks" / "provenance.json": _canonical_json(benchmarks),
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
