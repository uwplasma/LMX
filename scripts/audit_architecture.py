#!/usr/bin/env python3
"""Inventory the LMX architecture and enforce the M2 slimming baseline."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    ".pytest_cache",
    "__pycache__",
    "artifacts",
    "_build",
}
EXCLUDED_FILES = {".coverage", "coverage.xml"}
EXCLUDED_RELATIVE_FILES = {"provenance/architecture-baseline.json"}

RESEARCH_STAGE = {
    "_autodiff.py",
    "_fringing.py",
    "fringing.py",
    "_fringing_types.py",
    "q2d.py",
    "blanket_flow.py",
    "blanket_geometry.py",
    "centerline_fields.py",
    "dean.py",
    "scaling.py",
    "wall_study.py",
}
COMPATIBILITY = {
    "autodiff.py",
    "fringing.py",
    "plotting.py",
    "solvers.py",
    "validation.py",
}
VISUALIZATION = {"_plotting.py", "research_figures.py", "showcase.py"}
VALIDATION = {
    "_validation.py",
    "external_validation.py",
    "freemhd.py",
    "publication.py",
    "reference_data.py",
    "research_blockers.py",
    "research_closure.py",
}


def _role(name: str) -> str:
    if name in COMPATIBILITY:
        return "compatibility_facade"
    if name in RESEARCH_STAGE:
        return "research_stage_extension"
    if name in VISUALIZATION:
        return "visualization"
    if name in VALIDATION:
        return "validation_and_evidence"
    return "stable_core"


def _root_exports(root: Path) -> list[str]:
    tree = ast.parse((root / "lmx" / "__init__.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        ):
            exports = ast.literal_eval(node.value)
            if not isinstance(exports, list) or not all(
                isinstance(item, str) for item in exports
            ):
                break
            return exports
    raise ValueError("lmx.__all__ must be a literal list for architecture auditing")


def _checkout_size(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if (
            not path.is_file()
            or path.name in EXCLUDED_FILES
            or path.name.startswith(".coverage.")
        ):
            continue
        if relative.as_posix() in EXCLUDED_RELATIVE_FILES:
            continue
        if any(part in EXCLUDED_PARTS for part in relative.parts):
            continue
        total += path.stat().st_size
    return total


def _release_asset_candidates(root: Path) -> list[dict[str, Any]]:
    candidates = []
    for path in root.rglob("*"):
        if not path.is_file() or path.stat().st_size <= 128 * 1024:
            continue
        relative = path.relative_to(root)
        generated_doc = relative.parts[:3] == ("docs", "_static", "generated")
        generated_study = (
            len(relative.parts) >= 4
            and relative.parts[0] == "studies"
            and ("results" in relative.parts or "figures" in relative.parts)
        )
        if not (generated_doc or generated_study):
            continue
        candidates.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "disposition": "github_or_zenodo_release",
            }
        )
    return sorted(candidates, key=lambda item: item["path"])


def build_inventory(root: Path = ROOT) -> dict[str, Any]:
    modules = []
    for path in sorted((root / "lmx").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        role = _role(path.name)
        modules.append(
            {
                "path": path.relative_to(root).as_posix(),
                "lines": len(text.splitlines()),
                "bytes": path.stat().st_size,
                "role": role,
                "owner": role,
            }
        )
    examples = sorted(
        path.relative_to(root).as_posix()
        for pattern in ("*.py", "*.toml")
        for path in (root / "examples").glob(pattern)
        if path.is_file() and path.name != "catalog.toml"
    )
    catalog_path = root / "examples" / "catalog.toml"
    catalog = tomllib.loads(catalog_path.read_text(encoding="utf-8"))
    if catalog.get("schema_version") != 2:
        raise ValueError("examples/catalog.toml must use schema_version = 2")
    curated = catalog.get("example", [])
    if not isinstance(curated, list) or not curated:
        raise ValueError("examples/catalog.toml must define at least one [[example]]")
    curated_paths = [str(item["path"]) for item in curated]
    allowed_example_statuses = {"stable", "research-stage", "external-data"}
    allowed_runtime_tiers = {"portable", "external", "accelerator-optional"}
    for item in curated:
        if item.get("status") not in allowed_example_statuses:
            raise ValueError(f"Unknown curated workflow status: {item.get('status')!r}")
        if item.get("runtime") not in allowed_runtime_tiers:
            raise ValueError(
                f"Unknown curated workflow runtime: {item.get('runtime')!r}"
            )
        if not item.get("command") or not item.get("outputs") or not item.get("docs"):
            raise ValueError("Curated workflows need command, outputs, and docs")
        if not (root / str(item["docs"])).is_file():
            raise ValueError(f"Curated workflow docs do not exist: {item['docs']}")
    missing_curated = [path for path in curated_paths if not (root / path).is_file()]
    if missing_curated:
        raise ValueError(f"Curated examples do not exist: {missing_curated}")
    dispositions = catalog.get("disposition", [])
    if not isinstance(dispositions, list) or not dispositions:
        raise ValueError("examples/catalog.toml must classify uncurated workflows")
    allowed_actions = {"merge_into_curated", "move_to_campaigns", "move_to_cases"}
    disposition_paths: list[str] = []
    pending_disposition_paths: list[str] = []
    for item in dispositions:
        if item.get("action") not in allowed_actions:
            raise ValueError(
                f"Unknown workflow disposition action: {item.get('action')!r}"
            )
        if not item.get("id") or not item.get("target") or not item.get("reason"):
            raise ValueError("Each workflow disposition needs id, target, and reason")
        status = item.get("status", "pending")
        if status not in {"pending", "complete"}:
            raise ValueError(f"Unknown workflow disposition status: {status!r}")
        paths = item.get("paths")
        if not isinstance(paths, list) or not paths:
            raise ValueError(f"Workflow disposition {item.get('id')!r} has no paths")
        paths = [str(path) for path in paths]
        disposition_paths.extend(paths)
        if status == "pending":
            pending_disposition_paths.extend(paths)
        else:
            action = str(item["action"])
            target = root / str(item["target"])
            for source in paths:
                if (root / source).exists():
                    raise ValueError(
                        f"Completed workflow source still exists: {source}"
                    )
                if action in {"move_to_campaigns", "move_to_cases"}:
                    destination = target / Path(source).name
                    if not destination.is_file():
                        raise ValueError(
                            f"Completed workflow destination is missing: {destination}"
                        )
            if action == "merge_into_curated" and not target.is_file():
                raise ValueError(
                    f"Completed workflow merge target is missing: {target}"
                )
    duplicate_dispositions = sorted(
        path for path in set(disposition_paths) if disposition_paths.count(path) > 1
    )
    if duplicate_dispositions:
        raise ValueError(
            f"Workflows have multiple dispositions: {duplicate_dispositions}"
        )
    overlap = sorted(set(curated_paths) & set(pending_disposition_paths))
    if overlap:
        raise ValueError(f"Curated workflows also have dispositions: {overlap}")
    classified = set(curated_paths) | set(pending_disposition_paths)
    unclassified = sorted(set(examples) - classified)
    stale = sorted(classified - set(examples))
    if unclassified or stale:
        raise ValueError(
            f"Workflow catalog drift: unclassified={unclassified}, stale={stale}"
        )
    exports = _root_exports(root)
    release_candidates = _release_asset_candidates(root)
    checkout_bytes = _checkout_size(root)
    pyproject = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    project = pyproject["project"]
    return {
        "schema_version": 1,
        "inventory": {
            "package_module_count": len(modules),
            "total_package_lines": sum(module["lines"] for module in modules),
            "maintained_core_lines": sum(
                module["lines"]
                for module in modules
                if module["role"] in {"stable_core", "compatibility_facade"}
            ),
            "package_modules": modules,
            "root_export_count": len(exports),
            "root_exports": exports,
            "example_count": len(examples),
            "examples": examples,
            "curated_example_count": len(curated_paths),
            "curated_examples": curated,
            "uncurated_example_count": len(set(examples) - set(curated_paths)),
            "workflow_disposition_count": len(disposition_paths),
            "pending_workflow_disposition_count": len(pending_disposition_paths),
            "completed_workflow_disposition_count": len(disposition_paths)
            - len(pending_disposition_paths),
            "workflow_dispositions": dispositions,
            "checkout_bytes_excluding_build_artifacts": checkout_bytes,
            "release_asset_candidate_bytes": sum(
                item["bytes"] for item in release_candidates
            ),
            "checkout_bytes_excluding_release_candidates": checkout_bytes
            - sum(item["bytes"] for item in release_candidates),
            "release_asset_candidates": release_candidates,
            "dependencies": {
                "runtime": sorted(project.get("dependencies", [])),
                "optional": {
                    key: sorted(value)
                    for key, value in sorted(
                        project.get("optional-dependencies", {}).items()
                    )
                },
            },
        },
        "targets": {
            "maintained_core_lines_max": 15000,
            "stable_root_exports_max": 30,
            "curated_examples_max": 20,
            "checkout_bytes_max": 10 * 1024 * 1024,
            "new_module_lines_max": 1000,
        },
    }


def measure_import(root: Path = ROOT, repeats: int = 5) -> dict[str, Any]:
    samples = []
    command = [sys.executable, "-c", "import lmx"]
    for _ in range(repeats):
        started = time.perf_counter()
        completed = subprocess.run(command, cwd=root, capture_output=True, check=False)
        elapsed = time.perf_counter() - started
        if completed.returncode != 0:
            raise ValueError(completed.stderr.decode(errors="replace"))
        samples.append(elapsed)
    return {
        "command": "python -c 'import lmx'",
        "repeats": repeats,
        "median_seconds": statistics.median(samples),
        "min_seconds": min(samples),
        "max_seconds": max(samples),
    }


def write_inventory(
    output: Path, *, root: Path = ROOT, measure: bool = False
) -> dict[str, Any]:
    payload = build_inventory(root)
    if measure:
        payload["import_measurement"] = measure_import(root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("provenance/architecture-baseline.json"),
    )
    parser.add_argument("--measure-import", action="store_true")
    args = parser.parse_args()
    payload = write_inventory(args.output, measure=args.measure_import)
    inventory = payload["inventory"]
    print(
        f"modules={inventory['package_module_count']} "
        f"core_lines={inventory['maintained_core_lines']} "
        f"total_lines={inventory['total_package_lines']} "
        f"exports={inventory['root_export_count']} "
        f"curated_examples={inventory['curated_example_count']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
