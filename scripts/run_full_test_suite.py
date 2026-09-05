#!/usr/bin/env python3
"""Run all or selected LMX tests in parallel within a declared budget."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time

_HEAVY_FRINGING_TEST = "test_alex_b1_production_map_has_bounded_implicit_gradient"
_TEST_SHARDS = {
    "support": (
        "tests/test_cli.py",
        "tests/test_config.py",
        "tests/test_io.py",
        "tests/test_mesh.py",
        "tests/test_runtime_logging.py",
        "tests/test_units_and_wall_models.py",
        "tests/test_freemhd.py",
        "tests/test_run_benchmark_b_independence.py",
        "tests/test_benchmarks.py",
        "tests/test_example_runner.py",
    ),
    "fringing": ("tests/test_fringing.py",),
    "physics": (
        "tests/test_physics.py",
        "tests/test_solver.py",
        f"tests/test_fringing.py::{_HEAVY_FRINGING_TEST}",
    ),
}

_ALL_TESTS = tuple(dict.fromkeys(path.split("::")[0] for shard in _TEST_SHARDS.values() for path in shard))
_CHANGE_TEST_NAMES = {
    "__init__": "config cli example_runner",
    "__main__": "cli",
    "cases": "config solver physics fringing benchmarks",
    "cli": "cli example_runner",
    "io": "io cli fringing freemhd example_runner",
    "mesh": "mesh solver physics fringing benchmarks",
    "physics": "solver physics fringing",
    "q2d": "physics example_runner",
    "solvers": "solver physics fringing",
    "specs": "config solver physics fringing benchmarks cli",
    "validation": "benchmarks freemhd physics solver example_runner",
}
_CHANGE_TESTS = {
    module: tuple(f"tests/test_{name}.py" for name in names.split())
    for module, names in _CHANGE_TEST_NAMES.items()
}
_FRINGING_TESTS = (
    "tests/test_fringing.py",
    "tests/test_benchmarks.py",
    "tests/test_freemhd.py",
    "tests/test_example_runner.py",
)
_NO_PYTHON_TEST_PREFIXES = ("docs/", ".github/")
_NO_PYTHON_TEST_FILES = {
    ".gitignore",
    "CITATION.cff",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "README.md",
    "plan.md",
}


def _changed_files(base: str) -> tuple[str, ...]:
    """Return committed, working-tree, and untracked paths relative to ``base``."""

    commands = (
        ("git", "diff", "--name-only", "--diff-filter=ACMRTUXB", f"{base}...HEAD"),
        ("git", "diff", "--name-only", "--diff-filter=ACMRTUXB", "HEAD"),
        ("git", "ls-files", "--others", "--exclude-standard"),
    )
    paths = []
    for command in commands:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        paths.extend(completed.stdout.splitlines())
    return tuple(dict.fromkeys(paths))


def _tests_for_changes(paths: tuple[str, ...]) -> tuple[str, ...]:
    """Select a conservative test set; unknown executable paths fail closed to all tests."""

    selected = []
    for path in paths:
        if path.startswith("tests/") and path.endswith(".py"):
            selected.append(path)
        elif path.startswith("src/lmx/data/benchmarks/"):
            selected.extend(("tests/test_benchmarks.py", "tests/test_freemhd.py"))
        elif path.startswith("src/lmx/") and path.endswith(".py"):
            module = path.removeprefix("src/lmx/").removesuffix(".py")
            if module == "fringing" or module.startswith("_fringing_"):
                selected.extend(_FRINGING_TESTS)
            elif module in _CHANGE_TESTS:
                selected.extend(_CHANGE_TESTS[module])
            else:
                return _ALL_TESTS
        elif path.startswith("examples/"):
            selected.append("tests/test_example_runner.py")
        elif path == "scripts/run_benchmark_b_independence.py":
            selected.append("tests/test_run_benchmark_b_independence.py")
        elif path in {"scripts/run_freemhd_parity_suite.py", "validation/freemhd.py"}:
            selected.append("tests/test_freemhd.py")
        elif path in {"scripts/audit_architecture.py", "scripts/run_full_test_suite.py"}:
            selected.append("tests/test_config.py")
        elif path in {"pyproject.toml", "MANIFEST.in"}:
            selected.append("tests/test_config.py")
        elif path in _NO_PYTHON_TEST_FILES or path.startswith(_NO_PYTHON_TEST_PREFIXES):
            continue
        else:
            return _ALL_TESTS
    return tuple(dict.fromkeys(selected))


def _test_environment() -> dict[str, str]:
    environment = os.environ.copy()
    source = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")
    environment["PYTHONPATH"] = os.pathsep.join(
        value for value in (source, environment.get("PYTHONPATH")) if value
    )
    return environment


def _default_workers() -> int:
    return max(1, min(6, os.cpu_count() or 1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--budget-seconds", type=float, default=600.0)
    parser.add_argument("--warning-seconds", type=float, default=300.0)
    parser.add_argument("--test-timeout-seconds", type=float)
    parser.add_argument("--no-coverage", action="store_true")
    parser.add_argument(
        "--changed-from",
        metavar="GIT_REF",
        help="run only tests affected since a Git ref; implies --no-coverage",
    )
    parser.add_argument("--no-compilation-cache", action="store_true")
    parser.add_argument("--shard", choices=tuple(_TEST_SHARDS))
    parser.add_argument("--coverage-fail-under", type=float, default=95.0)
    parser.add_argument("--coverage-xml", default="coverage.xml")
    parser.add_argument("--junit-xml", default="artifacts/tests/full-suite-junit.xml")
    parser.add_argument("tests", nargs="*", help="Test paths to run; defaults to the complete suite")
    args = parser.parse_args(argv)

    workers = args.workers
    if workers is None:
        workers = 1 if args.shard == "support" else _default_workers()
    if workers < 1:
        parser.error("--workers must be positive")
    if args.budget_seconds <= 0.0:
        parser.error("--budget-seconds must be positive")
    if args.warning_seconds <= 0.0:
        parser.error("--warning-seconds must be positive")
    if args.test_timeout_seconds is not None and args.test_timeout_seconds <= 0.0:
        parser.error("--test-timeout-seconds must be positive")
    if not 0.0 <= args.coverage_fail_under <= 100.0:
        parser.error("--coverage-fail-under must be between 0 and 100")
    if args.shard and args.tests:
        parser.error("--shard cannot be combined with explicit test paths")
    if args.changed_from and (args.shard or args.tests):
        parser.error("--changed-from cannot be combined with --shard or explicit test paths")

    selected_tests = (
        _tests_for_changes(_changed_files(args.changed_from)) if args.changed_from else args.tests
    )
    if args.changed_from and not selected_tests:
        print(f"LMX change gate: no Python tests affected since {args.changed_from}")
        return 0

    junit_path = os.path.abspath(args.junit_xml)
    os.makedirs(os.path.dirname(junit_path), exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "pytest",
        "-n",
        str(workers),
        "--dist",
        "worksteal",
        "--durations=20",
        f"--junitxml={junit_path}",
    ]
    if args.test_timeout_seconds is not None:
        command.append(f"--timeout={args.test_timeout_seconds:g}")
    coverage = not args.no_coverage and not args.changed_from
    if coverage:
        command.extend(
            [
                "--cov=lmx",
                "--cov-branch",
                "--cov-report=term-missing:skip-covered",
                f"--cov-report=xml:{args.coverage_xml}",
                f"--cov-fail-under={args.coverage_fail_under}",
            ]
        )
    if not args.shard and not selected_tests:
        command.extend(("-m", "not curated"))
    command.extend(_TEST_SHARDS[args.shard] if args.shard else selected_tests or ["tests"])
    if args.shard == "fringing":
        command.extend(("-k", f"not {_HEAVY_FRINGING_TEST}"))

    environment = _test_environment()
    environment.setdefault("MPLBACKEND", "Agg")
    environment.setdefault("JAX_ENABLE_X64", "true")
    environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    environment.setdefault("NUMEXPR_NUM_THREADS", "1")
    if args.no_compilation_cache:
        environment["JAX_ENABLE_COMPILATION_CACHE"] = "false"
        environment.pop("JAX_COMPILATION_CACHE_DIR", None)
    else:
        environment.setdefault(
            "JAX_COMPILATION_CACHE_DIR",
            os.path.join(tempfile.gettempdir(), "lmx-jax-cache"),
        )

    started = time.monotonic()
    print(
        f"LMX full test gate: workers={workers}, budget={args.budget_seconds:.0f}s, coverage={coverage}",
        flush=True,
    )
    if args.changed_from:
        print(f"LMX change gate selected: {', '.join(selected_tests)}", flush=True)
    try:
        completed = subprocess.run(
            command,
            env=environment,
            check=False,
            timeout=args.budget_seconds,
        )
    except subprocess.TimeoutExpired:
        elapsed = time.monotonic() - started
        print(
            f"LMX full test gate exceeded its {args.budget_seconds:.0f}s budget after {elapsed:.1f}s",
            file=sys.stderr,
        )
        return 124

    elapsed = time.monotonic() - started
    print(f"LMX full test gate completed in {elapsed:.1f}s", flush=True)
    if elapsed > args.warning_seconds:
        print(
            f"LMX full test gate exceeded its {args.warning_seconds:.0f}s warning budget",
            file=sys.stderr,
        )
    return int(completed.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
