#!/usr/bin/env python3
"""Run all or selected LMX tests in parallel within a declared budget."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

_TEST_SHARDS = {
    "benchmarks": ("tests/test_benchmarks.py",),
    "core": (
        "tests/test_cli.py",
        "tests/test_config.py",
        "tests/test_io.py",
        "tests/test_mesh.py",
        "tests/test_runtime_logging.py",
        "tests/test_units_and_wall_models.py",
    ),
    "examples": ("tests/test_example_runner.py",),
    "physics": (
        "tests/test_fringing.py",
        "tests/test_physics.py",
        "tests/test_solver.py",
    ),
    "validation": (
        "tests/test_freemhd.py",
        "tests/test_run_benchmark_b_independence.py",
    ),
}


def _default_workers() -> int:
    return max(1, min(6, os.cpu_count() or 1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int)
    parser.add_argument("--budget-seconds", type=float, default=600.0)
    parser.add_argument("--warning-seconds", type=float, default=300.0)
    parser.add_argument("--test-timeout-seconds", type=float)
    parser.add_argument("--no-coverage", action="store_true")
    parser.add_argument("--shard", choices=tuple(_TEST_SHARDS))
    parser.add_argument("--coverage-fail-under", type=float, default=95.0)
    parser.add_argument("--coverage-xml", default="coverage.xml")
    parser.add_argument("--junit-xml", default="artifacts/tests/full-suite-junit.xml")
    parser.add_argument("tests", nargs="*", help="Test paths to run; defaults to the complete suite")
    args = parser.parse_args(argv)

    workers = args.workers
    if workers is None:
        workers = 1 if args.shard in {"benchmarks", "examples"} else _default_workers()
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
        f"--junitxml={junit_path}",
    ]
    if args.test_timeout_seconds is not None:
        command.append(f"--timeout={args.test_timeout_seconds:g}")
    if not args.no_coverage:
        command.extend(
            [
                "--cov=lmx",
                "--cov-branch",
                "--cov-report=term-missing:skip-covered",
                f"--cov-report=xml:{args.coverage_xml}",
                f"--cov-fail-under={args.coverage_fail_under}",
            ]
        )
    if not args.shard and not args.tests:
        command.extend(("-m", "not curated"))
    command.extend(_TEST_SHARDS[args.shard] if args.shard else args.tests or ["tests"])

    environment = os.environ.copy()
    environment.setdefault("MPLBACKEND", "Agg")
    environment.setdefault("JAX_ENABLE_X64", "true")
    environment.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    environment.setdefault("OMP_NUM_THREADS", "1")
    environment.setdefault("OPENBLAS_NUM_THREADS", "1")
    environment.setdefault("MKL_NUM_THREADS", "1")
    environment.setdefault("NUMEXPR_NUM_THREADS", "1")

    started = time.monotonic()
    print(
        f"LMX full test gate: workers={workers}, "
        f"budget={args.budget_seconds:.0f}s, coverage={not args.no_coverage}",
        flush=True,
    )
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
