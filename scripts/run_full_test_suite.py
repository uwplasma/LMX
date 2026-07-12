#!/usr/bin/env python3
"""Run the complete parallel LMX test and branch-coverage gate within a budget."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


def _default_workers() -> int:
    return max(1, min(4, os.cpu_count() or 1))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=_default_workers())
    parser.add_argument("--budget-seconds", type=float, default=600.0)
    parser.add_argument("--warning-seconds", type=float, default=450.0)
    parser.add_argument("--no-coverage", action="store_true")
    parser.add_argument("--coverage-xml", default="coverage.xml")
    parser.add_argument("--junit-xml", default="artifacts/tests/full-suite-junit.xml")
    args = parser.parse_args(argv)

    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.budget_seconds <= 0.0:
        parser.error("--budget-seconds must be positive")
    if args.warning_seconds <= 0.0:
        parser.error("--warning-seconds must be positive")

    junit_path = os.path.abspath(args.junit_xml)
    os.makedirs(os.path.dirname(junit_path), exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "pytest",
        "-n",
        str(args.workers),
        "--dist",
        "worksteal",
        f"--junitxml={junit_path}",
    ]
    if not args.no_coverage:
        command.extend(
            [
                "--cov=lmx",
                "--cov-branch",
                "--cov-report=term-missing:skip-covered",
                f"--cov-report=xml:{args.coverage_xml}",
                "--cov-fail-under=95",
            ]
        )

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
        f"LMX full test gate: workers={args.workers}, "
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
        print(f"LMX full test gate exceeded its {args.budget_seconds:.0f}s budget after {elapsed:.1f}s", file=sys.stderr)
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
