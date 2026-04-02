#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lmx.example_runner import run_case_example_cli


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Shercliff example and generate publication-ready plots.")
    parser.add_argument("--ha", type=float, default=20.0)
    parser.add_argument("--ny", type=int, default=64)
    parser.add_argument("--nz", type=int, default=64)
    parser.add_argument("--reference-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("./artifacts/examples/shercliff"))
    args = parser.parse_args(argv)
    return run_case_example_cli(
        case_kind="shercliff",
        ha=args.ha,
        ny=args.ny,
        nz=args.nz,
        out_dir=args.output,
        reference_root=args.reference_root,
    )


if __name__ == "__main__":
    raise SystemExit(main())
