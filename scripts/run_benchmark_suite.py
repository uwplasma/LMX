from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lmx.benchmarks import benchmark_solver, write_benchmark_report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the LMX benchmark suite.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmarks/benchmark.json"))
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--ha", type=float, default=20.0)
    parser.add_argument("--ny", type=int, default=48)
    parser.add_argument("--nz", type=int, default=48)
    args = parser.parse_args()

    report = benchmark_solver(repeats=args.repeats, ha=args.ha, ny=args.ny, nz=args.nz)
    write_benchmark_report(report, args.output)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
