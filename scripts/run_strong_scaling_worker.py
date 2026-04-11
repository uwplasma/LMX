from __future__ import annotations

import argparse
import json
from pathlib import Path

from lmx.scaling import benchmark_sharded_stencil


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a single strong-scaling benchmark worker.")
    parser.add_argument("--ny", type=int, default=1024)
    parser.add_argument("--nz", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--num-devices", type=int, required=True)
    parser.add_argument("--platform", type=str, default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    record = benchmark_sharded_stencil(
        ny=args.ny,
        nz=args.nz,
        iterations=args.iterations,
        repeats=args.repeats,
        num_devices=args.num_devices,
    )
    payload = {
        **record.__dict__,
        "platform": args.platform,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
