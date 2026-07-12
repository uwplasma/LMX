from __future__ import annotations

import argparse
import json
from pathlib import Path

from lmx.scaling import (
    benchmark_extruded_inductionless_solve,
    benchmark_sharded_extruded_operator,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a single strong-scaling benchmark worker."
    )
    parser.add_argument(
        "--benchmark-kind",
        choices=("extruded3d", "extruded_solve"),
        default="extruded3d",
    )
    parser.add_argument("--nx", type=int, default=384)
    parser.add_argument("--ny", type=int, default=1024)
    parser.add_argument("--nz", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--num-devices", type=int, required=True)
    parser.add_argument("--platform", type=str, default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, default=None)
    args = parser.parse_args(argv)

    if args.benchmark_kind == "extruded_solve":
        record = benchmark_extruded_inductionless_solve(
            nx=args.nx,
            ny=args.ny,
            nz=args.nz,
            max_steps=args.iterations,
            repeats=args.repeats,
            num_devices=args.num_devices,
            profile_dir=args.profile_dir,
        )
    elif args.benchmark_kind == "extruded3d":
        record = benchmark_sharded_extruded_operator(
            nx=args.nx,
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
