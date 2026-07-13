from __future__ import annotations

# ruff: noqa: E402 -- repository-root bootstrap must precede project imports.

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lmx
from lmx.scaling import (
    benchmark_extruded_inductionless_solve,
    benchmark_sharded_extruded_operator,
)

if ROOT not in Path(lmx.__file__).resolve().parents:
    raise RuntimeError(
        f"Scaling worker imported LMX outside its source tree: {lmx.__file__}"
    )


def _source_fingerprint() -> str:
    """Hash the source and frozen specifications used by this worker."""

    paths = [*sorted((ROOT / "lmx").glob("*.py")), Path(__file__).resolve()]
    paths.extend(
        sorted(
            path
            for path in (ROOT / "benchmarks" / "specs").rglob("*")
            if path.is_file()
        )
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


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
    parser.add_argument(
        "--restart",
        type=Path,
        default=None,
        help="Verified extruded restart used to initialize solver-faithful timing.",
    )
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
            restart_path=args.restart,
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
        "source_fingerprint": _source_fingerprint(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
