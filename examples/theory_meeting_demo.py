#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lmx.example_runner import run_theory_meeting_demo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a polished LMX meeting demo with steady comparison plots and Hunt startup movies."
    )
    parser.add_argument("--output", type=Path, default=Path("./artifacts/examples/theory_meeting_demo"))
    parser.add_argument("--resolution", type=int, default=32)
    parser.add_argument("--hunt-resolution", type=int, default=24)
    parser.add_argument("--hunt-dt", type=float, default=5e-6)
    parser.add_argument("--hunt-t-final", type=float, default=8e-5)
    parser.add_argument("--hunt-frames", type=int, default=6)
    parser.add_argument("--hartmann-ha", type=float, default=20.0)
    parser.add_argument("--shercliff-ha", type=float, default=20.0)
    parser.add_argument("--hunt-ha", type=float, default=20.0)
    parser.add_argument("--reference-root", type=Path, default=None)
    args = parser.parse_args(argv)

    report = run_theory_meeting_demo(
        out_dir=args.output,
        hartmann_ha=args.hartmann_ha,
        shercliff_ha=args.shercliff_ha,
        hunt_ha=args.hunt_ha,
        resolution=args.resolution,
        hunt_resolution=args.hunt_resolution,
        hunt_dt=args.hunt_dt,
        hunt_t_final=args.hunt_t_final,
        hunt_frames=args.hunt_frames,
        reference_root=args.reference_root,
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
