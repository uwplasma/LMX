#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lmx.freemhd import docker_cli_available, docker_daemon_available
from lmx.validation import inspect_freemhd_case


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a FreeMHD/OpenFOAM case directory for parity readiness.")
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("artifacts/freemhd/case_inspection.json"))
    args = parser.parse_args()

    inspection = inspect_freemhd_case(args.case_dir)
    payload = {
        "docker_cli_available": docker_cli_available(),
        "docker_daemon_available": docker_daemon_available(),
        "case_dir": inspection.case_dir,
        "control_dicts": list(inspection.control_dicts),
        "fv_schemes": list(inspection.fv_schemes),
        "fv_solutions": list(inspection.fv_solutions),
        "region_properties": list(inspection.region_properties),
        "block_mesh_dicts": list(inspection.block_mesh_dicts),
        "boundary_field_dirs": list(inspection.boundary_field_dirs),
        "latest_time_dirs": list(inspection.latest_time_dirs),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
