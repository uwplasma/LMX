#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lmx.validation import infer_sampling_geometry, latest_sampled_profiles


def sample_dict_text(
    dict_name: str,
    x_position: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    n_points: int,
) -> str:
    return f"""/*--------------------------------*- C++ -*----------------------------------*\\
| =========                 |                                                 |
| \\\\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox           |
|  \\\\    /   O peration     | Version:  2206                                  |
|   \\\\  /    A nd           | Website:  www.openfoam.com                      |
|    \\\\/     M anipulation  |                                                 |
\\*---------------------------------------------------------------------------*/
FoamFile
{{
    version     2.0;
    format      ascii;
    class       dictionary;
    object      {dict_name};
}}
// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

type            sets;
libs            ("libsampling.so");
writeControl    writeTime;
setFormat       raw;
interpolationScheme cellPoint;
fields          (U potE);

sets
(
    centerlineY
    {{
        type uniform;
        axis distance;
        start ({x_position} {y_min} 0.0);
        end ({x_position} {y_max} 0.0);
        nPoints {n_points};
    }}
    centerlineZ
    {{
        type uniform;
        axis distance;
        start ({x_position} 0.0 {z_min});
        end ({x_position} 0.0 {z_max});
        nPoints {n_points};
    }}
);
"""


def write_sample_dict(
    case_dir: str | Path,
    dict_name: str,
    x_position: float,
    y_min: float,
    y_max: float,
    z_min: float,
    z_max: float,
    n_points: int,
) -> Path:
    case_path = Path(case_dir).resolve()
    path = case_path / "system" / f"{dict_name}"
    path.write_text(
        sample_dict_text(
            dict_name=dict_name,
            x_position=x_position,
            y_min=y_min,
            y_max=y_max,
            z_min=z_min,
            z_max=z_max,
            n_points=n_points,
        )
    )
    return path


def run_postprocess_sampling(
    image: str,
    case_dir: str | Path,
    region: str,
    time: str,
    dict_name: str,
    platform: str = "linux/amd64",
) -> subprocess.CompletedProcess[str]:
    case_path = Path(case_dir).resolve()
    command = [
        "docker",
        "run",
        "--rm",
        "--platform",
        platform,
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "-e",
        "HOME=/tmp",
        "--entrypoint",
        "/bin/bash",
        "-v",
        f"{case_path}:/workspace/case",
        image,
        "-lc",
        (
            "set +eu; source /usr/lib/openfoam/openfoam2206/etc/bashrc; set -eu; "
            "cd /workspace/case; "
            f"postProcess -region {region} -func {dict_name} -time {time}"
        ),
    ]
    return subprocess.run(command, text=True, capture_output=True, check=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a FreeMHD sampling dictionary and run postProcess sampling.")
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--image", type=str, default="microfluidica/openfoam:2206")
    parser.add_argument("--platform", type=str, default="linux/amd64")
    parser.add_argument("--region", type=str, default="liquid")
    parser.add_argument("--time", type=str, default="0.0001")
    parser.add_argument("--dict-name", type=str, default="lmxSampleDict")
    parser.add_argument("--x-position", type=float, default=None)
    parser.add_argument("--y-min", type=float, default=None)
    parser.add_argument("--y-max", type=float, default=None)
    parser.add_argument("--z-min", type=float, default=None)
    parser.add_argument("--z-max", type=float, default=None)
    parser.add_argument("--n-points", type=int, default=201)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    if None in (args.x_position, args.y_min, args.y_max, args.z_min, args.z_max):
        inferred = infer_sampling_geometry(args.case_dir)
        x_position = inferred.x_position if args.x_position is None else args.x_position
        y_min = inferred.y_min if args.y_min is None else args.y_min
        y_max = inferred.y_max if args.y_max is None else args.y_max
        z_min = inferred.z_min if args.z_min is None else args.z_min
        z_max = inferred.z_max if args.z_max is None else args.z_max
    else:
        x_position = args.x_position
        y_min = args.y_min
        y_max = args.y_max
        z_min = args.z_min
        z_max = args.z_max

    dict_path = write_sample_dict(
        case_dir=args.case_dir,
        dict_name=args.dict_name,
        x_position=x_position,
        y_min=y_min,
        y_max=y_max,
        z_min=z_min,
        z_max=z_max,
        n_points=args.n_points,
    )
    result = run_postprocess_sampling(
        image=args.image,
        case_dir=args.case_dir,
        region=args.region,
        time=args.time,
        dict_name=args.dict_name,
        platform=args.platform,
    )
    sampled = latest_sampled_profiles(args.case_dir)
    payload = {
        "case_dir": str(args.case_dir.resolve()),
        "dict_path": str(dict_path),
        "image": args.image,
        "platform": args.platform,
        "region": args.region,
        "time": args.time,
        "x_position": x_position,
        "y_min": y_min,
        "y_max": y_max,
        "z_min": z_min,
        "z_max": z_max,
        "status": "ok" if result.returncode == 0 else "failed",
        "returncode": result.returncode,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
        "sampled_profile_y_path": sampled[0].path if sampled is not None else "",
        "sampled_profile_z_path": sampled[1].path if sampled is not None else "",
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0 if result.returncode == 0 else result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
