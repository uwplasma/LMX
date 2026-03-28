#!/usr/bin/env python3
from __future__ import annotations

import argparse
import stat
from pathlib import Path


DOCKERFILE = """\
FROM openfoam/openfoam2206-paraview:latest

ENV DEBIAN_FRONTEND=noninteractive
ENV WM_PROJECT_DIR=/opt/OpenFOAM/OpenFOAM-v2206
ENV WM_THIRD_PARTY_DIR=/opt/OpenFOAM/ThirdParty-v2206
ENV FOAM_INST_DIR=/opt/OpenFOAM

RUN apt-get update && apt-get install -y \\
    bash \\
    ca-certificates \\
    curl \\
    git \\
    python3 \\
    rsync \\
    wget && \\
    rm -rf /var/lib/apt/lists/*

WORKDIR /opt

RUN git clone --depth 1 https://github.com/PlasmaControl/FreeMHD.git /opt/FreeMHD

RUN /bin/bash -lc "source /opt/OpenFOAM/OpenFOAM-v2206/etc/bashrc && cd /opt/FreeMHD && \\
    if [ -x ./Allwmake ]; then ./Allwmake; fi && \\
    wmake MHD_Solvers/solvers/epotMultiRegionFoam && \\
    wmake MHD_Solvers/solvers/epotMultiRegionInterFoam"

RUN mkdir -p /opt/lmx
COPY run_freemhd_case.sh /opt/lmx/run_freemhd_case.sh
RUN chmod +x /opt/lmx/run_freemhd_case.sh

WORKDIR /workspace
CMD ["/bin/bash"]
"""


RUN_SCRIPT = """\
#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="${1:-/workspace/case}"
CORES="${2:-${CORES:-4}}"
SOLVER="${3:-${SOLVER:-epotMultiRegionFoam}}"

source /opt/OpenFOAM/OpenFOAM-v2206/etc/bashrc
cd "${CASE_DIR}"

if [[ -x "./Allclean" ]]; then
  ./Allclean
fi

if [[ -x "./Allrun" ]]; then
  ./Allrun
else
  if [[ -f "system/decomposeParDict" ]]; then
    decomposePar -force
    mpirun -np "${CORES}" "${SOLVER}" -parallel
    reconstructPar || true
  else
    "${SOLVER}"
  fi
fi
"""


README = """\
# FreeMHD Container Bundle

This bundle is the local handoff for building and running FreeMHD parity cases beside LMX.

## Files

- `Dockerfile`: OpenFOAM v2206 based image scaffold for FreeMHD.
- `run_freemhd_case.sh`: container entrypoint for `Allrun` or direct solver execution.

## Typical usage

```bash
docker build -t lmx-freemhd ./docker
/Users/rogerio/base_env/bin/python3 scripts/run_freemhd_case.py --image lmx-freemhd --case-dir /absolute/path/to/case
```
"""


def write_container_bundle(root: str | Path) -> list[Path]:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    dockerfile = root_path / "Dockerfile"
    readme = root_path / "README.md"
    run_script = root_path / "run_freemhd_case.sh"
    dockerfile.write_text(DOCKERFILE)
    readme.write_text(README)
    run_script.write_text(RUN_SCRIPT)
    run_script.chmod(run_script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return [dockerfile, readme, run_script]


def main() -> int:
    parser = argparse.ArgumentParser(description="Write the FreeMHD/OpenFOAM container bundle.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parents[1] / "docker")
    args = parser.parse_args()
    for path in write_container_bundle(args.output_dir):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
