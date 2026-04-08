#!/usr/bin/env python3
from __future__ import annotations

import argparse
import stat
from pathlib import Path


DOCKERFILE = """\
FROM microfluidica/openfoam:2206

ENV DEBIAN_FRONTEND=noninteractive
ENV WM_PROJECT_DIR=/usr/lib/openfoam/openfoam2206
ENV WM_THIRD_PARTY_DIR=/usr/lib/openfoam/openfoam2206/ThirdParty
ENV FOAM_ETC=/usr/lib/openfoam/openfoam2206/etc

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

RUN mkdir -p /opt/FreeMHD
COPY FreeMHD/ /opt/FreeMHD/

RUN /bin/bash -lc "source ${WM_PROJECT_DIR}/etc/bashrc && cd /opt/FreeMHD && \\
    if [ ! -d ./MHD_Solvers ]; then rm -rf /opt/FreeMHD && git clone --depth 1 https://github.com/PlasmaControl/FreeMHD.git /opt/FreeMHD && cd /opt/FreeMHD; fi && \\
    if [ -x ./Allwmake ]; then ./Allwmake; fi && \\
    wmake MHD_Solvers/solvers/epotMultiRegionFoam && \\
    wmake MHD_Solvers/solvers/epotMultiRegionInterFoam"

RUN mkdir -p /opt/lmx
COPY run_freemhd_case.sh /opt/lmx/run_freemhd_case.sh
RUN chmod +x /opt/lmx/run_freemhd_case.sh

WORKDIR /workspace
CMD ["/bin/bash"]
"""


RUN_SCRIPT = r"""\
#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="${1:-/workspace/case}"
CORES="${2:-${CORES:-4}}"
SOLVER="${3:-${SOLVER:-epotMultiRegionFoam}}"
MPI_EXTRA_ARGS="${MPI_EXTRA_ARGS:---oversubscribe}"
LMX_END_TIME="${LMX_END_TIME:-}"
LMX_WRITE_INTERVAL="${LMX_WRITE_INTERVAL:-}"
LMX_DELTA_T="${LMX_DELTA_T:-}"
LMX_START_FROM="${LMX_START_FROM:-}"
LMX_LOG_COUPLED_ITERATIONS="${LMX_LOG_COUPLED_ITERATIONS:-}"

set +eu
source "${WM_PROJECT_DIR}/etc/bashrc"
set -eu
cd "${CASE_DIR}"

sync_decompose_par_dict() {
  if [[ ! -d "system" ]]; then
    return 0
  fi
  find system -name decomposeParDict -print0 | while IFS= read -r -d '' path; do
    python3 - "$CORES" "$path" <<'PY'
from pathlib import Path
import re
import sys

target = sys.argv[1]
path = Path(sys.argv[2])
text = path.read_text()
updated, count = re.subn(
    r"(^\s*numberOfSubdomains\s+)\d+(\s*;)",
    rf"\g<1>{target}\g<2>",
    text,
    count=1,
    flags=re.MULTILINE,
)
if count:
    path.write_text(updated)
PY
  done
}

sync_control_dict() {
  if [[ ! -f "system/controlDict" ]]; then
    return 0
  fi
  python3 - "$LMX_END_TIME" "$LMX_WRITE_INTERVAL" "$LMX_DELTA_T" "$LMX_START_FROM" "$LMX_LOG_COUPLED_ITERATIONS" <<'PY'
from pathlib import Path
import re
import sys

path = Path("system/controlDict")
text = path.read_text()
replacements = {
    "endTime": sys.argv[1],
    "writeInterval": sys.argv[2],
    "deltaT": sys.argv[3],
    "startFrom": sys.argv[4],
    "logCoupledMhdIterations": sys.argv[5],
}
updated = text
for key, value in replacements.items():
    if not value:
        continue
    updated, count = re.subn(
        rf"(^\s*{re.escape(key)}\s+)[^;]+(\s*;)",
        rf"\g<1>{value}\g<2>",
        updated,
        count=1,
        flags=re.MULTILINE,
    )
    if count == 0:
        updated = updated.rstrip() + f"\n{key} {value};\n"
if updated != text:
    path.write_text(updated)
PY
}

if [[ -x "./Allclean" ]]; then
  ./Allclean
fi

rm -rf processor* processors* postProcessing
rm -f log* runLog*
sync_control_dict

if [[ -f "constant/regionProperties" && ! -f "constant/liquid/polyMesh/points" ]]; then
  blockMesh
  topoSet
  splitMeshRegions -cellZonesOnly -overwrite -fileHandler collated
  for region in $(foamListRegions); do
    changeDictionary -region "${region}" -fileHandler collated
  done
  setExprFields -region liquid -fileHandler collated
  setExprFields -region solidWalls -fileHandler collated
  setExprFields -region insulator -fileHandler collated
fi

if [[ -x "./Allrun" ]]; then
  ./Allrun
else
  if [[ -f "system/decomposeParDict" && "${CORES}" -gt 1 ]]; then
    sync_decompose_par_dict
    if [[ -f "constant/regionProperties" ]]; then
      decomposePar -allRegions -force -fileHandler collated
    else
      decomposePar -force
    fi
    mpirun ${MPI_EXTRA_ARGS} -np "${CORES}" "${SOLVER}" -parallel | tee "runLog.${SOLVER}"
    if [[ -f "constant/regionProperties" ]]; then
      reconstructPar -allRegions || true
    else
      reconstructPar || true
    fi
  else
    "${SOLVER}" | tee "runLog.${SOLVER}"
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
docker build --platform linux/amd64 -t lmx-freemhd ./docker
/Users/rogerio/base_env/bin/python3 scripts/build_freemhd_container.py --image lmx-freemhd
/Users/rogerio/base_env/bin/python3 scripts/run_freemhd_case.py --image lmx-freemhd --case-dir /absolute/path/to/case
```
"""


def write_container_bundle(root: str | Path) -> list[Path]:
    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    dockerfile = root_path / "Dockerfile"
    readme = root_path / "README.md"
    run_script = root_path / "run_freemhd_case.sh"
    freemhd_placeholder = root_path / "FreeMHD" / ".gitkeep"
    dockerfile.write_text(DOCKERFILE)
    readme.write_text(README)
    run_script.write_text(RUN_SCRIPT)
    freemhd_placeholder.parent.mkdir(parents=True, exist_ok=True)
    freemhd_placeholder.write_text("")
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
