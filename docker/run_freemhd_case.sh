#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="${1:-/workspace/case}"
CORES="${2:-${CORES:-4}}"
SOLVER="${3:-${SOLVER:-epotMultiRegionFoam}}"
MPI_EXTRA_ARGS="${MPI_EXTRA_ARGS:---oversubscribe}"
LMX_END_TIME="${LMX_END_TIME:-}"
LMX_WRITE_INTERVAL="${LMX_WRITE_INTERVAL:-}"
LMX_DELTA_T="${LMX_DELTA_T:-}"

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
  python3 - "$LMX_END_TIME" "$LMX_WRITE_INTERVAL" "$LMX_DELTA_T" <<'PY'
from pathlib import Path
import re
import sys

path = Path("system/controlDict")
text = path.read_text()
replacements = {
    "endTime": sys.argv[1],
    "writeInterval": sys.argv[2],
    "deltaT": sys.argv[3],
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
  if [[ -f "system/decomposeParDict" ]]; then
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
    "${SOLVER}"
  fi
fi
