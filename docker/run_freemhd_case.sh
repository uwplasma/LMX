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
