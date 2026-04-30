#!/usr/bin/env bash
set -eo pipefail

source /home/openfoam/foam/foam-extend-4.1/etc/bashrc
set -u

CASE_NAME="${CASE_NAME:-Q2DfullyDeveloped}"
RANKS="${RANKS:-2}"
OUT_DIR="${1:-/output}"
CASE_SRC="/home/openfoam/Q2DmhdFoam/tutorials/${CASE_NAME}"
CASE_WORK="/tmp/lmx_q2dmhdfoam_case"

if [[ ! -d "${CASE_SRC}" ]]; then
    echo "Unknown Q2DmhdFoam tutorial case: ${CASE_NAME}" >&2
    exit 2
fi

rm -rf "${CASE_WORK}"
mkdir -p "${OUT_DIR}"
cp -a "${CASE_SRC}" "${CASE_WORK}"
cd "${CASE_WORK}"

# foam-extend 4.1 runtime compatibility for the upstream tutorial dictionaries.
if ! grep -q "ssCriteria" system/controlDict; then
    perl -0pi -e 's/(maxCo\s+[^;]+;\n)/$1\nssCriteria      true;\n/s' system/controlDict
fi
perl -0pi -e 's/BiCGStab/PBiCG/g' system/fvSolution
perl -0pi -e 's/laplacianSchemes\n\{\n\s*default\s+none;/laplacianSchemes\n{\n    default                      Gauss linear corrected;/s' system/fvSchemes

blockMesh > log.blockMesh 2>&1

if [[ "${RANKS}" -gt 1 ]]; then
    cp system/decomposeParDict system/decomposeParDict.orig 2>/dev/null || true
    cat > system/decomposeParDict <<EOF
FoamFile
{
    version     2.0;
    format      ascii;
    class       dictionary;
    location    "system";
    object      decomposeParDict;
}

numberOfSubdomains ${RANKS};
method          scotch;
EOF
    decomposePar > log.decomposePar 2>&1
    mpirun --allow-run-as-root -np "${RANKS}" Q2DmhdFoam -parallel > log.Q2DmhdFoam 2>&1
    reconstructPar > log.reconstructPar 2>&1
else
    Q2DmhdFoam > log.Q2DmhdFoam 2>&1
fi

foamToVTK -ascii -latestTime > log.foamToVTK 2>&1 || foamToVTK -ascii > log.foamToVTK 2>&1
python /home/openfoam/extract_reference_profile.py "${CASE_WORK}" "${OUT_DIR}"

rm -rf "${OUT_DIR}/case" "${OUT_DIR}/VTK"
mkdir -p "${OUT_DIR}/case"
cp -a 0 constant system log.* "${OUT_DIR}/case/"
LATEST_TIME="$(find . -maxdepth 1 -type d -regex './[0-9].*' | sed 's#^./##' | sort -g | tail -1)"
if [[ -n "${LATEST_TIME}" ]]; then
    cp -a "${LATEST_TIME}" "${OUT_DIR}/case/"
fi
cp -a VTK "${OUT_DIR}/VTK"

echo "Q2DmhdFoam reference case complete."
echo "Artifacts written under ${OUT_DIR}"
