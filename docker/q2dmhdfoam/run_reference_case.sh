#!/usr/bin/env bash
set -eo pipefail

source /home/openfoam/foam/foam-extend-4.1/etc/bashrc
set -u

CASE_NAME="${CASE_NAME:-Q2DfullyDeveloped}"
CASE_RELATIVE_PATH="${CASE_RELATIVE_PATH:-}"
RANKS="${RANKS:-2}"
OUT_DIR="${1:-/output}"
if [[ -n "${CASE_RELATIVE_PATH}" ]]; then
    CASE_SRC="/home/openfoam/Q2DmhdFoam/${CASE_RELATIVE_PATH}"
else
    CASE_SRC="/home/openfoam/Q2DmhdFoam/tutorials/${CASE_NAME}"
fi
CASE_WORK="/tmp/lmx_q2dmhdfoam_case"

copy_failure_artifacts() {
    local code=$?
    if [[ "${code}" -eq 0 ]]; then
        return
    fi
    mkdir -p "${OUT_DIR}/case"
    if [[ -d "${CASE_WORK}" ]]; then
        find "${CASE_WORK}" -maxdepth 1 -type f -name 'log.*' -exec cp {} "${OUT_DIR}/case/" \; 2>/dev/null || true
        for subdir in 0 constant system; do
            if [[ -e "${CASE_WORK}/${subdir}" && ! -e "${OUT_DIR}/case/${subdir}" ]]; then
                cp -a "${CASE_WORK}/${subdir}" "${OUT_DIR}/case/" 2>/dev/null || true
            fi
        done
    fi
    cat > "${OUT_DIR}/summary.json" <<EOF
{
  "case": "${CASE_RELATIVE_PATH:-tutorials/${CASE_NAME}}",
  "status": "external_reference_case_failed",
  "exit_code": ${code},
  "rank_count": ${RANKS},
  "source": "Q2DmhdFoam foam-extend 4.1 generic runner"
}
EOF
}
trap copy_failure_artifacts EXIT

if [[ ! -d "${CASE_SRC}" ]]; then
    echo "Unknown Q2DmhdFoam case: ${CASE_SRC}" >&2
    exit 2
fi

rm -rf "${CASE_WORK}"
mkdir -p "${OUT_DIR}"
cp -a "${CASE_SRC}" "${CASE_WORK}"
cd "${CASE_WORK}"

# foam-extend 4.1 runtime compatibility for the upstream tutorial dictionaries.
if [[ ! -f constant/g ]]; then
    cat > constant/g <<'EOF'
/*--------------------------------*- C++ -*----------------------------------*\
| =========                 |                                                 |
| \\      /  F ield         | foam-extend: Open Source CFD                    |
|  \\    /   O peration     | Version:     4.1                                |
|   \\  /    A nd           | Web:         http://www.foam-extend.org         |
|    \\/     M anipulation  |                                                 |
\*---------------------------------------------------------------------------*/
FoamFile
{
    version     2.0;
    format      ascii;
    class       uniformDimensionedVectorField;
    location    "constant";
    object      g;
}

dimensions      [0 1 -2 0 0 0 0];
value           (0 0 0);
EOF
fi
if ! grep -q "ssCriteria" system/controlDict; then
    perl -0pi -e 's/(maxCo\s+[^;]+;\n)/$1\nssCriteria      true;\n/s' system/controlDict
fi
if [[ -f 0/theta && ! -f 0/T ]]; then
    cp 0/theta 0/T
    perl -0pi -e 's/object\s+theta;/object      T;/g' 0/T
    perl -0pi -e 's/\btheta\b/T/g' system/fvSolution system/fvSchemes system/controlDict 2>/dev/null || true
fi
if [[ "${FORCE_END_TIME:-}" == "1" ]]; then
    perl -0pi -e 's/stopAt\s+writeNow;/stopAt          endTime;/g' system/controlDict
fi
if [[ -n "${END_TIME:-}" ]]; then
    perl -0pi -e "s/endTime\\s+[^;]+;/endTime         ${END_TIME};/g" system/controlDict
fi
if [[ -n "${DELTA_T:-}" ]]; then
    perl -0pi -e "s/deltaT\\s+[^;]+;/deltaT          ${DELTA_T};/g" system/controlDict
fi
if [[ -n "${WRITE_INTERVAL:-}" ]]; then
    perl -0pi -e "s/writeInterval\\s+[^;]+;/writeInterval   ${WRITE_INTERVAL};/g" system/controlDict
fi
if [[ -n "${WRITE_CONTROL:-}" ]]; then
    perl -0pi -e "s/writeControl\\s+[^;]+;/writeControl    ${WRITE_CONTROL};/g" system/controlDict
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
LATEST_TIME="$(find . -maxdepth 1 -type d -regex './[0-9].*' | sed 's#^./##' | sort -g | tail -1)"
if [[ "${EXTRACT_PROFILE:-auto}" == "1" || ( "${EXTRACT_PROFILE:-auto}" == "auto" && "${CASE_NAME}" == "Q2DfullyDeveloped" && -z "${CASE_RELATIVE_PATH}" ) ]]; then
    python /home/openfoam/extract_reference_profile.py "${CASE_WORK}" "${OUT_DIR}"
else
    cat > "${OUT_DIR}/summary.json" <<EOF
{
  "case": "${CASE_RELATIVE_PATH:-tutorials/${CASE_NAME}}",
  "status": "external_reference_case_complete_no_profile_extraction",
  "final_time": ${LATEST_TIME:-null},
  "rank_count": ${RANKS},
  "source": "Q2DmhdFoam foam-extend 4.1 generic runner"
}
EOF
fi

rm -rf "${OUT_DIR}/case" "${OUT_DIR}/VTK"
mkdir -p "${OUT_DIR}/case"
cp -a 0 constant system log.* "${OUT_DIR}/case/"
if [[ -n "${LATEST_TIME}" ]]; then
    cp -a "${LATEST_TIME}" "${OUT_DIR}/case/"
fi
cp -a VTK "${OUT_DIR}/VTK"

echo "Q2DmhdFoam reference case complete."
echo "Artifacts written under ${OUT_DIR}"
