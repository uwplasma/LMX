#!/usr/bin/env bash
set -euo pipefail

CASE_DIR="${1:-/workspace/case}"
OUTPUT_DIR="${2:-/workspace/output}"
NPROC="${3:-${FREEMHD_NPROC:-4}}"

END_TIME="${FREEMHD_END_TIME:-1e-4}"
DELTA_T="${FREEMHD_DELTA_T:-1e-6}"
MAX_DELTA_T="${FREEMHD_MAX_DELTA_T:-1e-5}"
WRITE_INTERVAL="${FREEMHD_WRITE_INTERVAL:-${END_TIME}}"
SOLVER="${FREEMHD_SOLVER:-epotMultiRegionInterFoam}"

WORK_CASE="${TMPDIR:-/tmp}/freemhd-case"

if [[ ! -d "${CASE_DIR}" ]]; then
    echo "Case directory not found: ${CASE_DIR}" >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}"
rm -rf "${WORK_CASE}"
mkdir -p "${WORK_CASE}"

rsync -a \
    --delete \
    --exclude 'processor*' \
    --exclude 'postProcessing' \
    --exclude 'VTK' \
    --exclude 'runLog*' \
    --exclude 'log.*' \
    "${CASE_DIR}/" "${WORK_CASE}/"

cd "${WORK_CASE}"

touch case.foam

if [[ ! -f system/controlDict ]]; then
    echo "Missing system/controlDict in ${CASE_DIR}" >&2
    exit 1
fi

foamDictionary -entry application -set "${SOLVER}" system/controlDict >/dev/null
foamDictionary -entry startFrom -set startTime system/controlDict >/dev/null
foamDictionary -entry startTime -set 0 system/controlDict >/dev/null
foamDictionary -entry endTime -set "${END_TIME}" system/controlDict >/dev/null
foamDictionary -entry deltaT -set "${DELTA_T}" system/controlDict >/dev/null
foamDictionary -entry maxDeltaT -set "${MAX_DELTA_T}" system/controlDict >/dev/null || true
foamDictionary -entry writeControl -set adjustable system/controlDict >/dev/null
foamDictionary -entry writeInterval -set "${WRITE_INTERVAL}" system/controlDict >/dev/null
foamDictionary -entry purgeWrite -set 0 system/controlDict >/dev/null || true
foamDictionary -entry "functions.vtkWrite.writeControl" -set runTime system/controlDict >/dev/null || true
foamDictionary -entry "functions.vtkWrite.writeInterval" -set "${WRITE_INTERVAL}" system/controlDict >/dev/null || true
for dict in system/decomposeParDict system/*/decomposeParDict; do
    if [[ -f "${dict}" ]]; then
        foamDictionary -entry numberOfSubdomains -set "${NPROC}" "${dict}" >/dev/null
    fi
done

rm -rf processor* postProcessing VTK runLog runLog.* log.*

if [[ -f system/blockMeshDict && ! -f constant/polyMesh/points ]]; then
    blockMesh -fileHandler collated | tee "${OUTPUT_DIR}/log.blockMesh"
fi

if [[ ! -d constant/liquid/polyMesh || ! -d constant/solidWalls/polyMesh ]]; then
    if [[ -f system/topoSetDict ]]; then
        topoSet -fileHandler collated | tee "${OUTPUT_DIR}/log.topoSet"
    fi

    splitMeshRegions -cellZonesOnly -overwrite -fileHandler collated | tee "${OUTPUT_DIR}/log.splitMeshRegions"

    if foamListRegions >/tmp/freemhd_regions.txt 2>/dev/null; then
        while read -r region; do
            [[ -n "${region}" ]] || continue
            changeDictionary -region "${region}" -fileHandler collated | tee -a "${OUTPUT_DIR}/log.changeDictionary"
        done </tmp/freemhd_regions.txt
    fi

    for region in liquid solidWalls insulator; do
        if [[ -d "constant/${region}" ]]; then
            setExprFields -region "${region}" -fileHandler collated | tee -a "${OUTPUT_DIR}/log.setExprFields"
        fi
    done
fi

decomposePar -allRegions -force -fileHandler collated | tee "${OUTPUT_DIR}/log.decomposePar"

export OMPI_ALLOW_RUN_AS_ROOT=1
export OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1

mpirun -np "${NPROC}" "${SOLVER}" -parallel 2>&1 | tee "${OUTPUT_DIR}/run.log"

reconstructPar -allRegions -latestTime -fileHandler collated 2>&1 | tee "${OUTPUT_DIR}/log.reconstructPar"

foamToVTK -allRegions -latestTime 2>&1 | tee "${OUTPUT_DIR}/log.foamToVTK"

rsync -a --delete "${WORK_CASE}/VTK/" "${OUTPUT_DIR}/VTK/"

find "${WORK_CASE}" -maxdepth 1 -type d \
    | grep -E '/[0-9]+(\\.[0-9]+)?$' \
    | sort -V \
    | tail -n 1 \
    | while read -r latest_time; do
        rsync -a "${latest_time}/" "${OUTPUT_DIR}/latestTime/"
        printf '%s\n' "${latest_time##*/}" > "${OUTPUT_DIR}/latest_time.txt"
    done

cp -f system/controlDict "${OUTPUT_DIR}/controlDict.used"
cp -f system/decomposeParDict "${OUTPUT_DIR}/decomposeParDict.used"
cp -f case.foam "${OUTPUT_DIR}/case.foam"

echo "Wrote VTK outputs under ${OUTPUT_DIR}/VTK"
