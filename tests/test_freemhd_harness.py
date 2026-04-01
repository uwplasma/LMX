from pathlib import Path

import pytest

from scripts.write_freemhd_container_files import write_container_bundle


pytestmark = pytest.mark.unit


def test_write_container_bundle_creates_expected_files(tmp_path: Path):
    written = write_container_bundle(tmp_path / "docker")
    paths = {path.name: path for path in written}
    assert set(paths) == {"Dockerfile", "README.md", "run_freemhd_case.sh"}

    dockerfile = paths["Dockerfile"].read_text()
    runner = paths["run_freemhd_case.sh"].read_text()
    readme = paths["README.md"].read_text()

    assert "microfluidica/openfoam:2206" in dockerfile
    assert "COPY FreeMHD/ /opt/FreeMHD/" in dockerfile
    assert "if [ ! -d ./MHD_Solvers ]" in dockerfile
    assert "wmake MHD_Solvers/solvers/epotMultiRegionFoam" in dockerfile
    assert "wmake MHD_Solvers/solvers/epotMultiRegionInterFoam" in dockerfile
    assert "if [ -x ./Allwmake ]; then ./Allwmake; fi" in dockerfile
    assert "ENV WM_PROJECT_DIR=/usr/lib/openfoam/openfoam2206" in dockerfile
    assert "decomposePar -force" in runner
    assert "sync_decompose_par_dict" in runner
    assert "sync_control_dict" in runner
    assert "find system -name decomposeParDict -print0" in runner
    assert 'python3 - "$CORES"' in runner
    assert 'LMX_END_TIME="${LMX_END_TIME:-}"' in runner
    assert 'LMX_WRITE_INTERVAL="${LMX_WRITE_INTERVAL:-}"' in runner
    assert 'LMX_DELTA_T="${LMX_DELTA_T:-}"' in runner
    assert 'LMX_START_FROM="${LMX_START_FROM:-}"' in runner
    assert "rm -rf processor* processors* postProcessing" in runner
    assert "rm -f log* runLog*" in runner
    assert "splitMeshRegions -cellZonesOnly -overwrite -fileHandler collated" in runner
    assert 'changeDictionary -region "${region}" -fileHandler collated' in runner
    assert "decomposePar -allRegions -force -fileHandler collated" in runner
    assert "reconstructPar -allRegions || true" in runner
    assert 'MPI_EXTRA_ARGS="${MPI_EXTRA_ARGS:---oversubscribe}"' in runner
    assert "mpirun ${MPI_EXTRA_ARGS} -np" in runner
    assert '| tee "runLog.${SOLVER}"' in runner
    assert 'set +eu' in runner
    assert 'source "${WM_PROJECT_DIR}/etc/bashrc"' in runner
    assert 'set -eu' in runner
    assert 'CORES="${2:-${CORES:-4}}"' in runner
    assert "docker build --platform linux/amd64 -t lmx-freemhd ./docker" in readme
    assert "scripts/build_freemhd_container.py" in readme
    assert "scripts/run_freemhd_case.py" in readme
    assert (tmp_path / "docker" / "FreeMHD" / ".gitkeep").exists()
