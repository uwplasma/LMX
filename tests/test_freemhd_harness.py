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

    assert "openfoam/openfoam2206-paraview:latest" in dockerfile
    assert "epotMultiRegionFoam" in dockerfile
    assert "epotMultiRegionInterFoam" in dockerfile
    assert "if [ -x ./Allwmake ]; then ./Allwmake; fi" in dockerfile
    assert "decomposePar -force" in runner
    assert "mpirun -np" in runner
    assert 'CORES="${2:-${CORES:-4}}"' in runner
    assert "scripts/run_freemhd_case.py" in readme
