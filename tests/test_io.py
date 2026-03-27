from pathlib import Path

import pytest

from lmx.cases import make_hartmann_case
from lmx.io import write_paraview, write_vtu
from lmx.mesh import generate_pipe_ogrid_mesh
from lmx.solvers import solve_steady


pytestmark = pytest.mark.unit


def test_paraview_writer(tmp_path: Path):
    case = make_hartmann_case(ha=5.0, ny=16, nz=16)
    solution = solve_steady(case)
    paths = write_paraview(solution, tmp_path)
    assert all(path.exists() for path in paths)


def test_vtu_writer(tmp_path: Path):
    mesh = generate_pipe_ogrid_mesh(radius=1.0, nx=1, nr=4, ntheta=8)
    path = write_vtu(mesh, tmp_path)
    assert path.exists()
