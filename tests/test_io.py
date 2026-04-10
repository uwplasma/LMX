from pathlib import Path

import pytest
import numpy as np

from lmx.core import zeros_state
from lmx.cases import make_hartmann_case
from lmx.io import load_restart_bundle, validate_restart_bundle, write_paraview, write_solution_npz, write_vtu
from lmx.mesh import generate_pipe_ogrid_mesh
from lmx.solvers import _build_mesh, solve_steady


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


def test_vtu_writer_requires_point_coordinates(tmp_path: Path):
    case = make_hartmann_case(ha=5.0, ny=8, nz=8)
    solution = solve_steady(case)
    mesh = solution.mesh
    mesh_without_points = mesh.__class__(
        **{**mesh.__dict__, "point_coordinates": None}
    )

    with pytest.raises(ValueError, match="Mapped mesh requires point_coordinates"):
        write_vtu(mesh_without_points, tmp_path)


def test_zeros_state_matches_mesh_shape():
    mesh = generate_pipe_ogrid_mesh(radius=1.0, nx=1, nr=4, ntheta=8)
    state = zeros_state(mesh)

    assert state.u.shape == mesh.yz_shape
    assert state.phi.shape == mesh.yz_shape
    assert float(state.time) == 0.0


def test_write_solution_npz(tmp_path: Path):
    case = make_hartmann_case(ha=5.0, ny=8, nz=8)
    solution = solve_steady(case)

    path = write_solution_npz(solution, case, tmp_path / "hartmann_results.npz")

    assert path.exists()
    with np.load(path, allow_pickle=False) as data:
        assert "u" in data
        assert "phi" in data
        assert "state_time" in data
        assert "state_residual" in data
        assert "current_scaled_pressure_proxy_history" in data
        assert "linear_residual_history" in data
        assert "linear_iterations_history" in data
        assert "volumetric_flow_rate_history" in data
        assert "mean_current_magnitude_history" in data
        assert "lorentz_power_history" in data
        assert "div_current_max_history" in data
        assert "gauge_residual_history" in data
        assert "interface_current_residual_history" in data
        assert "raw_update_max_history" in data
        assert "limiter_scale_history" in data
        assert "limited_fraction_history" in data
        assert data["u"].shape == solution.state.u.shape


def test_load_restart_bundle_round_trips_solution_npz(tmp_path: Path):
    case = make_hartmann_case(ha=5.0, ny=8, nz=8)
    solution = solve_steady(case)
    path = write_solution_npz(solution, case, tmp_path / "hartmann_results.npz")

    bundle = load_restart_bundle(path)

    validate_restart_bundle(bundle, mesh=_build_mesh(case), geometry_kind=case.geometry.kind, case_name=case.name)
    assert bundle.path == path.resolve()
    assert bundle.geometry_kind == case.geometry.kind
    assert bundle.state.u.shape == solution.state.u.shape
    assert float(bundle.state.time) == pytest.approx(float(solution.state.time))
    assert float(bundle.state.residual) == pytest.approx(float(solution.state.residual))
    assert bundle.diagnostics.current_scaled_pressure_proxy_history.shape == solution.diagnostics.current_scaled_pressure_proxy_history.shape
    assert bundle.diagnostics.linear_residual_history.shape == solution.diagnostics.linear_residual_history.shape
    assert bundle.diagnostics.linear_iterations_history.shape == solution.diagnostics.linear_iterations_history.shape
    assert bundle.diagnostics.volumetric_flow_rate_history.shape == solution.diagnostics.volumetric_flow_rate_history.shape
    assert bundle.diagnostics.mean_current_magnitude_history.shape == solution.diagnostics.mean_current_magnitude_history.shape
    assert bundle.diagnostics.lorentz_power_history.shape == solution.diagnostics.lorentz_power_history.shape
    assert bundle.diagnostics.div_current_max_history.shape == solution.diagnostics.div_current_max_history.shape
    assert bundle.diagnostics.gauge_residual_history.shape == solution.diagnostics.gauge_residual_history.shape
    assert bundle.diagnostics.interface_current_residual_history.shape == solution.diagnostics.interface_current_residual_history.shape
    assert bundle.diagnostics.raw_update_max_history.shape == solution.diagnostics.raw_update_max_history.shape
    assert bundle.diagnostics.limiter_scale_history.shape == solution.diagnostics.limiter_scale_history.shape
    assert bundle.diagnostics.limited_fraction_history.shape == solution.diagnostics.limited_fraction_history.shape
