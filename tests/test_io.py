from pathlib import Path

import pytest
import numpy as np
import jax.numpy as jnp

from lmx.core import Diagnostics, MHDState, Solution, zeros_state
from lmx.cases import make_hartmann_case
from lmx.io import load_restart_bundle, validate_restart_bundle, write_paraview, write_solution_npz, write_vtu
from lmx.mesh import generate_pipe_ogrid_mesh
from lmx.solvers import _build_mesh


pytestmark = pytest.mark.unit


def _sample_solution(case) -> Solution:
    mesh = _build_mesh(case)
    shape = mesh.yz_shape
    base = jnp.ones(shape)
    state = MHDState(
        u=base,
        phi=2.0 * base,
        jy=3.0 * base,
        jz=4.0 * base,
        lorentz_x=5.0 * base,
        time=0.25,
        residual=1.0e-6,
    )
    diagnostics = Diagnostics(
        residual_history=jnp.asarray([1.0e-2, 1.0e-4, 1.0e-6]),
        courant_like=jnp.asarray([0.05, 0.04, 0.03]),
        ohmic_power=jnp.asarray([0.2, 0.15, 0.1]),
        time_history=jnp.asarray([0.0, 0.125, 0.25]),
        u_max_history=jnp.asarray([1.0, 1.0, 1.0]),
        mean_velocity_history=jnp.asarray([0.8, 0.85, 0.9]),
        applied_forcing_history=jnp.asarray([1.2, 1.1, 1.0]),
        pressure_proxy_history=jnp.asarray([0.3, 0.25, 0.2]),
        current_scaled_pressure_proxy_history=jnp.asarray([0.4, 0.35, 0.3]),
        raw_update_max_history=jnp.asarray([0.2, 0.1, 0.05]),
        limiter_scale_history=jnp.asarray([1.0, 0.9, 0.8]),
        limited_fraction_history=jnp.asarray([0.0, 0.1, 0.2]),
        current_max_history=jnp.asarray([1.5, 1.4, 1.3]),
        face_current_max_history=jnp.asarray([1.6, 1.5, 1.4]),
        emf_max_history=jnp.asarray([0.9, 0.8, 0.7]),
        lorentz_max_history=jnp.asarray([0.7, 0.6, 0.5]),
        potential_residual_history=jnp.asarray([1.0e-3, 1.0e-4, 1.0e-5]),
        potential_iterations_history=jnp.asarray([10.0, 8.0, 6.0]),
        linear_residual_history=jnp.asarray([1.0e-2, 1.0e-4, 1.0e-6]),
        linear_iterations_history=jnp.asarray([12.0, 10.0, 8.0]),
        volumetric_flow_rate_history=jnp.asarray([0.9, 0.95, 1.0]),
        mean_current_magnitude_history=jnp.asarray([0.5, 0.45, 0.4]),
        lorentz_power_history=jnp.asarray([0.3, 0.25, 0.2]),
        div_current_max_history=jnp.asarray([1.0e-6, 5.0e-7, 2.5e-7]),
        gauge_residual_history=jnp.asarray([1.0e-8, 5.0e-9, 2.5e-9]),
        interface_current_residual_history=jnp.asarray([1.0e-6, 8.0e-7, 6.0e-7]),
    )
    return Solution(mesh=mesh, state=state, diagnostics=diagnostics, case_name=case.name)


def test_paraview_writer(tmp_path: Path):
    case = make_hartmann_case(ha=5.0, ny=16, nz=16)
    solution = _sample_solution(case)
    paths = write_paraview(solution, tmp_path)
    assert all(path.exists() for path in paths)


def test_vtu_writer(tmp_path: Path):
    mesh = generate_pipe_ogrid_mesh(radius=1.0, nx=1, nr=4, ntheta=8)
    path = write_vtu(mesh, tmp_path)
    assert path.exists()


def test_vtu_writer_requires_point_coordinates(tmp_path: Path):
    case = make_hartmann_case(ha=5.0, ny=8, nz=8)
    mesh = _build_mesh(case)
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
    solution = _sample_solution(case)

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
    solution = _sample_solution(case)
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
