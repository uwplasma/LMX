from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import jax.numpy as jnp

from lmx.core import Diagnostics, MHDState, Solution, zeros_state
from lmx.cases import make_hartmann_case
from lmx.io import (
    load_restart_bundle,
    validate_restart_bundle,
    write_extruded_solution_npz,
    write_extruded_solution_outputs,
    write_paraview,
    write_solution_npz,
    write_solution_outputs,
    write_vtu,
)
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
        charge_balance_residual_history=jnp.asarray([1.0e-7, 8.0e-8, 6.0e-8]),
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
        assert "charge_balance_residual_history" in data
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
    assert bundle.diagnostics.charge_balance_residual_history.shape == solution.diagnostics.charge_balance_residual_history.shape
    assert bundle.diagnostics.gauge_residual_history.shape == solution.diagnostics.gauge_residual_history.shape
    assert bundle.diagnostics.interface_current_residual_history.shape == solution.diagnostics.interface_current_residual_history.shape
    assert bundle.diagnostics.raw_update_max_history.shape == solution.diagnostics.raw_update_max_history.shape
    assert bundle.diagnostics.limiter_scale_history.shape == solution.diagnostics.limiter_scale_history.shape
    assert bundle.diagnostics.limited_fraction_history.shape == solution.diagnostics.limited_fraction_history.shape


def test_load_restart_bundle_falls_back_to_metadata_and_residual_history(tmp_path: Path):
    path = tmp_path / "restart_minimal.npz"
    np.savez_compressed(
        path,
        metadata_json='{"case": "demo", "time": 0.75, "geometry_kind": "rect_duct"}',
        y_faces=np.array([-1.0, 1.0]),
        z_faces=np.array([-1.0, 1.0]),
        u=np.array([[1.0]]),
        phi=np.array([[0.0]]),
        jy=np.array([[0.0]]),
        jz=np.array([[0.0]]),
        lorentz_x=np.array([[0.0]]),
        residual_history=np.array([1.0e-2, 1.0e-3]),
    )

    bundle = load_restart_bundle(path)

    assert float(bundle.state.time) == pytest.approx(0.75)
    assert float(bundle.state.residual) == pytest.approx(1.0e-3)
    assert bundle.diagnostics.time_history.shape == (0,)


def test_validate_restart_bundle_rejects_geometry_shape_faces_and_case_mismatch(tmp_path: Path):
    case = make_hartmann_case(ha=5.0, ny=8, nz=8)
    solution = _sample_solution(case)
    path = write_solution_npz(solution, case, tmp_path / "hartmann_results.npz")
    bundle = load_restart_bundle(path)
    mesh = _build_mesh(case)

    with pytest.raises(ValueError, match="geometry_kind"):
        validate_restart_bundle(bundle, mesh=mesh, geometry_kind="pipe", case_name=case.name)

    wrong_shape_bundle = bundle.__class__(
        **{**bundle.__dict__, "state": bundle.state.__class__(**{**bundle.state.__dict__, "u": jnp.zeros((1, 1))})}
    )
    with pytest.raises(ValueError, match="field shape"):
        validate_restart_bundle(wrong_shape_bundle, mesh=mesh, geometry_kind=case.geometry.kind, case_name=case.name)

    wrong_y_faces = bundle.__class__(**{**bundle.__dict__, "y_faces": np.array([0.0, 1.0])})
    with pytest.raises(ValueError, match="y_faces"):
        validate_restart_bundle(wrong_y_faces, mesh=mesh, geometry_kind=case.geometry.kind, case_name=case.name)

    wrong_z_faces = bundle.__class__(**{**bundle.__dict__, "z_faces": np.array([0.0, 1.0])})
    with pytest.raises(ValueError, match="z_faces"):
        validate_restart_bundle(wrong_z_faces, mesh=mesh, geometry_kind=case.geometry.kind, case_name=case.name)

    wrong_case = bundle.__class__(**{**bundle.__dict__, "metadata": {**bundle.metadata, "case": "other_case"}})
    with pytest.raises(ValueError, match="Restart case"):
        validate_restart_bundle(wrong_case, mesh=mesh, geometry_kind=case.geometry.kind, case_name=case.name)


def test_write_solution_outputs_respects_output_flags(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    case = make_hartmann_case(ha=5.0, ny=8, nz=8)
    case = case.__class__(
        **{
            **case.__dict__,
            "output": case.output.__class__(
                **{
                    **case.output.__dict__,
                    "write_paraview": False,
                    "write_csv_profiles": False,
                    "write_npz": False,
                    "write_plots": False,
                }
            ),
        }
    )
    solution = _sample_solution(case)

    monkeypatch.setattr("lmx.io.write_paraview", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected paraview")))
    monkeypatch.setattr("lmx.io.write_solution_npz", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected npz")))

    outputs = write_solution_outputs(solution, case, tmp_path, write_npz=True, write_plots=True)

    assert outputs == {"paraview": [], "csv": [], "npz": [], "plots": []}


def test_write_extruded_solution_npz_and_outputs(tmp_path: Path):
    case = make_hartmann_case(ha=5.0, ny=8, nz=8)
    case = case.__class__(
        **{
            **case.__dict__,
            "name": "fringing_rect_demo",
            "solver": case.solver.__class__(**{**case.solver.__dict__, "kind": "extruded_inductionless"}),
            "output": case.output.__class__(**{**case.output.__dict__, "write_plots": True}),
        }
    )
    bundle = SimpleNamespace(
        x=jnp.asarray([0.0, 1.0, 2.0]),
        y=jnp.asarray([-0.5, 0.5]),
        z=jnp.asarray([-0.5, 0.5]),
        field_scale=jnp.asarray([0.0, 1.0, 0.0]),
        u=jnp.ones((3, 2, 2)),
        v=jnp.zeros((3, 2, 2)),
        w=jnp.zeros((3, 2, 2)),
        p=jnp.zeros((3, 2, 2)),
        phi=jnp.zeros((3, 2, 2)),
        jx=jnp.zeros((3, 2, 2)),
        jy=jnp.zeros((3, 2, 2)),
        jz=jnp.zeros((3, 2, 2)),
        lorentz_x=jnp.zeros((3, 2, 2)),
        lorentz_y=jnp.zeros((3, 2, 2)),
        lorentz_z=jnp.zeros((3, 2, 2)),
        residual=jnp.asarray([1.0e-3, 2.0e-4, 3.0e-5]),
        volumetric_flow_rate=jnp.asarray([1.0, 1.1, 1.2]),
        mean_velocity=jnp.asarray([0.5, 0.55, 0.6]),
        axial_current=jnp.asarray([0.0, 0.1, 0.0]),
        wall_current_leakage=jnp.asarray([1.0e-6, 2.0e-6, 1.0e-6]),
        current_scaled_pressure_proxy=jnp.asarray([0.1, 0.2, 0.1]),
        charge_balance_residual=jnp.asarray([1.0e-7, 2.0e-7, 1.0e-7]),
        boundary_current_residual=jnp.asarray([3.0e-8, 3.0e-8, 3.0e-8]),
        geometry_kind="rect_duct",
        solver_kind="extruded_inductionless",
    )
    validation = SimpleNamespace(
        station_count=3,
        max_residual=1.0e-3,
        max_charge_balance_residual=2.0e-7,
        mean_velocity_span=0.1,
        volumetric_flow_rate_span=0.2,
        axial_current_span=0.1,
        max_wall_current_leakage=2.0e-6,
        net_boundary_current_residual=3.0e-6,
        field_mean_velocity_correlation=-0.9,
    )
    solution = SimpleNamespace(
        bundle=bundle,
        validation=validation,
        station_history=(
            {
                "x": 0.0,
                "field_scale": 0.0,
                "u_max": 1.0,
                "mean_velocity": 0.5,
                "volumetric_flow_rate": 1.0,
                "axial_current": 0.0,
                "wall_current_leakage": 1.0e-6,
                "current_scaled_pressure_proxy": 0.1,
                "residual": 1.0e-3,
                "charge_balance_residual": 1.0e-7,
            },
            {
                "x": 1.0,
                "field_scale": 1.0,
                "u_max": 1.0,
                "mean_velocity": 0.55,
                "volumetric_flow_rate": 1.1,
                "axial_current": 0.1,
                "wall_current_leakage": 2.0e-6,
                "current_scaled_pressure_proxy": 0.2,
                "residual": 2.0e-4,
                "charge_balance_residual": 2.0e-7,
            },
        ),
    )

    npz_path = write_extruded_solution_npz(solution, case, tmp_path / "fringing_results.npz")
    outputs = write_extruded_solution_outputs(solution, case, tmp_path, write_plots=True)

    assert npz_path.exists()
    assert outputs["csv"][0].exists()
    assert outputs["npz"][0].exists()
    assert outputs["plots"][0].exists()
    with np.load(npz_path, allow_pickle=False) as data:
        assert data["u"].shape == (3, 2, 2)
        assert data["validation_station_count"] == pytest.approx(3)
