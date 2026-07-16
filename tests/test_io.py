from dataclasses import fields, replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import numpy as np
import jax.numpy as jnp

from lmx.core import Diagnostics, MHDState, Solution, zeros_state
from lmx.cases import make_hartmann_case
from lmx.io import (
    load_extruded_restart_bundle,
    load_restart_bundle,
    prepare_extruded_output_layout,
    validate_extruded_restart_bundle,
    validate_restart_bundle,
    write_extruded_restart_npz,
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
        face_lorentz_max_history=jnp.asarray([0.75, 0.65, 0.55]),
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
    return Solution(
        mesh=mesh, state=state, diagnostics=diagnostics, case_name=case.name
    )


def _extruded_rect_case(*, nx=3, write_plots=False, write_stride=1):
    case = make_hartmann_case(ha=5.0, ny=8, nz=8)
    return replace(
        case,
        name="fringing_rect_demo",
        geometry=replace(
            case.geometry, kind="rect_duct", length=6.0, nx=nx, ny=2, nz=2
        ),
        solver=replace(case.solver, kind="extruded_inductionless"),
        output=replace(
            case.output, write_plots=write_plots, write_stride=write_stride
        ),
    )


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
    mesh_without_points = mesh.__class__(**{**mesh.__dict__, "point_coordinates": None})

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
        assert {item.name for item in fields(Diagnostics)} <= set(data.files)
        assert data["u"].shape == solution.state.u.shape


def test_load_restart_bundle_round_trips_solution_npz(tmp_path: Path):
    case = make_hartmann_case(ha=5.0, ny=8, nz=8)
    solution = _sample_solution(case)
    path = write_solution_npz(solution, case, tmp_path / "hartmann_results.npz")

    bundle = load_restart_bundle(path)

    validate_restart_bundle(
        bundle,
        mesh=_build_mesh(case),
        geometry_kind=case.geometry.kind,
        case_name=case.name,
    )
    assert bundle.path == path.resolve()
    assert bundle.geometry_kind == case.geometry.kind
    assert bundle.state.u.shape == solution.state.u.shape
    assert float(bundle.state.time) == pytest.approx(float(solution.state.time))
    assert float(bundle.state.residual) == pytest.approx(float(solution.state.residual))
    for item in fields(Diagnostics):
        np.testing.assert_allclose(
            getattr(bundle.diagnostics, item.name),
            getattr(solution.diagnostics, item.name),
        )


def test_load_restart_bundle_falls_back_to_metadata_and_residual_history(
    tmp_path: Path,
):
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


def test_validate_restart_bundle_rejects_geometry_shape_faces_and_case_mismatch(
    tmp_path: Path,
):
    case = make_hartmann_case(ha=5.0, ny=8, nz=8)
    solution = _sample_solution(case)
    path = write_solution_npz(solution, case, tmp_path / "hartmann_results.npz")
    bundle = load_restart_bundle(path)
    mesh = _build_mesh(case)

    with pytest.raises(ValueError, match="geometry_kind"):
        validate_restart_bundle(
            bundle, mesh=mesh, geometry_kind="pipe", case_name=case.name
        )

    wrong_shape_bundle = bundle.__class__(
        **{
            **bundle.__dict__,
            "state": bundle.state.__class__(
                **{**bundle.state.__dict__, "u": jnp.zeros((1, 1))}
            ),
        }
    )
    with pytest.raises(ValueError, match="field shape"):
        validate_restart_bundle(
            wrong_shape_bundle,
            mesh=mesh,
            geometry_kind=case.geometry.kind,
            case_name=case.name,
        )

    wrong_y_faces = bundle.__class__(
        **{**bundle.__dict__, "y_faces": np.array([0.0, 1.0])}
    )
    with pytest.raises(ValueError, match="y_faces"):
        validate_restart_bundle(
            wrong_y_faces,
            mesh=mesh,
            geometry_kind=case.geometry.kind,
            case_name=case.name,
        )

    wrong_z_faces = bundle.__class__(
        **{**bundle.__dict__, "z_faces": np.array([0.0, 1.0])}
    )
    with pytest.raises(ValueError, match="z_faces"):
        validate_restart_bundle(
            wrong_z_faces,
            mesh=mesh,
            geometry_kind=case.geometry.kind,
            case_name=case.name,
        )

    wrong_case = bundle.__class__(
        **{**bundle.__dict__, "metadata": {**bundle.metadata, "case": "other_case"}}
    )
    with pytest.raises(ValueError, match="Restart case"):
        validate_restart_bundle(
            wrong_case, mesh=mesh, geometry_kind=case.geometry.kind, case_name=case.name
        )


def test_write_solution_outputs_respects_output_flags(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
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

    monkeypatch.setattr(
        "lmx.io.write_paraview",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("unexpected paraview")
        ),
    )
    monkeypatch.setattr(
        "lmx.io.write_solution_npz",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected npz")),
    )

    outputs = write_solution_outputs(
        solution, case, tmp_path, write_npz=True, write_plots=True
    )

    assert outputs == {"paraview": [], "csv": [], "npz": [], "plots": []}


def test_write_extruded_solution_npz_and_outputs(tmp_path: Path):
    case = _extruded_rect_case(write_plots=True)
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
        rho_phi_plus=jnp.ones((3, 3, 1, 2)),
        rho_phi_inlet=jnp.ones((1, 2)),
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
        axial_pressure_loss_gradient=jnp.asarray([1.0, 1.5, 1.0]),
        transverse_pressure_difference=jnp.asarray([0.0, 0.25, 0.0]),
        charge_balance_residual=jnp.asarray([1.0e-7, 2.0e-7, 1.0e-7]),
        boundary_current_residual=jnp.asarray([3.0e-8, 3.0e-8, 3.0e-8]),
        iteration_electric_linear_history=jnp.asarray(
            [[1.0e-8, 1.0e-9, 2.0e-8, 12.0, 1.0, 1.0]]
        ),
        iteration_potential_residual_history=jnp.asarray([3.0e-5]),
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

    npz_path = write_extruded_solution_npz(
        solution, case, tmp_path / "fringing_results.npz"
    )
    outputs = write_extruded_solution_outputs(
        solution, case, tmp_path, write_plots=True
    )

    assert npz_path.exists()
    assert outputs["csv"][0].exists()
    assert outputs["npz"][0].exists()
    assert outputs["plots"][0].exists()
    assert outputs["csv"][0].parent.name == "postProcessing"
    assert outputs["npz"][0].parent.name == "fields"
    assert outputs["plots"][0].parent.name == "plots"
    assert outputs["archive"][0].name.endswith("_extruded_manifest.json")
    assert outputs["archive"][1].parent.name == "stations"
    with np.load(npz_path, allow_pickle=False) as data:
        assert data["u"].shape == (3, 2, 2)
        assert data["validation_station_count"] == pytest.approx(3)
        assert "rho_phi_plus" not in data
        assert "rho_phi_inlet" not in data


def test_extruded_restart_bundle_round_trip_and_layout(tmp_path: Path):
    case = _extruded_rect_case()
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
        rho_phi_plus=jnp.arange(18.0).reshape((3, 3, 1, 2)),
        rho_phi_inlet=jnp.asarray([[0.4, 0.6]]),
        aitken_state=(jnp.ones((4, 3, 2, 2)), 0.75, 1),
        stopping_state=(1, 1, "in_progress"),
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
        axial_pressure_loss_gradient=jnp.asarray([1.0, 1.5, 1.0]),
        transverse_pressure_difference=jnp.asarray([0.0, 0.25, 0.0]),
        charge_balance_residual=jnp.asarray([1.0e-7, 2.0e-7, 1.0e-7]),
        boundary_current_residual=jnp.asarray([3.0e-8, 3.0e-8, 3.0e-8]),
        iteration_residual_history=jnp.asarray([1.0e-3]),
        iteration_momentum_defect_history=jnp.asarray([4.0e-5]),
        iteration_component_residual_history=jnp.asarray(
            [[1.0e-3, 2.0e-4, 3.0e-5, 1.0e-6, 2.0e-6, 3.0e-6]]),
        iteration_pressure_residual_history=jnp.asarray([2.0e-4]),
        iteration_pressure_linear_history=jnp.asarray(
            [[1.0e-9, 2.0e-10, 18.0, 1.0, 1.0]]
        ),
        iteration_electric_linear_history=jnp.asarray(
            [[1.0e-8, 1.0e-9, 2.0e-8, 12.0, 1.0, 1.0]]
        ),
        iteration_potential_residual_history=jnp.asarray([3.0e-5]),
        iteration_courant_history=jnp.asarray([[1.0e-3, 0.02, 0.04]]),
        geometry_kind="rect_duct",
        solver_kind="extruded_inductionless",
    )
    solution = SimpleNamespace(bundle=bundle, station_history=({"x": 0.0},))

    restart_path = write_extruded_restart_npz(
        solution, case, tmp_path / "restart" / "fringing_restart.npz"
    )
    restart_bundle = load_extruded_restart_bundle(restart_path)
    validate_extruded_restart_bundle(restart_bundle, case=case)

    layout = prepare_extruded_output_layout(tmp_path / "run")
    assert restart_path.exists()
    assert restart_bundle.bundle.u.shape == (3, 2, 2)
    assert restart_bundle.metadata["restart_schema"] == "b2_diagnostics_v5"
    for name in ("rho_phi_plus", "rho_phi_inlet", "axial_pressure_loss_gradient",
                 "transverse_pressure_difference", "iteration_pressure_linear_history",
                 "iteration_electric_linear_history", "iteration_potential_residual_history",
                 "iteration_courant_history", "iteration_momentum_defect_history"):
        assert getattr(restart_bundle.bundle, name) == pytest.approx(getattr(bundle, name))
    assert restart_bundle.bundle.stopping_state == (1, 1, "in_progress")
    assert all(path.exists() for path in vars(layout).values())
    bundle.iteration_pressure_linear_history = jnp.zeros((2, 5))
    with pytest.raises(ValueError, match="inconsistent shape"):
        write_extruded_restart_npz(solution, case, tmp_path / "malformed.npz")
    bundle.iteration_pressure_linear_history = jnp.zeros((1, 5))
    bundle.iteration_momentum_defect_history = jnp.zeros(2)
    with pytest.raises(ValueError, match="Momentum defect history"):
        write_extruded_restart_npz(solution, case, tmp_path / "malformed_defect.npz")
    bundle.iteration_momentum_defect_history = jnp.asarray([4.0e-5])

    with np.load(restart_path, allow_pickle=False) as data:
        v5_payload = {key: data[key] for key in data.files}
        v5_metadata = json.loads(str(data["metadata_json"]))
        bad_stopping_payload = dict(v5_payload)
        bad_stopping_payload["metadata_json"] = json.dumps(
            {**v5_metadata, "stopping_state": [0, 1, "corrupt"]}
        )
        bad_momentum_rank_payload = dict(v5_payload)
        bad_momentum_rank_payload["iteration_momentum_defect_history"] = np.asarray(
            [[4.0e-5]]
        )
        missing_v4 = [(message, {key: data[key] for key in data.files if key != field})
                      for field, message in (
                          ("steady_streak", "missing accelerator state"),
                          ("iteration_courant_history", "missing CFL history"),
                          ("iteration_pressure_linear_history", "missing pressure linear history"),
                          ("iteration_momentum_defect_history", "missing momentum defect history"),
                      )]
        v4_short = {key: data[key] for key in data.files}
        v4_short["iteration_momentum_defect_history"] = np.zeros(0)
        v4_metadata = json.loads(str(data["metadata_json"]))
        v4_metadata["restart_schema"] = "b2_diagnostics_v4"
        v4_payload = {key: data[key] for key in data.files}
        v4_payload["metadata_json"] = json.dumps(v4_metadata)
        v3_metadata = {**v4_metadata, "restart_schema": "b2_diagnostics_v3"}
        v3_payload = {key: data[key] for key in data.files
                      if key != "iteration_momentum_defect_history"}
        v3_payload["metadata_json"] = json.dumps(v3_metadata)
        v2_metadata = {**v3_metadata, "restart_schema": "b2_diagnostics_v2"}
        v2_payload = {key: value for key, value in v3_payload.items()
                      if key != "iteration_pressure_linear_history"}
        v2_payload["metadata_json"] = json.dumps(v2_metadata)
        v1_metadata = {**v2_metadata, "restart_schema": "b2_aitken_v1"}
        v1_payload = {key: value for key, value in v2_payload.items()
                      if key != "iteration_courant_history"}
        v1_payload["metadata_json"] = json.dumps(v1_metadata)
        legacy_payload = {key: data[key] for key in data.files if key not in {
            "metadata_json", "rho_phi_plus", "rho_phi_inlet", "iteration_courant_history"}}
    bad_stopping_path = tmp_path / "restart" / "bad_stopping.npz"
    np.savez_compressed(bad_stopping_path, **bad_stopping_payload)
    with pytest.raises(ValueError, match="stopping state has inconsistent step count"):
        load_extruded_restart_bundle(bad_stopping_path)
    bad_rank_path = tmp_path / "restart" / "bad_momentum_rank.npz"
    np.savez_compressed(bad_rank_path, **bad_momentum_rank_payload)
    with pytest.raises(
        ValueError, match="diagnostic restart histories have inconsistent lengths"
    ):
        load_extruded_restart_bundle(bad_rank_path)
    for index, (message, payload) in enumerate(missing_v4):
        missing_path = tmp_path / "restart" / f"diagnostics_v4_missing_{index}.npz"
        np.savez_compressed(missing_path, **payload)
        with pytest.raises(ValueError, match=message):
            load_extruded_restart_bundle(missing_path)
    short_path = tmp_path / "restart" / "diagnostics_v4_short.npz"
    np.savez_compressed(short_path, **v4_short)
    with pytest.raises(ValueError, match="inconsistent lengths"):
        load_extruded_restart_bundle(short_path)

    v4_path = tmp_path / "restart" / "diagnostics_v4.npz"
    np.savez_compressed(v4_path, **v4_payload)
    v4 = load_extruded_restart_bundle(v4_path)
    assert v4.metadata["restart_schema"] == "b2_diagnostics_v4"
    assert not v4.bundle.iteration_momentum_defect_history.size

    v3_path = tmp_path / "restart" / "diagnostics_v3.npz"
    np.savez_compressed(v3_path, **v3_payload)
    v3 = load_extruded_restart_bundle(v3_path)
    assert v3.metadata["restart_schema"] == "b2_diagnostics_v3"
    assert not v3.bundle.iteration_momentum_defect_history.size

    v2_path = tmp_path / "restart" / "diagnostics_v2.npz"
    np.savez_compressed(v2_path, **v2_payload)
    v2 = load_extruded_restart_bundle(v2_path)
    assert v2.metadata["restart_schema"] == "b2_diagnostics_v2"
    assert not v2.bundle.iteration_pressure_linear_history.size

    v1_path = tmp_path / "restart" / "aitken_v1.npz"
    np.savez_compressed(v1_path, **v1_payload)
    v1 = load_extruded_restart_bundle(v1_path)
    assert v1.metadata["restart_schema"] == "b2_aitken_v1"
    assert not v1.bundle.iteration_pressure_linear_history.size

    legacy_path = tmp_path / "restart" / "legacy.npz"
    np.savez_compressed(legacy_path, **legacy_payload)
    legacy = load_extruded_restart_bundle(legacy_path)
    assert legacy.bundle.rho_phi_plus is legacy.bundle.rho_phi_inlet is None
    assert legacy.metadata["restart_schema"] == "legacy_nonexact"
    assert not legacy.bundle.iteration_courant_history.size

    partial = SimpleNamespace(**{**bundle.__dict__, "rho_phi_inlet": None})
    with pytest.raises(ValueError, match="requires both"):
        write_extruded_restart_npz(
            SimpleNamespace(bundle=partial, station_history=()), case, tmp_path / "bad.npz"
        )


def test_validate_extruded_restart_bundle_rejects_mismatch():
    case = make_hartmann_case(ha=5.0, ny=8, nz=8)
    case = case.__class__(
        **{
            **case.__dict__,
            "name": "fringing_pipe_demo",
            "geometry": case.geometry.__class__(
                **{
                    **case.geometry.__dict__,
                    "kind": "pipe_ogrid",
                    "nx": 3,
                    "ny": 4,
                    "nz": 8,
                    "nr": 4,
                    "ntheta": 8,
                }
            ),
            "solver": case.solver.__class__(
                **{**case.solver.__dict__, "kind": "extruded_inductionless"}
            ),
        }
    )
    bad_bundle = SimpleNamespace(
        geometry_kind="rect_duct",
        solver_kind="extruded_inductionless",
        metadata={"case": "other_case"},
        bundle=SimpleNamespace(x=jnp.zeros((2,)), y=jnp.zeros((4,)), z=jnp.zeros((8,))),
    )

    with pytest.raises(ValueError, match="geometry_kind"):
        validate_extruded_restart_bundle(bad_bundle, case=case)


def test_validate_extruded_restart_bundle_rejects_solver_case_and_resolution_mismatch():
    case = _extruded_rect_case()
    bundle = SimpleNamespace(
        geometry_kind="rect_duct",
        solver_kind="fully_developed_inductionless",
        metadata={"case": "other_case"},
        bundle=SimpleNamespace(x=jnp.zeros((2,)), y=jnp.zeros((3,)), z=jnp.zeros((2,))),
    )

    with pytest.raises(ValueError, match="solver_kind"):
        validate_extruded_restart_bundle(bundle, case=case)

    good_solver = SimpleNamespace(
        **{**bundle.__dict__, "solver_kind": "extruded_inductionless"}
    )
    with pytest.raises(ValueError, match="Extruded restart case"):
        validate_extruded_restart_bundle(good_solver, case=case)

    good_case = SimpleNamespace(
        **{**good_solver.__dict__, "metadata": {"case": case.name}}
    )
    with pytest.raises(ValueError, match="station count"):
        validate_extruded_restart_bundle(good_case, case=case)

    good_x = SimpleNamespace(
        **{
            **good_case.__dict__,
            "bundle": SimpleNamespace(
                x=jnp.zeros((3,)), y=jnp.zeros((3,)), z=jnp.zeros((2,))
            ),
        }
    )
    with pytest.raises(ValueError, match="y resolution"):
        validate_extruded_restart_bundle(good_x, case=case)

    good_y = SimpleNamespace(
        **{
            **good_x.__dict__,
            "bundle": SimpleNamespace(
                x=jnp.zeros((3,)), y=jnp.zeros((2,)), z=jnp.zeros((3,))
            ),
        }
    )
    with pytest.raises(ValueError, match="z/theta resolution"):
        validate_extruded_restart_bundle(good_y, case=case)

    compact = SimpleNamespace(
        x=jnp.zeros(3),
        y=jnp.zeros(2),
        z=jnp.zeros(2),
        rho_phi_plus=jnp.zeros((3, 3, 1, 2)),
        rho_phi_inlet=jnp.zeros((1, 2)),
    )
    valid = SimpleNamespace(**{**good_case.__dict__, "bundle": compact})
    validate_extruded_restart_bundle(valid, case=case)
    malformed = (
        (jnp.zeros((2, 3, 1, 2)), jnp.zeros((1, 2)), "shape"),
        (jnp.zeros((3, 2, 1, 2)), jnp.zeros((1, 2)), "shape"),
        (jnp.zeros((3, 3, 1, 2)), jnp.zeros((2, 1)), "shape"),
        (jnp.zeros((3, 3, 1, 2)), None, "requires both"),
    )
    for plus, inlet, message in malformed:
        candidate = SimpleNamespace(
            **{**compact.__dict__, "rho_phi_plus": plus, "rho_phi_inlet": inlet}
        )
        with pytest.raises(ValueError, match=message):
            validate_extruded_restart_bundle(
                SimpleNamespace(**{**valid.__dict__, "bundle": candidate}), case=case
            )


def test_write_extruded_solution_outputs_archives_last_station_with_stride(
    tmp_path: Path,
):
    case = _extruded_rect_case(nx=4, write_stride=2)
    bundle = SimpleNamespace(
        x=jnp.asarray([0.0, 1.0, 2.0, 3.0]),
        y=jnp.asarray([-0.5, 0.5]),
        z=jnp.asarray([-0.5, 0.5]),
        field_scale=jnp.asarray([0.0, 1.0, 0.5, 0.0]),
        u=jnp.ones((4, 2, 2)),
        v=jnp.zeros((4, 2, 2)),
        w=jnp.zeros((4, 2, 2)),
        p=jnp.zeros((4, 2, 2)),
        phi=jnp.zeros((4, 2, 2)),
        jx=jnp.zeros((4, 2, 2)),
        jy=jnp.zeros((4, 2, 2)),
        jz=jnp.zeros((4, 2, 2)),
        lorentz_x=jnp.zeros((4, 2, 2)),
        lorentz_y=jnp.zeros((4, 2, 2)),
        lorentz_z=jnp.zeros((4, 2, 2)),
        residual=jnp.asarray([1.0e-3, 2.0e-4, 3.0e-5, 4.0e-6]),
        volumetric_flow_rate=jnp.asarray([1.0, 1.1, 1.2, 1.3]),
        mean_velocity=jnp.asarray([0.5, 0.55, 0.6, 0.65]),
        axial_current=jnp.asarray([0.0, 0.1, 0.05, 0.0]),
        wall_current_leakage=jnp.asarray([1.0e-6, 2.0e-6, 1.5e-6, 1.0e-6]),
        current_scaled_pressure_proxy=jnp.asarray([0.1, 0.2, 0.15, 0.1]),
        charge_balance_residual=jnp.asarray([1.0e-7, 2.0e-7, 1.5e-7, 1.0e-7]),
        boundary_current_residual=jnp.asarray([3.0e-8, 3.0e-8, 3.0e-8, 3.0e-8]),
        geometry_kind="rect_duct",
        solver_kind="extruded_inductionless",
    )
    validation = SimpleNamespace(
        station_count=4,
        max_residual=1.0e-3,
        max_charge_balance_residual=2.0e-7,
        mean_velocity_span=0.15,
        volumetric_flow_rate_span=0.3,
        axial_current_span=0.1,
        max_wall_current_leakage=2.0e-6,
        net_boundary_current_residual=3.0e-6,
        field_mean_velocity_correlation=-0.9,
    )
    station_history = tuple(
        {
            "x": float(i),
            "field_scale": float(bundle.field_scale[i]),
            "u_max": 1.0,
            "mean_velocity": float(bundle.mean_velocity[i]),
            "volumetric_flow_rate": float(bundle.volumetric_flow_rate[i]),
            "axial_current": float(bundle.axial_current[i]),
            "wall_current_leakage": float(bundle.wall_current_leakage[i]),
            "current_scaled_pressure_proxy": float(
                bundle.current_scaled_pressure_proxy[i]
            ),
            "residual": float(bundle.residual[i]),
            "charge_balance_residual": float(bundle.charge_balance_residual[i]),
            "boundary_current_residual": float(bundle.boundary_current_residual[i]),
        }
        for i in range(4)
    )
    solution = SimpleNamespace(
        bundle=bundle, validation=validation, station_history=station_history
    )

    outputs = write_extruded_solution_outputs(
        solution, case, tmp_path, write_plots=False
    )

    archived = [path.name for path in outputs["archive"] if path.suffix == ".npz"]
    assert archived == ["station_0000.npz", "station_0002.npz", "station_0003.npz"]
