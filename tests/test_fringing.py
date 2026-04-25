import jax.numpy as jnp
import numpy as np
import pytest
from dataclasses import replace

from lmx.field_models import make_divergence_free_cross_section_field, sample_cross_section_field, write_tabulated_field_npz
from lmx.fringing import (
    build_bent_pipe_extruded_problem,
    build_layered_duct_extruded_problem,
    build_magnetic_obstacle_rect_extruded_problem,
    build_pipe_ogrid_extruded_problem,
    build_square_duct_extruded_problem,
    build_square_duct_fringing_benchmark,
    build_variable_field_duct_extruded_problem,
    build_variable_field_bent_pipe_extruded_problem,
    build_variable_field_layered_extruded_problem,
    build_variable_field_pipe_ogrid_extruded_problem,
    build_wham_mirror_pipe_extruded_problem,
    _cross_section_mesh,
    _pipe_conservative_current_diagnostics_3d,
    _pipe_conservative_emf_rhs_3d,
    _pipe_poisson_sparse_3d,
    _station_axial_current_from_fluxes,
    _poisson_jacobi_3d,
    _variable_coefficient_poisson_jacobi_3d,
    _variable_coefficient_poisson_sparse_3d,
    clone_case_with_field,
    magnetic_obstacle_literature_reference_cases,
    run_extruded_inductionless_slice,
    run_fringing_station_sweep,
    solve_extruded_inductionless,
    smooth_fringing_profile,
    validate_bent_pipe_low_de_baseline,
    validate_extruded_inductionless_solution,
    validate_magnetic_obstacle_benchmark,
    validate_magnetic_obstacle_baseline,
    validate_magnetic_obstacle_external_readiness,
    validate_magnetic_obstacle_literature_slice,
    validate_wham_mirror_pipe_baseline,
    validate_variable_field_pipe_solution,
    validate_variable_field_extruded_solution,
)
from lmx.specs import GeometrySpec
from lmx.specs import MagneticFieldSpec


pytestmark = pytest.mark.unit


def test_smooth_fringing_profile_produces_bounded_station_scales():
    profile = smooth_fringing_profile(
        length=6.0,
        nx=9,
        entry_center=1.5,
        exit_center=4.5,
        transition_width=0.3,
        peak_scale=1.2,
    )

    assert profile.axis == "z"
    assert profile.x.shape == (9,)
    assert jnp.all(profile.field_scale >= 0.0)
    assert float(jnp.max(profile.field_scale)) <= 1.2


def test_smooth_fringing_profile_rejects_invalid_axis():
    with pytest.raises(ValueError, match="Unsupported magnetic axis"):
        smooth_fringing_profile(
            length=1.0,
            nx=3,
            entry_center=0.2,
            exit_center=0.8,
            transition_width=0.1,
            axis="bad",
        )


def test_clone_case_with_field_replaces_constant_field():
    base_case, _ = build_square_duct_fringing_benchmark(nx_stations=5, ny=8, nz=8)
    shifted = clone_case_with_field(base_case, axis="y", magnitude=3.0, suffix="probe")

    assert shifted.name.endswith("probe")
    assert shifted.magnetic_field.value == (0.0, 3.0, 0.0)


def test_clone_case_with_field_supports_x_axis():
    base_case, _ = build_square_duct_fringing_benchmark(nx_stations=5, ny=8, nz=8)
    shifted = clone_case_with_field(base_case, axis="x", magnitude=2.0)
    assert shifted.magnetic_field.value == (2.0, 0.0, 0.0)


def test_clone_case_with_field_rejects_invalid_axis():
    base_case, _ = build_square_duct_fringing_benchmark(nx_stations=5, ny=8, nz=8)
    with pytest.raises(ValueError, match="Unsupported magnetic axis"):
        clone_case_with_field(base_case, axis="bad", magnitude=1.0)


def test_station_axial_current_from_fluxes_averages_adjacent_x_faces():
    fx = jnp.asarray([
        [[0.0, 1.0]],
        [[2.0, 3.0]],
        [[4.0, 5.0]],
    ])
    cell_area = jnp.asarray([[2.0, 4.0]])

    axial_current = _station_axial_current_from_fluxes(fx, cell_area)

    assert axial_current.shape == (2,)
    assert axial_current[0] == pytest.approx(10.0)
    assert axial_current[1] == pytest.approx(22.0)


def test_run_fringing_station_sweep_chains_initial_state(monkeypatch: pytest.MonkeyPatch):
    base_case, profile = build_square_duct_fringing_benchmark(nx_stations=3, ny=8, nz=8)
    calls: list[tuple[str, object]] = []

    class _State:
        def __init__(self, value: float):
            self.time = 0.0
            self.residual = value

    class _Solution:
        def __init__(self, value: float):
            self.state = _State(value)

    def fake_solver(case, initial_state=None):
        calls.append((case.name, initial_state))
        return _Solution(float(len(calls)))

    monkeypatch.setattr(
        "lmx.fringing.validation_summary",
        lambda solution, case_name, ha=None: {
            "u_max": 0.1,
            "mean_velocity": 0.2,
            "volumetric_flow_rate": 0.3,
            "current_scaled_pressure_proxy": 0.4,
        },
    )

    history = run_fringing_station_sweep(base_case, profile, solver=fake_solver)

    assert len(history) == 3
    assert calls[0][1] is None
    assert calls[1][1] is not None
    assert history[-1]["current_scaled_pressure_proxy"] == pytest.approx(0.4)


def test_run_fringing_station_sweep_requires_constant_field():
    base_case, profile = build_square_duct_fringing_benchmark(nx_stations=3, ny=8, nz=8)
    bad_case = replace(base_case, magnetic_field=replace(base_case.magnetic_field, kind="analytic", fn=lambda y, z: jnp.zeros(y.shape + (3,))))
    with pytest.raises(ValueError, match="constant-field"):
        run_fringing_station_sweep(bad_case, profile)


def test_run_extruded_inductionless_slice_stacks_station_fields():
    base_case, profile = build_square_duct_fringing_benchmark(nx_stations=4, ny=6, nz=6)
    shape = (base_case.geometry.ny, base_case.geometry.nz)
    y_centers = jnp.linspace(-1.0, 1.0, shape[0])
    z_centers = jnp.linspace(-1.0, 1.0, shape[1])

    class _State:
        def __init__(self, value: float):
            self.u = jnp.full(shape, value)
            self.phi = jnp.full(shape, 0.1 * value)
            self.jy = jnp.zeros(shape)
            self.jz = jnp.zeros(shape)
            self.lorentz_x = jnp.full(shape, 0.01 * value)
            self.time = 0.0
            self.residual = value

    class _Diagnostics:
        def __init__(self, value: float):
            self.volumetric_flow_rate_history = jnp.asarray([value])
            self.mean_velocity_history = jnp.asarray([0.5 * value])
            self.current_scaled_pressure_proxy_history = jnp.asarray([0.25 * value])
            self.charge_balance_residual_history = jnp.asarray([1.0e-6 * value])

    class _Solution:
        def __init__(self, value: float):
            self.mesh = type("Mesh", (), {"y_centers": y_centers, "z_centers": z_centers})()
            self.state = _State(value)
            self.diagnostics = _Diagnostics(value)

    call_count = {"value": 0}

    def fake_solver(case, initial_state=None):
        call_count["value"] += 1
        return _Solution(float(call_count["value"]))

    bundle = run_extruded_inductionless_slice(base_case, profile, solver=fake_solver)

    assert bundle.u.shape == (4, 6, 6)
    assert bundle.phi.shape == (4, 6, 6)
    assert bundle.v.shape == (4, 6, 6)
    assert bundle.w.shape == (4, 6, 6)
    assert bundle.p.shape == (4, 6, 6)
    assert bundle.jx.shape == (4, 6, 6)
    assert bundle.x.shape == (4,)
    assert bundle.y.shape == (6,)
    assert bundle.z.shape == (6,)
    assert bundle.geometry_kind == base_case.geometry.kind
    assert bundle.solver_kind == base_case.solver.kind
    assert jnp.isfinite(bundle.u).all()
    assert jnp.isfinite(bundle.axial_current).all()
    assert jnp.isfinite(bundle.wall_current_leakage).all()
    assert jnp.isfinite(bundle.charge_balance_residual).all()


def test_run_extruded_inductionless_slice_rejects_invalid_inputs():
    base_case, profile = build_square_duct_fringing_benchmark(nx_stations=3, ny=6, nz=6)
    bad_case = replace(base_case, geometry=replace(base_case.geometry, kind="pipe_ogrid"))
    with pytest.raises(ValueError, match="rectangular and layered ducts"):
        run_extruded_inductionless_slice(bad_case, profile)
    bad_field_case = replace(base_case, magnetic_field=replace(base_case.magnetic_field, kind="analytic", fn=lambda y, z: jnp.zeros(y.shape + (3,))))
    with pytest.raises(ValueError, match="constant-field"):
        run_extruded_inductionless_slice(bad_field_case, profile)


def test_build_square_duct_extruded_problem_marks_solver_family():
    problem = build_square_duct_extruded_problem(nx_stations=5, ny=8, nz=8)

    assert problem.case.solver.kind == "extruded_inductionless"
    assert problem.profile.x.shape == (5,)


def test_build_layered_duct_extruded_problem_marks_solver_family():
    problem = build_layered_duct_extruded_problem(nx_stations=5, ny=8, nz=8, wall_cells=1, insulator_cells=1)

    assert problem.case.solver.kind == "extruded_inductionless"
    assert problem.case.geometry.kind == "layered_duct"
    assert problem.profile.x.shape == (5,)


def test_build_pipe_ogrid_extruded_problem_marks_solver_family():
    problem = build_pipe_ogrid_extruded_problem(nx_stations=5, nr=6, ntheta=12)

    assert problem.case.solver.kind == "extruded_inductionless"
    assert problem.case.geometry.kind == "pipe_ogrid"
    assert problem.profile.x.shape == (5,)


def test_build_bent_pipe_extruded_problem_marks_solver_family():
    problem = build_bent_pipe_extruded_problem(nx_stations=5, nr=6, ntheta=12)

    assert problem.case.solver.kind == "extruded_inductionless"
    assert problem.case.geometry.kind == "bent_pipe"
    assert problem.profile.x.shape == (5,)


def test_build_wham_mirror_pipe_extruded_problem_marks_solver_family(tmp_path):
    x = np.linspace(-0.2, 0.2, 5)
    y = np.linspace(-0.1, 0.1, 5)
    z = np.linspace(-0.1, 0.1, 5)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    path = write_tabulated_field_npz(
        tmp_path / "field.npz",
        x=x,
        y=y,
        z=z,
        bx=np.zeros_like(xx),
        by=np.zeros_like(xx),
        bz=1.0 + 0.1 * np.cos(np.pi * xx / max(np.max(np.abs(x)), 1.0e-12)),
    )
    problem = build_wham_mirror_pipe_extruded_problem(
        table_path=str(path),
        radius=0.1,
        nr=4,
        ntheta=12,
        length=0.4,
        nx_stations=5,
    )
    assert problem.case.solver.kind == "extruded_inductionless"
    assert problem.case.geometry.kind == "pipe_ogrid"
    assert problem.case.magnetic_field.kind == "tabulated"
    assert np.allclose(np.asarray(problem.profile.field_scale, dtype=float), 1.0)


def test_cross_section_mesh_supports_pipe_ogrid_geometry():
    problem = build_square_duct_extruded_problem(nx_stations=3, ny=4, nz=4)
    pipe_case = replace(problem.case, geometry=GeometrySpec(kind="pipe_ogrid", width=1.0, height=1.0, radius=0.5, nr=4, ntheta=8))
    mesh = _cross_section_mesh(pipe_case)
    assert mesh.geometry == "pipe_ogrid"


def test_validate_extruded_inductionless_solution_reports_metrics():
    base_case, profile = build_square_duct_fringing_benchmark(nx_stations=4, ny=6, nz=6)
    bundle = run_extruded_inductionless_slice(
        base_case,
        profile,
        solver=lambda case, initial_state=None: type(
            "Solution",
            (),
            {
                "mesh": type("Mesh", (), {"y_centers": jnp.linspace(-1.0, 1.0, 6), "z_centers": jnp.linspace(-1.0, 1.0, 6)})(),
                "state": type(
                    "State",
                    (),
                    {
                        "u": jnp.ones((6, 6)),
                        "phi": jnp.zeros((6, 6)),
                        "jy": jnp.zeros((6, 6)),
                        "jz": jnp.zeros((6, 6)),
                        "lorentz_x": jnp.zeros((6, 6)),
                        "time": 0.0,
                        "residual": 1.0e-6,
                    },
                )(),
                "diagnostics": type(
                    "Diagnostics",
                    (),
                    {
                        "volumetric_flow_rate_history": jnp.asarray([1.0]),
                        "mean_velocity_history": jnp.asarray([0.5]),
                        "current_scaled_pressure_proxy_history": jnp.asarray([0.2]),
                        "charge_balance_residual_history": jnp.asarray([1.0e-7]),
                    },
                )(),
            },
        )(),
    )

    report = validate_extruded_inductionless_solution(bundle)
    assert report.station_count == 4
    assert report.max_charge_balance_residual >= 0.0
    assert report.axial_current_span >= 0.0
    assert report.peak_velocity_span >= 0.0
    assert report.pressure_span_range >= 0.0
    assert report.max_wall_current_leakage >= 0.0
    assert report.net_boundary_current_residual >= 0.0
    assert jnp.isfinite(report.field_mean_velocity_correlation)


def test_solve_extruded_inductionless_wraps_history_bundle_and_validation(monkeypatch: pytest.MonkeyPatch):
    problem = build_square_duct_extruded_problem(nx_stations=3, ny=6, nz=6)
    fake_bundle = type(
        "Bundle",
        (),
        {
            "x": jnp.asarray([0.0, 0.5, 1.0]),
            "field_scale": jnp.asarray([0.0, 1.0, 0.0]),
            "mean_velocity": jnp.asarray([0.1, 0.3, 0.12]),
            "volumetric_flow_rate": jnp.asarray([0.2, 0.4, 0.25]),
            "axial_current": jnp.asarray([0.01, 0.02, 0.015]),
            "wall_current_leakage": jnp.asarray([1.0e-6, 2.0e-6, 1.5e-6]),
            "boundary_current_residual": jnp.asarray([1.0e-7, 2.0e-7, 1.5e-7]),
            "residual": jnp.asarray([1.0e-4, 1.0e-5, 1.0e-5]),
            "charge_balance_residual": jnp.asarray([1.0e-8, 2.0e-8, 1.5e-8]),
            "y": jnp.asarray([0.0]),
            "z": jnp.asarray([0.0]),
            "u": jnp.ones((3, 1, 1)),
            "v": jnp.zeros((3, 1, 1)),
            "w": jnp.zeros((3, 1, 1)),
            "p": jnp.zeros((3, 1, 1)),
            "phi": jnp.zeros((3, 1, 1)),
            "jx": jnp.zeros((3, 1, 1)),
            "jy": jnp.zeros((3, 1, 1)),
            "jz": jnp.zeros((3, 1, 1)),
            "lorentz_x": jnp.zeros((3, 1, 1)),
            "lorentz_y": jnp.zeros((3, 1, 1)),
            "lorentz_z": jnp.zeros((3, 1, 1)),
            "current_scaled_pressure_proxy": jnp.asarray([0.1, 0.15, 0.1]),
            "geometry_kind": "rect_duct",
            "solver_kind": "extruded_inductionless",
        },
    )()
    monkeypatch.setattr("lmx.fringing._solve_extruded_projection", lambda problem, initial_bundle=None: fake_bundle)

    solution = solve_extruded_inductionless(problem)
    assert len(solution.station_history) == 3
    assert solution.validation.station_count == 3
    assert solution.bundle.solver_kind == "extruded_inductionless"
    assert "axial_current" in solution.station_history[0]
    assert "wall_current_leakage" in solution.station_history[0]
    assert "boundary_current_residual" in solution.station_history[0]
    assert "pressure_span" in solution.station_history[0]


def test_solve_extruded_inductionless_projection_returns_finite_rectangular_bundle():
    problem = build_square_duct_extruded_problem(ha_peak=8.0, nx_stations=4, ny=4, nz=4)

    solution = solve_extruded_inductionless(problem)

    assert solution.bundle.u.shape == (4, 4, 4)
    assert solution.bundle.p.shape == (4, 4, 4)
    assert solution.bundle.v.shape == (4, 4, 4)
    assert solution.bundle.w.shape == (4, 4, 4)
    assert solution.bundle.jx.shape == (4, 4, 4)
    assert jnp.isfinite(solution.bundle.u).all()
    assert jnp.isfinite(solution.bundle.p).all()
    assert jnp.isfinite(solution.bundle.axial_current).all()
    assert jnp.isfinite(solution.bundle.wall_current_leakage).all()
    assert solution.validation.max_wall_current_leakage >= 0.0
    assert solution.validation.net_boundary_current_residual >= 0.0
    assert solution.validation.station_count == 4


def test_rectangular_projection_uses_sparse_electric_solve(monkeypatch: pytest.MonkeyPatch):
    problem = build_square_duct_extruded_problem(ha_peak=8.0, nx_stations=3, ny=4, nz=4)
    sparse_calls = {"count": 0}
    jacobi_calls = {"count": 0}

    def wrapped_sparse(*args, **kwargs):
        sparse_calls["count"] += 1
        return _variable_coefficient_poisson_sparse_3d(*args, **kwargs)

    def wrapped_jacobi(*args, **kwargs):
        jacobi_calls["count"] += 1
        return _variable_coefficient_poisson_jacobi_3d(*args, **kwargs)

    monkeypatch.setattr("lmx.fringing._variable_coefficient_poisson_sparse_3d", wrapped_sparse)
    monkeypatch.setattr("lmx.fringing._variable_coefficient_poisson_jacobi_3d", wrapped_jacobi)

    solve_extruded_inductionless(problem)

    assert sparse_calls["count"] > 0
    assert jacobi_calls["count"] == 0


def test_projection_solver_can_break_early_with_loose_tolerance():
    problem = build_square_duct_extruded_problem(ha_peak=4.0, nx_stations=3, ny=4, nz=4)
    loose_problem = replace(problem, case=replace(problem.case, solver=replace(problem.case.solver, coupling_tolerance=10.0)))
    solution = solve_extruded_inductionless(loose_problem)
    assert solution.validation.max_residual <= 10.0


def test_solve_extruded_inductionless_projection_returns_finite_layered_bundle():
    problem = build_layered_duct_extruded_problem(
        ha_peak=8.0,
        nx_stations=4,
        ny=4,
        nz=4,
        wall_cells=1,
        insulator_cells=1,
    )

    solution = solve_extruded_inductionless(problem)

    assert solution.bundle.u.shape[0] == 4
    assert solution.bundle.geometry_kind == "layered_duct"
    assert jnp.isfinite(solution.bundle.u).all()
    assert jnp.isfinite(solution.bundle.phi).all()
    assert solution.validation.max_charge_balance_residual < 1.0e-4
    assert solution.validation.net_boundary_current_residual == pytest.approx(0.0)


def test_layered_projection_keeps_throughput_span_bounded_on_heavier_case():
    problem = build_layered_duct_extruded_problem(
        ha_peak=20.0,
        nx_stations=5,
        ny=6,
        nz=6,
        wall_cells=1,
        insulator_cells=1,
    )
    problem = replace(
        problem,
        case=replace(
            problem.case,
            time_stepper=replace(problem.case.time_stepper, max_steps=12, potential_iterations=48),
            solver=replace(problem.case.solver, coupling_iterations=8),
        ),
    )

    solution = solve_extruded_inductionless(problem)

    assert solution.validation.volumetric_flow_rate_span < 5.0e-3
    assert solution.validation.field_mean_velocity_correlation < -5.0e-1
    assert solution.validation.max_charge_balance_residual < 1.0e-4
    assert solution.validation.axial_current_mirror_residual < 1.0e-3
    assert solution.validation.pressure_span_mirror_residual < 1.0e-3
    assert abs(solution.validation.center_axial_current) < 1.0e-4


def test_solve_extruded_inductionless_projection_returns_finite_pipe_bundle():
    problem = build_pipe_ogrid_extruded_problem(ha_peak=6.0, nx_stations=4, nr=4, ntheta=8)

    solution = solve_extruded_inductionless(problem)

    assert solution.bundle.geometry_kind == "pipe_ogrid"
    assert solution.bundle.u.shape == (4, 4, 8)
    assert jnp.isfinite(solution.bundle.u).all()
    assert jnp.isfinite(solution.bundle.axial_current).all()
    assert jnp.isfinite(solution.bundle.wall_current_leakage).all()
    assert float(jnp.max(jnp.abs(solution.bundle.u[:, -1, :]))) > 0.0
    assert solution.validation.max_charge_balance_residual < 0.5
    assert solution.validation.net_boundary_current_residual == pytest.approx(0.0)


def test_pipe_sparse_potential_cancels_conservative_emf_divergence():
    nx, nr, ntheta = 4, 5, 12
    dx = 0.3
    r_faces = jnp.linspace(0.0, 0.4, nr + 1)
    r_centers = 0.5 * (r_faces[:-1] + r_faces[1:])
    dtheta = 2.0 * jnp.pi / ntheta
    x = jnp.linspace(-1.0, 1.0, nx)[:, None, None]
    r = r_centers[None, :, None]
    theta = jnp.linspace(0.0, 2.0 * jnp.pi, ntheta, endpoint=False)[None, None, :]
    sigma = jnp.ones((nx, nr, ntheta))
    uxb_x = jnp.broadcast_to(0.03 * jnp.sin(jnp.pi * x) * jnp.cos(theta), sigma.shape)
    uxb_r = jnp.broadcast_to(0.02 * r * jnp.cos(2.0 * theta), sigma.shape)
    uxb_theta = jnp.broadcast_to(0.01 * jnp.sin(theta) * (1.0 + x), sigma.shape)

    emf_rhs = _pipe_conservative_emf_rhs_3d(
        sigma,
        uxb_x,
        uxb_r,
        uxb_theta,
        dx=dx,
        r_faces=r_faces,
        r_centers=r_centers,
        dtheta=float(dtheta),
    )
    phi, residual, _, _ = _pipe_poisson_sparse_3d(
        -emf_rhs,
        sigma,
        dx=dx,
        r_faces=r_faces,
        r_centers=r_centers,
        dtheta=float(dtheta),
        iterations=40,
        tolerance=1.0e-12,
    )
    div_j = _pipe_conservative_current_diagnostics_3d(
        sigma,
        phi,
        uxb_x,
        uxb_r,
        uxb_theta,
        dx=dx,
        r_faces=r_faces,
        r_centers=r_centers,
        dtheta=float(dtheta),
    )[0]

    assert residual < 1.0e-10
    assert float(jnp.max(jnp.abs(div_j))) < 1.0e-10


def test_wham_mirror_pipe_baseline_reports_finite_metrics(tmp_path):
    x = np.linspace(-0.2, 0.2, 7)
    y = np.linspace(-0.12, 0.12, 7)
    z = np.linspace(-0.12, 0.12, 7)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    bz = 1.0 + 0.3 * np.exp(-(xx / 0.12) ** 2)
    path = write_tabulated_field_npz(
        tmp_path / "wham_small.npz",
        x=x,
        y=y,
        z=z,
        bx=np.zeros_like(xx),
        by=np.zeros_like(xx),
        bz=bz,
    )
    problem = build_wham_mirror_pipe_extruded_problem(
        table_path=str(path),
        radius=0.12,
        nr=4,
        ntheta=12,
        length=0.4,
        nx_stations=7,
    )
    problem = replace(
        problem,
        profile=replace(problem.profile, x=jnp.asarray(x, dtype=float)),
        case=replace(
            problem.case,
            time_stepper=replace(problem.case.time_stepper, max_steps=4, potential_iterations=12),
            solver=replace(problem.case.solver, coupling_iterations=4),
        ),
    )
    solution = solve_extruded_inductionless(problem)
    metrics = validate_wham_mirror_pipe_baseline(solution)
    assert np.isfinite(metrics["pressure_drop_proxy"])
    assert np.isfinite(metrics["field_velocity_correlation"])
    assert np.isfinite(metrics["max_charge_balance_residual"])


def test_solve_extruded_inductionless_projection_returns_finite_bent_pipe_bundle():
    bent_problem = build_bent_pipe_extruded_problem(
        ha_peak=6.0,
        bend_radius=4.0,
        bend_angle=1.0,
        nx_stations=4,
        nr=4,
        ntheta=8,
    )
    straight_problem = build_pipe_ogrid_extruded_problem(
        ha_peak=6.0,
        radius=float(bent_problem.case.geometry.radius),
        length=float(bent_problem.case.geometry.length),
        nx_stations=4,
        nr=4,
        ntheta=8,
    )
    straight_problem = replace(straight_problem, profile=bent_problem.profile)

    bent_solution = solve_extruded_inductionless(bent_problem)
    straight_solution = solve_extruded_inductionless(straight_problem)
    validation = validate_bent_pipe_low_de_baseline(bent_solution, straight_solution)

    assert bent_solution.bundle.geometry_kind == "bent_pipe"
    assert jnp.isfinite(bent_solution.bundle.u).all()
    assert validation["dean_number"] >= 0.0
    assert validation["dean_vortex_observables_available"] is True
    assert validation["secondary_flow_rms_ratio"] >= 0.0
    assert validation["secondary_flow_peak_ratio"] >= 0.0
    assert np.isfinite(validation["normalized_velocity_centroid_shift"])
    assert np.isfinite(validation["inner_outer_velocity_ratio"])
    assert validation["research_grade_dean_validation_pass"] is False
    assert isinstance(validation["research_grade_charge_balance_pass"], bool)
    assert validation["research_grade_charge_balance_tolerance"] < validation["bounded_charge_balance_tolerance"]
    assert validation["cross_section_l2_error"] <= 0.2
    assert isinstance(validation["validation_pass"], bool)


def test_solve_extruded_inductionless_supports_analytic_variable_field():
    problem = build_variable_field_duct_extruded_problem(nx_stations=7, ny=10, nz=10)
    solution = solve_extruded_inductionless(problem)
    validation = validate_variable_field_extruded_solution(solution, field_ny=41, field_nz=41)

    assert solution.bundle.geometry_kind == "rect_duct"
    assert jnp.all(jnp.isfinite(solution.bundle.u))
    assert validation["mean_velocity_change"] > 0.0
    assert validation["current_proxy_change"] > 0.0
    assert isinstance(validation["validation_pass"], bool)


def test_solve_extruded_inductionless_supports_layered_analytic_variable_field():
    problem = build_variable_field_layered_extruded_problem(nx_stations=7, ny=10, nz=10)
    solution = solve_extruded_inductionless(problem)
    validation = validate_variable_field_extruded_solution(solution, field_ny=41, field_nz=41)

    assert solution.bundle.geometry_kind == "layered_duct"
    assert jnp.all(jnp.isfinite(solution.bundle.u))
    assert validation["mean_velocity_change"] > 0.0
    assert isinstance(validation["validation_pass"], bool)


def test_solve_extruded_inductionless_supports_tabulated_variable_field(tmp_path):
    field_fn = make_divergence_free_cross_section_field(width=2.4, height=1.6, base_bz=12.0, perturbation=0.12)
    y, z, field = sample_cross_section_field(field_fn, width=2.4, height=1.6, ny=41, nz=41)
    path = write_tabulated_field_npz(
        tmp_path / "field.npz",
        y=y,
        z=z,
        bx=field[..., 0],
        by=field[..., 1],
        bz=field[..., 2],
    )
    problem = build_square_duct_extruded_problem(nx_stations=7, ny=10, nz=10, width=2.4, height=1.6, ha_peak=12.0)
    problem = replace(problem, case=replace(problem.case, magnetic_field=MagneticFieldSpec(kind="tabulated", table_path=str(path))))
    solution = solve_extruded_inductionless(problem)
    validation = validate_variable_field_extruded_solution(solution, field_ny=41, field_nz=41)

    assert solution.bundle.geometry_kind == "rect_duct"
    assert validation["rms_divergence"] >= 0.0
    assert isinstance(validation["validation_pass"], bool)


def test_solve_extruded_inductionless_supports_variable_field_pipe_and_bent_pipe():
    straight_problem = build_variable_field_pipe_ogrid_extruded_problem(nx_stations=7, nr=8, ntheta=16)
    bent_problem = build_variable_field_bent_pipe_extruded_problem(nx_stations=7, nr=8, ntheta=16)
    straight_solution = solve_extruded_inductionless(straight_problem)
    bent_solution = solve_extruded_inductionless(bent_problem)

    pipe_validation = validate_variable_field_pipe_solution(straight_solution, field_ny=41, field_nz=41)
    bent_field_validation = validate_variable_field_pipe_solution(bent_solution, field_ny=41, field_nz=41)
    bent_low_de_validation = validate_bent_pipe_low_de_baseline(bent_solution, straight_solution)

    assert straight_solution.bundle.geometry_kind == "pipe_ogrid"
    assert bent_solution.bundle.geometry_kind == "bent_pipe"
    assert pipe_validation["current_proxy_change"] > 0.0
    assert bent_field_validation["current_proxy_change"] > 0.0
    assert isinstance(bent_low_de_validation["validation_pass"], bool)


def test_magnetic_obstacle_baseline_reports_velocity_deficit():
    problem = build_magnetic_obstacle_rect_extruded_problem(nx_stations=9, ny=12, nz=12)
    solution = solve_extruded_inductionless(problem)
    validation = validate_magnetic_obstacle_baseline(solution, field_ny=41, field_nz=41)

    assert solution.bundle.geometry_kind == "rect_duct"
    assert validation["obstacle_velocity_deficit"] > 0.0
    assert validation["current_proxy_peak"] > 0.0
    assert validation["divergence_to_field_ratio"] >= 0.0
    assert validation["field_quality_pass"] in {True, False}
    assert validation["reference_kind"] == "none"
    assert validation["external_reference_available"] is False
    assert validation["research_grade_validation_pass"] is False
    assert isinstance(validation["validation_pass"], bool)


def test_magnetic_obstacle_benchmark_reports_normalized_response():
    problem = build_magnetic_obstacle_rect_extruded_problem(base_bz=60.0, nx_stations=9, ny=12, nz=12, forcing=2.0)
    problem = replace(
        problem,
        case=replace(
            problem.case,
            time_stepper=replace(problem.case.time_stepper, max_steps=12, potential_iterations=24),
            solver=replace(problem.case.solver, coupling_iterations=6),
        ),
    )
    solution = solve_extruded_inductionless(problem)
    reference_problem = replace(problem, profile=replace(problem.profile, field_scale=jnp.zeros_like(problem.profile.field_scale)))
    reference_solution = solve_extruded_inductionless(reference_problem)
    validation = validate_magnetic_obstacle_benchmark(solution, reference_solution, field_ny=41, field_nz=41)

    assert solution.bundle.geometry_kind == "rect_duct"
    assert validation["current_proxy_peak"] > 0.0
    assert validation["peak_pressure_excess"] >= 0.0
    assert validation["peak_velocity_deficit_ratio"] >= 0.0
    assert validation["integrated_velocity_deficit_ratio"] >= 0.0
    assert validation["peak_centerline_deficit_ratio"] >= 0.0
    assert validation["peak_centerline_station_deficit_ratio"] >= 0.0
    assert validation["recovery_station"] >= float(solution.bundle.x[0])
    assert validation["y_l2_distortion"] > 0.0
    assert validation["z_l2_distortion"] > 0.0
    assert validation["y_peak_cut_abs_error"] >= 0.0
    assert validation["z_peak_cut_abs_error"] >= 0.0
    assert validation["peak_crosscut_distortion"] == pytest.approx(max(validation["y_l2_distortion"], validation["z_l2_distortion"]))
    assert validation["reference_kind"] == "matched_no_field_lmx"
    assert validation["external_reference_available"] is False
    assert validation["internal_response_pass"] == validation["benchmark_pass"]
    assert validation["research_grade_validation_pass"] is False
    assert isinstance(validation["benchmark_pass"], bool)


def test_magnetic_obstacle_literature_slice_reports_recovery_metrics():
    problem = build_magnetic_obstacle_rect_extruded_problem(base_bz=60.0, nx_stations=9, ny=12, nz=12, forcing=2.0)
    problem = replace(
        problem,
        case=replace(
            problem.case,
            time_stepper=replace(problem.case.time_stepper, max_steps=12, potential_iterations=24),
            solver=replace(problem.case.solver, coupling_iterations=6),
        ),
    )
    solution = solve_extruded_inductionless(problem)
    reference_problem = replace(problem, profile=replace(problem.profile, field_scale=jnp.zeros_like(problem.profile.field_scale)))
    reference_solution = solve_extruded_inductionless(reference_problem)
    validation = validate_magnetic_obstacle_literature_slice(solution, reference_solution, field_ny=41, field_nz=41)
    references = magnetic_obstacle_literature_reference_cases()
    readiness = validate_magnetic_obstacle_external_readiness(solution, field_ny=41, field_nz=41)

    assert validation["peak_station"] >= float(solution.bundle.x[0])
    assert validation["recovery_distance"] >= 0.0
    assert 0.0 <= validation["normalized_recovery_distance"] <= 1.0
    assert validation["literature_shape_gate"] in {True, False}
    assert validation["literature_status"] == "internal_lmx_response_only"
    assert validation["external_reference_available"] is False
    assert validation["research_grade_validation_pass"] is False
    assert validation["literature_pass"] is False
    assert "votyakov_zienicke_kolesnikov_jfm" in references
    assert readiness["reference_case"] == "votyakov_zienicke_kolesnikov_jfm"
    assert "centerline_velocity_deficit_ratio" in readiness["observables"]
    assert readiness["external_reference_available"] is False
    assert readiness["research_grade_validation_pass"] is False


def test_poisson_helpers_can_stop_early():
    rhs = jnp.zeros((2, 2, 2))
    field, residual, iterations, initial = _poisson_jacobi_3d(rhs, dx=1.0, dy=1.0, dz=1.0, iterations=4, tolerance=1.0)
    assert iterations == 1
    assert residual <= initial
    conductivity = jnp.ones((2, 2, 2))
    field_var, residual_var, iterations_var, initial_var = _variable_coefficient_poisson_jacobi_3d(
        rhs,
        conductivity,
        dx=1.0,
        dy=1.0,
        dz=1.0,
        iterations=4,
        tolerance=1.0,
    )
    assert iterations_var == 1
    assert residual_var <= initial_var
    field_sparse, residual_sparse, iterations_sparse, initial_sparse = _variable_coefficient_poisson_sparse_3d(
        rhs,
        conductivity,
        dx=1.0,
        dy=1.0,
        dz=1.0,
        iterations=4,
        tolerance=1.0,
    )
    assert iterations_sparse >= 1
    assert residual_sparse <= initial_sparse
    assert jnp.isfinite(field).all()
    assert jnp.isfinite(field_var).all()
    assert jnp.isfinite(field_sparse).all()


def test_solve_extruded_inductionless_uses_projection_for_pipe_geometry(monkeypatch: pytest.MonkeyPatch):
    problem = build_square_duct_extruded_problem(nx_stations=3, ny=4, nz=4)
    pipe_case = replace(problem.case, geometry=GeometrySpec(kind="pipe_ogrid", width=1.0, height=1.0, radius=0.5, nr=4, ntheta=8))
    pipe_problem = replace(problem, case=pipe_case)
    monkeypatch.setattr(
        "lmx.fringing._solve_extruded_projection",
        lambda problem, initial_bundle=None: type(
            "Bundle",
            (),
            {
                "x": jnp.asarray([0.0]),
                "field_scale": jnp.asarray([1.0]),
                "mean_velocity": jnp.asarray([0.1]),
                "volumetric_flow_rate": jnp.asarray([0.2]),
                "axial_current": jnp.asarray([0.0]),
                "wall_current_leakage": jnp.asarray([0.0]),
                "boundary_current_residual": jnp.asarray([0.0]),
                "residual": jnp.asarray([1.0e-4]),
                "charge_balance_residual": jnp.asarray([1.0e-6]),
                "y": jnp.asarray([0.0]),
                "z": jnp.asarray([0.0]),
                "u": jnp.zeros((1, 1, 1)),
                "v": jnp.zeros((1, 1, 1)),
                "w": jnp.zeros((1, 1, 1)),
                "p": jnp.zeros((1, 1, 1)),
                "phi": jnp.zeros((1, 1, 1)),
                "jx": jnp.zeros((1, 1, 1)),
                "jy": jnp.zeros((1, 1, 1)),
                "jz": jnp.zeros((1, 1, 1)),
                "lorentz_x": jnp.zeros((1, 1, 1)),
                "lorentz_y": jnp.zeros((1, 1, 1)),
                "lorentz_z": jnp.zeros((1, 1, 1)),
                "current_scaled_pressure_proxy": jnp.asarray([0.0]),
                "geometry_kind": "pipe_ogrid",
                "solver_kind": "extruded_inductionless",
            },
        )(),
    )
    solution = solve_extruded_inductionless(pipe_problem)
    assert solution.validation.station_count == 1


def test_solve_extruded_inductionless_projection_accepts_matching_initial_bundle():
    problem = build_square_duct_extruded_problem(ha_peak=5.0, nx_stations=3, ny=4, nz=4)
    first = solve_extruded_inductionless(problem)

    resumed = solve_extruded_inductionless(problem, initial_bundle=first.bundle)

    assert resumed.bundle.u.shape == first.bundle.u.shape
    assert jnp.isfinite(resumed.bundle.u).all()
    assert resumed.validation.station_count == first.validation.station_count
