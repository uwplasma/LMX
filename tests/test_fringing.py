import jax.numpy as jnp
import pytest

from lmx.fringing import (
    build_square_duct_extruded_problem,
    build_square_duct_fringing_benchmark,
    clone_case_with_field,
    run_extruded_inductionless_slice,
    run_fringing_station_sweep,
    solve_extruded_inductionless,
    smooth_fringing_profile,
    validate_extruded_inductionless_solution,
)


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


def test_clone_case_with_field_replaces_constant_field():
    base_case, _ = build_square_duct_fringing_benchmark(nx_stations=5, ny=8, nz=8)
    shifted = clone_case_with_field(base_case, axis="y", magnitude=3.0, suffix="probe")

    assert shifted.name.endswith("probe")
    assert shifted.magnetic_field.value == (0.0, 3.0, 0.0)


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
    assert jnp.isfinite(bundle.charge_balance_residual).all()


def test_build_square_duct_extruded_problem_marks_solver_family():
    problem = build_square_duct_extruded_problem(nx_stations=5, ny=8, nz=8)

    assert problem.case.solver.kind == "extruded_inductionless"
    assert problem.profile.x.shape == (5,)


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
    monkeypatch.setattr("lmx.fringing._solve_rectangular_extruded_projection", lambda problem: fake_bundle)

    solution = solve_extruded_inductionless(problem)
    assert len(solution.station_history) == 3
    assert solution.validation.station_count == 3
    assert solution.bundle.solver_kind == "extruded_inductionless"


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
    assert solution.validation.station_count == 4
