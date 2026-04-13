import jax.numpy as jnp
import pytest

from lmx.fringing import (
    build_square_duct_fringing_benchmark,
    clone_case_with_field,
    run_extruded_inductionless_slice,
    run_fringing_station_sweep,
    smooth_fringing_profile,
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
    assert bundle.x.shape == (4,)
    assert bundle.y.shape == (6,)
    assert bundle.z.shape == (6,)
    assert bundle.geometry_kind == base_case.geometry.kind
    assert bundle.solver_kind == base_case.solver.kind
    assert jnp.isfinite(bundle.u).all()
    assert jnp.isfinite(bundle.charge_balance_residual).all()
