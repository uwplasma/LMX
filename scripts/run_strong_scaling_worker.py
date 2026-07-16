from __future__ import annotations

# ruff: noqa: E402 -- repository-root bootstrap must precede project imports.

import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import resource
import sys
import tempfile
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import NamedSharding, PartitionSpec as P

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import lmx
from lmx.fringing import (
    _axial_field_sharding,
    _explicit_deviatoric_stress_duct,
    _face_flux_pressure_projection_duct,
    _flow_rate_inlet_profile,
    _initialize_duct_mass_flux,
    _solvax_implicit_momentum_duct,
    _solvax_pressure_poisson_duct,
    _unpack_duct_mass_flux,
)
from lmx.scaling import (
    _bundle_memory_bytes,
    benchmark_extruded_inductionless_solve,
    benchmark_sharded_extruded_operator,
    summarize_pressure_linear_history,
)

_B2_RESTART_FLUX_ATOL = 1.0e-6
_B2_RESTART_FLUX_RTOL = 1.0e-5
_B2_REPEAT_ATOL = 2.0e-9
_B2_REPEAT_RTOL = 2.0e-8
_B2_PROFILE_ITERATION_ATOL = 3
_B2_FIELD_NAMES = (
    "u", "v", "w", "p", "phi", "jx", "jy", "jz", "rho_phi_plus", "rho_phi_inlet"
)

if ROOT not in Path(lmx.__file__).resolve().parents:
    raise RuntimeError(
        f"Scaling worker imported LMX outside its source tree: {lmx.__file__}"
    )


def _source_fingerprint() -> str:
    """Hash the source and frozen specifications used by this worker."""

    paths = [*sorted((ROOT / "lmx").glob("*.py")), Path(__file__).resolve(),
        ROOT / "scripts" / "run_freemhd_parity_suite.py"]
    paths.extend(
        sorted(
            path
            for path in (ROOT / "benchmarks" / "specs").rglob("*")
            if path.is_file()
        )
    )
    paths.extend(sorted((ROOT / "benchmarks" / "references").glob("*.csv")))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _placement(array) -> dict[str, int | bool]:
    return {"addressable_shards": len(array.addressable_shards),
        "global_shards": len(array.global_shards),
        "replicated": bool(array.sharding.is_fully_replicated)}


def _max_abs_difference(left, right) -> float:
    """Return a fail-closed maximum difference for restart state arrays."""

    left, right = np.asarray(left), np.asarray(right)
    if left.shape != right.shape or not (
        np.all(np.isfinite(left)) and np.all(np.isfinite(right))
    ):
        return float("inf")
    return float(np.max(np.abs(left - right), initial=0.0))


def _anderson_diagnostics(
    problem, checkpoint, direct, resumed, serialized, *, num_devices: int
) -> dict[str, object]:
    """Audit the bounded B2 depth-two state without timing serialization."""

    acceleration = problem.case.solver.coupling_acceleration
    depth = int(problem.case.solver.coupling_history_depth)
    schema = str(serialized.metadata.get("restart_schema", "unknown"))
    result: dict[str, object] = {
        "coupling_acceleration": acceleration,
        "coupling_history_depth": depth,
        "restart_schema": schema,
        "schema6_active": acceleration == "anderson",
        "anderson_state_all_or_none": True,
        "anderson_depth_two_update_executed": False,
        "anderson_serialized_max_abs": None,
        "anderson_replay_max_abs": None,
        "anderson_replay_field_max_abs": None,
        "anderson_replay_flux_max_abs": None,
        "anderson_replay_flux_relative_l2": None,
        "anderson_gram": None,
        "anderson_weights": None,
        "anderson_weights_sum_error": None,
        "anderson_state_placement": {},
        "anderson_validation_passed": True,
    }
    if acceleration != "anderson":
        return result

    states = tuple(getattr(bundle, "anderson_state", None) for bundle in (
        checkpoint, direct, resumed, serialized.bundle))
    all_or_none = all(
        state is not None and len(state) == 4 and all(value is not None for value in state)
        for state in states
    )
    result["anderson_state_all_or_none"] = all_or_none
    if not all_or_none:
        result["anderson_validation_passed"] = False
        return result

    checkpoint_state, direct_state, resumed_state, serialized_state = states
    serialized_max = max(
        _max_abs_difference(left, right)
        for left, right in zip(checkpoint_state, serialized_state, strict=True)
    )
    replay_differences = tuple(
        _max_abs_difference(left, right)
        for left, right in zip(direct_state, resumed_state, strict=True)
    )
    replay_max = max(replay_differences)
    replay_field_max, replay_flux_max = (
        max(replay_differences[:2]), max(replay_differences[2:])
    )
    replay_flux_relative = max(
        float(np.linalg.norm(np.asarray(left) - np.asarray(right)) /
            max(np.linalg.norm(np.asarray(left)), np.linalg.norm(np.asarray(right)), 1.0e-30))
        for left, right in zip(direct_state[2:], resumed_state[2:], strict=True)
    )
    from solvax import anderson_weights

    residuals = jnp.stack((checkpoint_state[1], direct_state[1]))
    flat = residuals.reshape((2, -1))
    gram = flat @ flat.T
    weights = anderson_weights(
        residuals,
        regularization=problem.case.solver.coupling_regularization,
    )
    jax.block_until_ready((gram, weights))
    gram_np, weights_np = np.asarray(gram), np.asarray(weights)
    state_placement = {
        name: _placement(value)
        for name, value in zip(
            ("mapped", "residual", "rho_phi_plus", "rho_phi_inlet"),
            direct_state,
            strict=True,
        )
    }
    placement_passed = all(
        value["global_shards"] == num_devices
        and (num_devices == 1 or value["replicated"] == (name == "rho_phi_inlet"))
        for name, value in state_placement.items()
    )
    depth_two = (
        depth == 2
        and checkpoint.stopping_state[0] == 1
        and direct.stopping_state[0] == resumed.stopping_state[0] == 2
    )
    weights_sum_error = float(abs(np.sum(weights_np) - 1.0))
    valid = bool(
        schema == "b2_diagnostics_v6"
        and depth_two
        and serialized_max <= 1.0e-12
        and replay_field_max <= 1.0e-12
        and replay_flux_max <= _B2_RESTART_FLUX_ATOL
        and replay_flux_relative <= _B2_RESTART_FLUX_RTOL
        and placement_passed
        and np.all(np.isfinite(gram_np))
        and np.allclose(gram_np, gram_np.T, rtol=0.0, atol=1.0e-12)
        and np.all(np.isfinite(weights_np))
        and weights_sum_error <= 1.0e-12
    )
    result.update(
        anderson_depth_two_update_executed=depth_two,
        anderson_serialized_max_abs=serialized_max,
        anderson_replay_max_abs=replay_max,
        anderson_replay_field_max_abs=replay_field_max,
        anderson_replay_flux_max_abs=replay_flux_max,
        anderson_replay_flux_relative_l2=replay_flux_relative,
        anderson_gram=gram_np.tolist(),
        anderson_weights=weights_np.tolist(),
        anderson_weights_sum_error=weights_sum_error,
        anderson_state_placement=state_placement,
        anderson_validation_passed=valid,
    )
    return result


def _b2_ready_arrays(bundle) -> tuple[object, ...]:
    """Include accelerator state in the timed completion barrier when present."""

    fields = tuple(getattr(bundle, name) for name in _B2_FIELD_NAMES)
    anderson = getattr(bundle, "anderson_state", None)
    return fields if anderson is None else (*fields, *anderson)


def _b2_repeat_signature(bundle) -> np.ndarray:
    """Compress every sharded B2 field by axial station for repeat gates."""

    values = []
    for name in ("u", "v", "w", "p", "phi", "jx", "jy", "jz", "rho_phi_plus"):
        array = np.asarray(getattr(bundle, name), dtype=float)
        if name == "phi":
            array = array - np.mean(array)
        axial_axis = 1 if name == "rho_phi_plus" else 0
        stations = np.moveaxis(array, axial_axis, 0).reshape(array.shape[axial_axis], -1)
        values.append(np.stack((np.min(stations, axis=1), np.max(stations, axis=1),
            np.mean(stations, axis=1), np.linalg.norm(stations, axis=1))).reshape(-1))
    values.append(np.asarray(bundle.rho_phi_inlet, dtype=float).reshape(-1))
    return np.concatenate(values)


def _duct_step_gate(*, nx: int, ny: int, nz: int, iterations: int, num_devices: int):
    """Run one tiny production-faithful compact-flux projection/momentum step."""

    if (nx, ny, nz) != (8, 4, 3):
        raise ValueError("duct_step_gate requires the fixed 8x4x3 mesh")
    if nx % num_devices:
        raise ValueError("duct_step_gate axial cells must divide the device count")
    dx, dt = 0.25, 0.02
    dy, dz = jnp.asarray([0.18, 0.22, 0.27, 0.23]), jnp.asarray([0.21, 0.31, 0.28])
    x = np.linspace(-0.9, 0.8, nx)[:, None, None]
    y = np.linspace(-0.8, 0.7, ny)[None, :, None]
    z = np.linspace(-0.6, 0.9, nz)[None, None, :]
    shape = (nx, ny, nz)
    velocity = np.stack((
        np.broadcast_to(0.24 + 0.035 * x + 0.012 * y - 0.009 * z, shape),
        np.broadcast_to(0.018 * np.sin(np.pi * x) * (1.0 - y**2) + 0.003 * z, shape),
        np.broadcast_to(-0.014 * np.cos(np.pi * x) * (1.0 - z**2) + 0.002 * y, shape)), -1)
    force = np.stack((np.broadcast_to(0.08 + 0.01 * y, shape),
        np.broadcast_to(-0.025 + 0.006 * z, shape),
        np.broadcast_to(0.018 - 0.004 * y, shape)), -1)
    density, viscosity = np.ones(shape), np.full(shape, 0.035)
    pressure, mask, target_flow = np.zeros(shape), np.ones(shape, dtype=bool), 0.16
    mesh = _axial_field_sharding(num_devices).mesh
    field_sharding = NamedSharding(mesh, P("x", None, None))
    vector_sharding = NamedSharding(mesh, P("x", None, None, None))
    flux_sharding = NamedSharding(mesh, P(None, "x", None, None))
    axial_sharding, replicated = NamedSharding(mesh, P("x")), NamedSharding(mesh, P())
    velocity, force = (jax.device_put(value, vector_sharding) for value in (velocity, force))
    density, viscosity, pressure, mask = (jax.device_put(value, field_sharding)
        for value in (density, viscosity, pressure, mask))
    area = dy[:, None] * dz[None, :]

    def initialize(velocity0, density0):
        inlet = velocity0[0].at[..., 0].set(
            _flow_rate_inlet_profile(velocity0[0, ..., 0], area, target_flow))
        return _initialize_duct_mass_flux(
            velocity0, density0, inlet, dx=dx, dy=dy, dz=dz,
            sharding=field_sharding)

    initialize = jax.jit(initialize,
        in_shardings=(vector_sharding, field_sharding),
        out_shardings=(field_sharding,) * 3 + (replicated,))
    *initial_components, initial_inlet = initialize(velocity, density)
    pack_flux = jax.jit(lambda x, y, z: jnp.stack((x, y, z)),
        in_shardings=(field_sharding,) * 3, out_shardings=flux_sharding)
    initial_plus = pack_flux(*initial_components)

    def project(velocity0, pressure0, density0, mask0):
        return _face_flux_pressure_projection_duct(
            velocity0[..., 0], velocity0[..., 1], velocity0[..., 2], density0, mask0,
            inlet_flow_rate=target_flow, dt=dt, dx=dx, dy=dy, dz=dz,
            iterations=iterations, tolerance=1.0e-10, fluid_bounds=(0, ny, 0, nz),
            initial_pressure=pressure0, single_reduction=True, include_axial_line=False,
            field_sharding=field_sharding)

    project = jax.jit(project,
        in_shardings=(vector_sharding, field_sharding, field_sharding, field_sharding),
        out_shardings=(field_sharding,) * 4
        + (axial_sharding, replicated, replicated)
        + (field_sharding,) * 3 + (replicated,) * 6)
    projected = project(velocity, pressure, density, mask)
    projected_velocity, rho_phi_plus = jax.jit(
        lambda u, v, w, x, y, z: (jnp.stack((u, v, w), axis=-1),
            jnp.stack((x, y, z))),
        in_shardings=(field_sharding,) * 6,
        out_shardings=(vector_sharding, flux_sharding))(
            *projected[:3], *projected[7:10])
    rho_phi_inlet = projected[10]

    probe_y, probe_z = (jnp.linspace(-0.7, 0.7, 5)[None, :, None],
        jnp.linspace(-0.6, 0.6, 5)[None, None, :])
    probe_rhs = jnp.broadcast_to(0.3 * jnp.sin(2.0 * jnp.linspace(-1.0, 1.0, nx)[:, None, None])
        + 0.2 * probe_y - 0.1 * probe_z, (nx, 5, 5))
    probe_mobility = jnp.full_like(probe_rhs, 0.02)

    def mixed_pressure(rhs0, mobility0):
        return _solvax_pressure_poisson_duct(rhs0, mobility0, dx=0.25,
            dy=jnp.asarray([0.15, 0.20, 0.30, 0.20, 0.15]),
            dz=jnp.asarray([0.12, 0.22, 0.32, 0.22, 0.12]), iterations=96,
            tolerance=1.0e-10, include_axial_line=False, single_reduction=True,
            axial_pressure_mode="inlet_neumann_outlet_dirichlet_zero",
            field_sharding=field_sharding)

    mixed_pressure = jax.jit(mixed_pressure,
        in_shardings=(field_sharding, field_sharding),
        out_shardings=(field_sharding,) + (replicated,) * 6)
    mixed = mixed_pressure(*(jax.device_put(np.asarray(value), field_sharding)
        for value in (probe_rhs, probe_mobility)))

    def momentum(velocity0, force0, density0, viscosity0, plus0, inlet0):
        inlet_patch = velocity0[0].at[..., 0].set(inlet0 / (density0[0] * area))
        zero_y, zero_z = jnp.zeros_like(velocity0[:, 0]), jnp.zeros_like(velocity0[:, :, 0])
        boundaries = (inlet_patch, velocity0[-1], zero_y, zero_y, zero_z, zero_z)
        corrected_force = force0 + _explicit_deviatoric_stress_duct(
            velocity0, density0 * viscosity0, boundaries,
            (jnp.full((nx,), dx), dy, dz))
        return _solvax_implicit_momentum_duct(
            velocity0, corrected_force, density0, viscosity0,
            _unpack_duct_mass_flux(plus0, inlet0), boundaries,
            dt=dt, dx=dx, dy=dy, dz=dz, iterations=iterations,
            tolerance=1.0e-10)

    momentum = jax.jit(momentum,
        in_shardings=(vector_sharding,) * 2 + (field_sharding,) * 2
        + (flux_sharding, replicated),
        out_shardings=(vector_sharding, replicated, replicated))
    solved, momentum_residual, momentum_converged = momentum(
        projected_velocity, force, density, viscosity, rho_phi_plus, rho_phi_inlet)
    def flux_diagnostics(plus, inlet):
        full_flux = _unpack_duct_mass_flux(plus, inlet)
        return (sum(jnp.linalg.norm(value) for value in full_flux),
            jnp.maximum(jnp.max(jnp.abs(full_flux[1][:, 0])),
                jnp.max(jnp.abs(full_flux[2][:, :, 0]))))
    flux_diagnostics = jax.jit(flux_diagnostics,
        in_shardings=(flux_sharding, replicated),
        out_shardings=(replicated, replicated))
    convection_flux_l2, lower_wall_flux = flux_diagnostics(
        rho_phi_plus, rho_phi_inlet)
    jax.block_until_ready((projected, mixed, solved, momentum_residual,
        convection_flux_l2))
    signature = np.concatenate([np.asarray(value).reshape(-1) for value in (
        initial_plus, initial_inlet, *projected[:5], rho_phi_plus, rho_phi_inlet,
        mixed[0], solved)])
    cut = np.asarray(rho_phi_plus[0, nx // 2 - 1])
    pressure_diagnostics = summarize_pressure_linear_history(
        np.asarray(projected[11:16], dtype=float)[None, :], expected_steps=1
    )
    return {"benchmark_kind": "duct_step_gate", "num_devices": num_devices,
        "signature": signature.tolist(), "divergence": float(projected[5]),
        "flow_error": float(projected[6]), "momentum_residual": float(momentum_residual),
        "momentum_converged": bool(momentum_converged),
        **pressure_diagnostics,
        "mixed_pressure_l2": float(jnp.linalg.norm(mixed[0])),
        "mixed_pressure_converged": bool(mixed[2]),
        "mixed_pressure_local_residual": float(mixed[-1]),
        "convection_flux_l2": float(convection_flux_l2),
        "lower_wall_flux": float(lower_wall_flux),
        "cut_boundary_separation": float(min(np.linalg.norm(cut - np.asarray(rho_phi_inlet)),
            np.linalg.norm(cut - np.asarray(rho_phi_plus[0, -1])))),
        "placement": {name: _placement(value) for name, value in (
            ("initial_flux", initial_plus), ("velocity", projected[0]),
            ("pressure", projected[3]), ("corrected_flux", rho_phi_plus),
            ("mixed_pressure", mixed[0]),
            ("inlet_flux", rho_phi_inlet), ("momentum", solved))}}


def _matched_b2_smoke_benchmark(
    input_path: Path, evaluator: Path, *, repeats: int, num_devices: int,
    profile_dir: Path | None = None,
) -> dict[str, object]:
    """Time the exact direct smoke; replay, I/O, and observation stay untimed."""

    if repeats < 4:
        raise ValueError("matched_b2_smoke requires one cold and three warm runs")
    from lmx.benchmarks import load_benchmark_b_spec
    from lmx.freemhd import load_matched_b2_lmx_input, observe_lmx_b2_output
    from lmx.io import (
        load_extruded_restart_bundle,
        validate_extruded_restart_bundle,
        write_extruded_bundle_restart_npz,
    )
    from scripts.run_freemhd_parity_suite import (
        _resume_matched_b2_lmx,
        _run_matched_b2_lmx_direct,
        _write_matched_b2_lmx_output,
    )

    problem = load_matched_b2_lmx_input(input_path)
    timings, signatures, checkpoint, direct = [], [], None, None
    for _ in range(repeats):
        started = time.perf_counter()
        checkpoint, direct = _run_matched_b2_lmx_direct(
            problem, num_devices=num_devices
        )
        jax.block_until_ready(_b2_ready_arrays(direct))
        timings.append(time.perf_counter() - started)
        signatures.append(_b2_repeat_signature(direct))
    profiled = profile_signature = None
    if profile_dir is not None:
        profile_dir.mkdir(parents=True, exist_ok=True)
        options = jax.profiler.ProfileOptions()
        options.python_tracer_level = 0
        options.raise_error_on_start_failure = True
        options.advanced_configuration["gpu_max_activity_api_events"] = 4_000_000
        jax.profiler.start_trace(
            str(profile_dir), create_perfetto_trace=True, profiler_options=options
        )
        try:
            _, profiled = _run_matched_b2_lmx_direct(problem, num_devices=num_devices)
            jax.block_until_ready(_b2_ready_arrays(profiled))
        finally:
            jax.profiler.stop_trace()
        profile_signature = _b2_repeat_signature(profiled)
    profile_signature_max_abs = (None if profile_signature is None else
        float(np.max(np.abs(profile_signature - signatures[0]))))
    profile_signature_passed = (None if profile_signature is None else bool(np.allclose(
        profile_signature, signatures[0], rtol=_B2_REPEAT_RTOL, atol=_B2_REPEAT_ATOL)))
    profile_histories = (() if profiled is None else tuple((
        np.asarray(getattr(profiled, name)), np.asarray(getattr(direct, name)), offset)
        for name, offset in (("iteration_pressure_linear_history", 2),
            ("iteration_electric_linear_history", 3))))
    profile_history_shapes_passed = all(left.shape == right.shape
        for left, right, _ in profile_histories)
    profile_linear_iteration_max_abs = (None if profiled is None else float(max(
        (np.max(np.abs(left[:, offset] - right[:, offset]))
            for left, right, offset in profile_histories), default=0.0))
        if profile_history_shapes_passed else float("inf"))
    profile_linear_history_passed = (None if profiled is None else bool(
        profile_history_shapes_passed
        and profile_linear_iteration_max_abs <= _B2_PROFILE_ITERATION_ATOL
        and all(np.array_equal(left[:, offset + 1:], right[:, offset + 1:])
            for left, right, offset in profile_histories)))
    acceptance_role = (
        "harness-smoke" if direct.u.shape == (8, 7, 7) else "scaling-calibration"
    )
    try:
        with tempfile.TemporaryDirectory(prefix="lmx-b2-serialized-restart-") as temporary:
            restart_path = Path(temporary) / "checkpoint.npz"
            write_extruded_bundle_restart_npz(checkpoint, problem.case, restart_path)
            serialized = load_extruded_restart_bundle(restart_path)
            validate_extruded_restart_bundle(serialized, case=problem.case)
            resumed = _resume_matched_b2_lmx(
                problem, serialized.bundle, num_devices=num_devices
            )
            anderson = _anderson_diagnostics(
                problem, checkpoint, direct, resumed, serialized,
                num_devices=num_devices,
            )
    except Exception as error:
        warm = np.asarray(timings[1:])
        return {
            "benchmark_kind": "matched_b2_smoke",
            "acceptance_role": acceptance_role,
            "operator_path": "solve_extruded_inductionless",
            "backend": jax.default_backend(), "device_kind": jax.devices()[0].device_kind,
            "num_devices": num_devices, "nx": direct.u.shape[0],
            "ny": direct.u.shape[1], "nz": direct.u.shape[2], "iterations": 2,
            "repeats": repeats, "cold_seconds": timings[0],
            "warm_samples_seconds": warm.tolist(),
            "warm_seconds": float(np.median(warm)), "validation_passed": False,
            "failure": {"phase": "restart", "type": type(error).__name__,
                "message": str(error)},
        }
    with tempfile.TemporaryDirectory(prefix="lmx-b2-scaling-") as temporary:
        evidence = Path(temporary) / "output"
        _write_matched_b2_lmx_output(
            input_path, evaluator, evidence, (checkpoint, direct, resumed),
            num_devices=num_devices, wall_seconds=timings[-1],
        )
        observed = observe_lmx_b2_output(evidence, input_path, evaluator)

    arrays = {name: getattr(direct, name) for name in _B2_FIELD_NAMES}
    placement = {name: _placement(value) for name, value in arrays.items()}
    for name, value in placement.items():
        expected_replicated = name == "rho_phi_inlet"
        if value["global_shards"] != num_devices or (
            num_devices > 1 and value["replicated"] != expected_replicated
        ):
            raise RuntimeError(f"Matched B2 field {name} has invalid placement {value}")
    pressure_history = np.asarray(direct.iteration_pressure_linear_history, dtype=float)
    pressure_diagnostics = summarize_pressure_linear_history(
        pressure_history, expected_steps=observed["steps"])
    limits = load_benchmark_b_spec("B2-fringing-square")["harness_smoke_execution"]
    repeat_signature_max_abs = max(float(np.max(np.abs(signature - signatures[0])))
        for signature in signatures[1:])
    repeat_signature_passed = all(np.allclose(
        signature, signatures[0], rtol=_B2_REPEAT_RTOL, atol=_B2_REPEAT_ATOL)
        for signature in signatures[1:])
    validation_passed = bool(
        observed["steps"] == 2 and observed["stop_reason"] == "step_limit"
        and max(observed["courant_max"]) <= limits["courant_max"]
        and all(observed[name] <= limits[f"{name}_max"] for name in (
            "mass_balance", "current_balance", "interface_current_balance"))
        and observed["interface_current_activity"] >= limits["interface_current_activity_min"]
        # Fast timing permits bounded face-flux reduction noise; all remaining
        # velocity, pressure, accelerator, and CFL state must replay tightly.
        and observed["restart_state_max_abs"] <= limits["restart_absolute_tolerance"]
        and observed["restart_flux_max_abs"] <= _B2_RESTART_FLUX_ATOL
        and observed["restart_flux_relative_l2"] <= _B2_RESTART_FLUX_RTOL
        and pressure_diagnostics["pressure_linear_diagnostics_complete"]
        and pressure_diagnostics["pressure_solves_converged"]
        and repeat_signature_passed
        and profile_signature_passed is not False
        and profile_linear_history_passed is not False
        and anderson["anderson_validation_passed"]
    )
    warm = np.asarray(timings[1:])
    velocity_l2 = float(np.sqrt(sum(np.linalg.norm(np.asarray(getattr(direct, name))) ** 2
        for name in ("u", "v", "w"))))
    current_l2 = float(np.sqrt(sum(np.linalg.norm(np.asarray(getattr(direct, name))) ** 2
        for name in ("jx", "jy", "jz"))))
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    device_memory = []
    for device in jax.devices()[:num_devices]:
        stats = device.memory_stats() or {}
        device_memory.append({"device": str(device), **{
            key: int(value) for key, value in stats.items()
            if "byte" in key.lower() and isinstance(value, (int, np.integer))}})
    return {
        "benchmark_kind": "matched_b2_smoke", "operator_path": "solve_extruded_inductionless",
        "acceptance_role": acceptance_role,
        "backend": jax.default_backend(), "device_kind": jax.devices()[0].device_kind,
        "num_devices": num_devices, "nx": direct.u.shape[0],
        "ny": direct.u.shape[1], "nz": direct.u.shape[2],
        "iterations": observed["steps"], "repeats": repeats,
        "timed_signature_excluded": True,
        "cold_seconds": timings[0],
        "warm_seconds": float(np.median(warm)), "mean_seconds": float(np.mean(timings)),
        "warm_samples_seconds": warm.tolist(), "warm_std_seconds": float(np.std(warm)),
        "warm_cv": float(np.std(warm) / max(np.mean(warm), 1.0e-30)),
        "total_cells": int(direct.u.size), "cell_updates": int(2 * direct.u.size),
        "warm_cell_updates_per_second": float(2 * direct.u.size / np.median(warm)),
        "velocity_l2": velocity_l2, "potential_l2": float(np.linalg.norm(np.asarray(direct.phi))),
        "current_l2": current_l2,
        "memory_bytes_estimate": _bundle_memory_bytes(direct),
        "peak_host_rss_bytes": int(peak_rss if sys.platform == "darwin" else 1024 * peak_rss),
        "device_memory": device_memory, "placement": placement,
        "spatially_sharded": num_devices > 1, "global_shard_count": num_devices,
        "validation_passed": validation_passed, "steady_state_passed": False,
        **pressure_diagnostics,
        "pressure_linear_history": pressure_history.tolist(),
        "signature_relative_tolerance": _B2_REPEAT_RTOL,
        "repeat_signature_max_abs": repeat_signature_max_abs,
        "repeat_signature_passed": repeat_signature_passed,
        "profile_signature_max_abs": profile_signature_max_abs,
        "profile_signature_passed": profile_signature_passed,
        "profile_linear_history_passed": profile_linear_history_passed,
        "profile_linear_iteration_max_abs": profile_linear_iteration_max_abs,
        "profile_linear_iteration_absolute_tolerance": _B2_PROFILE_ITERATION_ATOL,
        "electric_linear_history": np.asarray(
            direct.iteration_electric_linear_history).tolist(),
        "profile_pressure_linear_history": (None if profiled is None else
            np.asarray(profiled.iteration_pressure_linear_history).tolist()),
        "profile_electric_linear_history": (None if profiled is None else
            np.asarray(profiled.iteration_electric_linear_history).tolist()),
        "observables": observed,
        **anderson,
        "restart_flux_absolute_tolerance": _B2_RESTART_FLUX_ATOL,
        "restart_flux_relative_tolerance": _B2_RESTART_FLUX_RTOL,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run a single strong-scaling benchmark worker."
    )
    parser.add_argument(
        "--benchmark-kind",
        choices=("extruded3d", "extruded_solve", "duct_step_gate", "matched_b2_smoke"),
        default="extruded3d",
    )
    parser.add_argument("--nx", type=int, default=384)
    parser.add_argument("--ny", type=int, default=1024)
    parser.add_argument("--nz", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--num-devices", type=int, required=True)
    parser.add_argument("--platform", type=str, default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile-dir", type=Path, default=None)
    parser.add_argument("--matched-input", type=Path, default=None)
    parser.add_argument("--evaluator", type=Path, default=None)
    parser.add_argument(
        "--restart",
        type=Path,
        default=None,
        help="Verified extruded restart used to initialize solver-faithful timing.",
    )
    args = parser.parse_args(argv)

    if args.benchmark_kind == "matched_b2_smoke":
        if args.matched_input is None or args.evaluator is None:
            parser.error("matched_b2_smoke requires --matched-input and --evaluator")
        try:
            payload = _matched_b2_smoke_benchmark(
                args.matched_input, args.evaluator, repeats=args.repeats,
                num_devices=args.num_devices, profile_dir=args.profile_dir)
        except Exception as error:
            payload = {
                "benchmark_kind": "matched_b2_smoke", "num_devices": args.num_devices,
                "validation_passed": False,
                "failure": {"phase": "worker", "type": type(error).__name__,
                    "message": str(error)},
            }
    elif args.benchmark_kind == "duct_step_gate":
        payload = _duct_step_gate(nx=args.nx, ny=args.ny, nz=args.nz,
            iterations=args.iterations, num_devices=args.num_devices)
    elif args.benchmark_kind == "extruded_solve":
        record = benchmark_extruded_inductionless_solve(
            nx=args.nx,
            ny=args.ny,
            nz=args.nz,
            max_steps=args.iterations,
            repeats=args.repeats,
            num_devices=args.num_devices,
            profile_dir=args.profile_dir,
            restart_path=args.restart,
        )
    elif args.benchmark_kind == "extruded3d":
        record = benchmark_sharded_extruded_operator(
            nx=args.nx,
            ny=args.ny,
            nz=args.nz,
            iterations=args.iterations,
            repeats=args.repeats,
            num_devices=args.num_devices,
        )
    if args.benchmark_kind not in {"duct_step_gate", "matched_b2_smoke"}:
        payload = {**record.__dict__}
    payload.update(platform=args.platform, source_fingerprint=_source_fingerprint(),
        python_version=platform.python_version(), jax_version=jax.__version__,
        jaxlib_version=version("jaxlib"), solvax_version=version("solvax"))
    if args.matched_input is not None and args.evaluator is not None:
        payload.update(input_sha256=hashlib.sha256(args.matched_input.read_bytes()).hexdigest(),
            evaluator_sha256=hashlib.sha256(args.evaluator.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    printed = payload if args.benchmark_kind != "duct_step_gate" else {
        key: value for key, value in payload.items() if key != "signature"}
    print(json.dumps(printed, indent=2))
    if args.benchmark_kind == "matched_b2_smoke" and not payload.get(
        "validation_passed", False
    ):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
