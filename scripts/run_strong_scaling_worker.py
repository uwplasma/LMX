from __future__ import annotations

# ruff: noqa: E402 -- repository-root bootstrap must precede project imports.

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import resource
import shlex
import shutil
import subprocess
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
    _SUSTAINED_WARM_CV_MAX,
    _SUSTAINED_MIN_CELLS,
    _SUSTAINED_MIN_CELL_UPDATES,
    _SUSTAINED_WARM_SAMPLES,
    _SUSTAINED_WARM_SECONDS,
    _bundle_memory_bytes,
    benchmark_extruded_inductionless_solve,
    benchmark_sharded_extruded_operator,
    summarize_pressure_linear_history,
    summarize_strong_scaling_records,
    write_strong_scaling_summary_table,
)

_B2_RESTART_FLUX_ATOL = 1.0e-6
_B2_RESTART_FLUX_RTOL = 1.0e-5
_B2_REPEAT_ATOL = 2.0e-9
_B2_REPEAT_RTOL = 2.0e-8
_B2_PROFILE_ITERATION_ATOL = 3
_B2_TIMING_CONTRACT = {
    "cold_sample_count": 1,
    "compile_in_cold_sample": True,
    "warm_samples_exclude_compilation": True,
    "synchronization": "jax.block_until_ready",
    "timed_observers_excluded": True,
    "optional_progress_callbacks_excluded": True,
    "diagnostic_phase_timing_excluded": True,
    "restart_audit_excluded": True,
}
_B2_FIELD_NAMES = (
    "u", "v", "w", "p", "phi", "jx", "jy", "jz", "rho_phi_plus", "rho_phi_inlet"
)

if ROOT not in Path(lmx.__file__).resolve().parents:
    raise RuntimeError(
        f"Scaling worker imported LMX outside its source tree: {lmx.__file__}"
    )


def _sustained_timing_passed(
    minimum_warm_seconds: float, warm_samples: object
) -> bool:
    """Require a predeclared multi-minute workload for scaling evidence."""

    samples = np.asarray(warm_samples, dtype=float)
    mean = float(np.mean(samples)) if samples.size else 0.0
    return bool(
        samples.ndim == 1
        and samples.size >= _SUSTAINED_WARM_SAMPLES
        and minimum_warm_seconds >= _SUSTAINED_WARM_SECONDS
        and np.all(np.isfinite(samples))
        and np.all(samples >= _SUSTAINED_WARM_SECONDS)
        and mean > 0.0
        and np.std(samples) / mean <= _SUSTAINED_WARM_CV_MAX
    )


def _source_fingerprint_paths() -> tuple[Path, ...]:
    """Return package source and data files that define scaling numerics."""

    data = ROOT / "lmx" / "data" / "benchmarks"
    paths = [*sorted((ROOT / "lmx").glob("*.py")), Path(__file__).resolve(),
        ROOT / "scripts" / "run_freemhd_parity_suite.py"]
    paths.extend(sorted(path for path in (data / "specs").rglob("*") if path.is_file()))
    paths.extend(sorted(path for path in (data / "references").rglob("*") if path.is_file()))
    return tuple(paths)


def _source_fingerprint() -> str:
    """Hash the source and packaged specifications used by this worker."""

    digest = hashlib.sha256()
    for path in _source_fingerprint_paths():
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
        "anderson_replay_field_relative_l2": None,
        "anderson_replay_field_tolerance_ratio": None,
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
    replay_field_relative = max(
        float(np.linalg.norm(np.asarray(left) - np.asarray(right)) /
            max(np.linalg.norm(np.asarray(left)), np.linalg.norm(np.asarray(right)), 1.0e-30))
        for left, right in zip(direct_state[:2], resumed_state[:2], strict=True)
    )
    replay_field_tolerance_ratio = max(
        float(np.max(np.abs(np.asarray(left) - np.asarray(right)) /
            (_B2_REPEAT_ATOL + _B2_REPEAT_RTOL * np.maximum(
                np.abs(np.asarray(left)), np.abs(np.asarray(right))))))
        for left, right in zip(direct_state[:2], resumed_state[:2], strict=True)
    )
    replay_flux_relative = max(
        float(np.linalg.norm(np.asarray(left) - np.asarray(right)) /
            max(np.linalg.norm(np.asarray(left)), np.linalg.norm(np.asarray(right)), 1.0e-30))
        for left, right in zip(direct_state[2:], resumed_state[2:], strict=True)
    )
    from solvax import anderson_weights

    # This endpoint signature checks restart/topology parity. The solver still
    # applies the schema-6 depth-two Anderson update at every trajectory step.
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
    requested_steps = int(direct.stopping_state[0])
    checkpoint_step = (requested_steps + 1) // 2
    depth_two = (
        depth == 2
        and requested_steps >= 2
        and checkpoint.stopping_state[0] == checkpoint_step
        and direct.stopping_state == resumed.stopping_state
        and direct.stopping_state[2] == "step_limit"
    )
    weights_sum_error = float(abs(np.sum(weights_np) - 1.0))
    valid = bool(
        schema == "b2_diagnostics_v6"
        and depth_two
        and serialized_max <= 1.0e-12
        and replay_field_tolerance_ratio <= 1.0
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
        anderson_replay_field_relative_l2=replay_field_relative,
        anderson_replay_field_tolerance_ratio=replay_field_tolerance_ratio,
        anderson_replay_flux_max_abs=replay_flux_max,
        anderson_replay_flux_relative_l2=replay_flux_relative,
        anderson_gram=gram_np.tolist(),
        anderson_weights=weights_np.tolist(),
        anderson_endpoint_steps=[checkpoint_step, requested_steps],
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


def _b2_linear_history_comparison(candidate, control) -> tuple[bool, float]:
    """Compare synchronized/profiled pressure and electric Krylov histories."""

    histories = tuple((
        np.asarray(getattr(candidate, name)), np.asarray(getattr(control, name)), offset
    ) for name, offset in (
        ("iteration_pressure_linear_history", 2),
        ("iteration_electric_linear_history", 3),
    ))
    shapes_match = all(left.shape == right.shape for left, right, _ in histories)
    if not shapes_match:
        return False, float("inf")
    iteration_max_abs = float(max((
        np.max(np.abs(left[:, offset] - right[:, offset]))
        for left, right, offset in histories
    ), default=0.0))
    passed = bool(
        iteration_max_abs <= _B2_PROFILE_ITERATION_ATOL
        and all(np.array_equal(left[:, offset + 1:], right[:, offset + 1:])
            for left, right, offset in histories)
    )
    return passed, iteration_max_abs


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
    iterations: int | None = None,
    minimum_warm_seconds: float = 0.0,
    phase_timing: bool = False,
) -> dict[str, object]:
    """Time a fixed-work B2 trajectory; replay, I/O, and observation stay untimed."""

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
    executed_steps = int(problem.case.time_stepper.max_steps)
    if iterations is not None and iterations != executed_steps:
        raise ValueError(
            f"matched input executes {executed_steps} steps, not --iterations {iterations}"
        )
    if minimum_warm_seconds < 0.0:
        raise ValueError("minimum_warm_seconds must be nonnegative")
    timings, signatures, timed_direct = [], [], None
    for _ in range(repeats):
        started = time.perf_counter()
        _, timed_direct = _run_matched_b2_lmx_direct(
            problem, num_devices=num_devices, capture_checkpoint=False
        )
        jax.block_until_ready(_b2_ready_arrays(timed_direct))
        timings.append(time.perf_counter() - started)
        signatures.append(_b2_repeat_signature(timed_direct))
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
            _, profiled = _run_matched_b2_lmx_direct(
                problem, num_devices=num_devices, capture_checkpoint=False
            )
            jax.block_until_ready(_b2_ready_arrays(profiled))
        finally:
            jax.profiler.stop_trace()
        profile_signature = _b2_repeat_signature(profiled)
    profile_signature_max_abs = (None if profile_signature is None else
        float(np.max(np.abs(profile_signature - signatures[0]))))
    profile_signature_passed = (None if profile_signature is None else bool(np.allclose(
        profile_signature, signatures[0], rtol=_B2_REPEAT_RTOL, atol=_B2_REPEAT_ATOL)))
    if profiled is None:
        profile_linear_history_passed = profile_linear_iteration_max_abs = None
    else:
        (
            profile_linear_history_passed,
            profile_linear_iteration_max_abs,
        ) = _b2_linear_history_comparison(profiled, timed_direct)
    phase_timing_payload = None
    phase_signature_passed = phase_linear_history_passed = None
    if phase_timing:
        phase_samples: list[dict[str, object]] = []

        def record_phase(name: str, wall_seconds: float) -> None:
            occurrence = 1 + sum(item["phase"] == name for item in phase_samples)
            phase_samples.append({
                "phase": name,
                "occurrence": occurrence,
                "wall_seconds": wall_seconds,
            })

        phase_started = time.perf_counter()
        _, phase_direct = _run_matched_b2_lmx_direct(
            problem,
            num_devices=num_devices,
            capture_checkpoint=False,
            phase_timing_callback=record_phase,
        )
        jax.block_until_ready(_b2_ready_arrays(phase_direct))
        phase_total = time.perf_counter() - phase_started
        phase_signature = _b2_repeat_signature(phase_direct)
        phase_signature_max_abs = float(
            np.max(np.abs(phase_signature - signatures[0]))
        )
        phase_signature_passed = bool(np.allclose(
            phase_signature, signatures[0],
            rtol=_B2_REPEAT_RTOL, atol=_B2_REPEAT_ATOL,
        ))
        (
            phase_linear_history_passed,
            phase_linear_iteration_max_abs,
        ) = _b2_linear_history_comparison(phase_direct, timed_direct)
        phase_names = tuple(dict.fromkeys(
            str(item["phase"]) for item in phase_samples
        ))
        phase_seconds = {
            name: float(sum(
                float(item["wall_seconds"])
                for item in phase_samples if item["phase"] == name
            ))
            for name in phase_names
        }
        phase_sum = sum(phase_seconds.values())
        phase_timing_payload = {
            "schema_version": 1,
            "scope": "separate-synchronized-diagnostic",
            "excluded_from_warm_timing": True,
            "synchronization": "jax.block_until_ready",
            "synchronized_total_seconds": phase_total,
            "named_phase_sum_seconds": phase_sum,
            "unattributed_seconds": max(0.0, phase_total - phase_sum),
            "phase_seconds": phase_seconds,
            "phase_fraction_of_synchronized_total": {
                name: seconds / max(phase_total, 1.0e-30)
                for name, seconds in phase_seconds.items()
            },
            "samples": phase_samples,
            "pressure_iterations": np.asarray(
                phase_direct.iteration_pressure_linear_history
            )[:, 2].tolist(),
            "electric_iterations": np.asarray(
                phase_direct.iteration_electric_linear_history
            )[:, 3].tolist(),
            "signature_max_abs": phase_signature_max_abs,
            "signature_passed": phase_signature_passed,
            "linear_history_passed": phase_linear_history_passed,
            "linear_iteration_max_abs": phase_linear_iteration_max_abs,
        }
    timed_peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    timed_peak_rss_bytes = int(
        timed_peak_rss if sys.platform == "darwin" else 1024 * timed_peak_rss
    )
    timed_device_memory = []
    for device in jax.devices()[:num_devices]:
        stats = device.memory_stats() or {}
        timed_device_memory.append({"device": str(device), **{
            key: int(value) for key, value in stats.items()
            if "byte" in key.lower() and isinstance(value, (int, np.integer))}})

    # Restart validation is intentionally outside timing and memory measurement.
    checkpoint, direct = _run_matched_b2_lmx_direct(
        problem, num_devices=num_devices
    )
    captured_signature_passed = bool(np.allclose(
        _b2_repeat_signature(direct), signatures[0],
        rtol=_B2_REPEAT_RTOL, atol=_B2_REPEAT_ATOL,
    ))
    captured_histories = tuple((
        np.asarray(getattr(direct, name)), np.asarray(getattr(timed_direct, name)))
        for name in (
            "iteration_pressure_linear_history",
            "iteration_electric_linear_history",
        ))
    captured_linear_history_passed = bool(all(
        left.shape == right.shape and np.array_equal(left, right)
        for left, right in captured_histories
    ))
    if direct.u.shape == (8, 7, 7) and executed_steps == 2:
        acceptance_role = "harness-smoke"
    elif minimum_warm_seconds >= _SUSTAINED_WARM_SECONDS:
        acceptance_role = "sustained-candidate"
    else:
        acceptance_role = "fixed-work-debug"
    warm = np.asarray(timings[1:])
    sustained = _sustained_timing_passed(minimum_warm_seconds, warm)
    total_cells = int(direct.u.size)
    cell_updates = executed_steps * total_cells
    large_work = bool(total_cells >= _SUSTAINED_MIN_CELLS
        and cell_updates >= _SUSTAINED_MIN_CELL_UPDATES)
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
        return {
            "benchmark_kind": "matched_b2_smoke",
            "acceptance_role": acceptance_role,
            "operator_path": "solve_extruded_inductionless",
            "backend": jax.default_backend(), "device_kind": jax.devices()[0].device_kind,
            "num_devices": num_devices, "nx": direct.u.shape[0],
            "ny": direct.u.shape[1], "nz": direct.u.shape[2],
            "iterations": executed_steps,
            "total_cells": total_cells, "cell_updates": cell_updates,
            "repeats": repeats, "cold_seconds": timings[0],
            "timing_contract": _B2_TIMING_CONTRACT,
            "warm_samples_seconds": warm.tolist(),
            "warm_seconds": float(np.median(warm)), "validation_passed": False,
            "captured_signature_passed": captured_signature_passed,
            "captured_linear_history_passed": captured_linear_history_passed,
            "peak_host_rss_bytes": timed_peak_rss_bytes,
            "device_memory": timed_device_memory,
            "sustained_minimum_warm_seconds": _SUSTAINED_WARM_SECONDS,
            "sustained_duration_passed": sustained,
            "large_fixed_work_passed": large_work,
            "measurement_class": ("sustained-multiminute"
                if sustained and large_work else "debug-or-calibration"),
            "sustained_timing_eligible": False,
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
        observed["steps"] == executed_steps and observed["stop_reason"] == "step_limit"
        and max(observed["courant_max"]) <= limits["courant_max"]
        and all(observed[name] <= limits[f"{name}_max"] for name in (
            "mass_balance", "current_balance", "interface_current_balance"))
        and observed["interface_current_activity"] >= limits["interface_current_activity_min"]
        # Fast timing permits bounded face-flux reduction noise; all remaining
        # velocity, pressure, accelerator, and CFL state must replay tightly.
        and observed["restart_state_tolerance_ratio"] <= 1.0
        and observed["restart_flux_max_abs"] <= _B2_RESTART_FLUX_ATOL
        and observed["restart_flux_relative_l2"] <= _B2_RESTART_FLUX_RTOL
        and pressure_diagnostics["pressure_linear_diagnostics_complete"]
        and pressure_diagnostics["pressure_solves_converged"]
        and repeat_signature_passed
        and profile_signature_passed is not False
        and profile_linear_history_passed is not False
        and phase_signature_passed is not False
        and phase_linear_history_passed is not False
        and captured_signature_passed
        and captured_linear_history_passed
        and anderson["anderson_validation_passed"]
        and all(sample >= minimum_warm_seconds for sample in timings[1:])
    )
    velocity_l2 = float(np.sqrt(sum(np.linalg.norm(np.asarray(getattr(direct, name))) ** 2
        for name in ("u", "v", "w"))))
    current_l2 = float(np.sqrt(sum(np.linalg.norm(np.asarray(getattr(direct, name))) ** 2
        for name in ("jx", "jy", "jz"))))
    return {
        "benchmark_kind": "matched_b2_smoke", "operator_path": "solve_extruded_inductionless",
        "acceptance_role": acceptance_role,
        "backend": jax.default_backend(), "device_kind": jax.devices()[0].device_kind,
        "num_devices": num_devices, "nx": direct.u.shape[0],
        "ny": direct.u.shape[1], "nz": direct.u.shape[2],
        "iterations": observed["steps"], "repeats": repeats,
        "timed_signature_excluded": True,
        "timing_contract": _B2_TIMING_CONTRACT,
        "cold_seconds": timings[0],
        "captured_signature_passed": captured_signature_passed,
        "captured_linear_history_passed": captured_linear_history_passed,
        "warm_seconds": float(np.median(warm)), "mean_seconds": float(np.mean(timings)),
        "warm_samples_seconds": warm.tolist(), "warm_std_seconds": float(np.std(warm)),
        "warm_cv": float(np.std(warm) / max(np.mean(warm), 1.0e-30)),
        "total_cells": total_cells, "cell_updates": cell_updates,
        "warm_cell_updates_per_second": float(
            executed_steps * direct.u.size / np.median(warm)
        ),
        "minimum_warm_seconds": minimum_warm_seconds,
        "requested_duration_passed": bool(
            all(sample >= minimum_warm_seconds for sample in warm)),
        "sustained_minimum_warm_seconds": _SUSTAINED_WARM_SECONDS,
        "sustained_duration_passed": sustained,
        "large_fixed_work_passed": large_work,
        "measurement_class": ("sustained-multiminute"
            if sustained and large_work else "debug-or-calibration"),
        "sustained_timing_eligible": bool(validation_passed and sustained and large_work),
        "velocity_l2": velocity_l2, "potential_l2": float(np.linalg.norm(np.asarray(direct.phi))),
        "current_l2": current_l2,
        "memory_bytes_estimate": _bundle_memory_bytes(direct),
        "peak_host_rss_bytes": timed_peak_rss_bytes,
        "device_memory": timed_device_memory, "placement": placement,
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
        "phase_timing": phase_timing_payload,
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
        "restart_state_absolute_tolerance": _B2_REPEAT_ATOL,
        "restart_state_relative_tolerance": _B2_REPEAT_RTOL,
    }


def _worker_main(argv: list[str] | None = None) -> int:
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
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--num-devices", type=int, required=True)
    parser.add_argument("--platform", type=str, default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit", type=str, default=None)
    parser.add_argument("--profile-dir", type=Path, default=None)
    parser.add_argument(
        "--phase-timing",
        action="store_true",
        help="run one extra synchronized B2 phase diagnostic outside timing",
    )
    parser.add_argument("--matched-input", type=Path, default=None)
    parser.add_argument("--evaluator", type=Path, default=None)
    parser.add_argument(
        "--minimum-warm-seconds",
        type=float,
        default=0.0,
        help="Fail validation if any warm trajectory is shorter than this duration.",
    )
    parser.add_argument(
        "--restart",
        type=Path,
        default=None,
        help="Verified extruded restart used to initialize solver-faithful timing.",
    )
    args = parser.parse_args(argv)
    if args.phase_timing and args.benchmark_kind != "matched_b2_smoke":
        parser.error("--phase-timing requires --benchmark-kind matched_b2_smoke")

    if args.benchmark_kind == "matched_b2_smoke":
        if args.matched_input is None or args.evaluator is None:
            parser.error("matched_b2_smoke requires --matched-input and --evaluator")
        try:
            payload = _matched_b2_smoke_benchmark(
                args.matched_input, args.evaluator, repeats=args.repeats,
                num_devices=args.num_devices, profile_dir=args.profile_dir,
                iterations=args.iterations,
                minimum_warm_seconds=args.minimum_warm_seconds,
                phase_timing=args.phase_timing,
            )
        except Exception as error:
            payload = {
                "benchmark_kind": "matched_b2_smoke", "num_devices": args.num_devices,
                "validation_passed": False,
                "failure": {"phase": "worker", "type": type(error).__name__,
                    "message": str(error)},
            }
    elif args.benchmark_kind == "duct_step_gate":
        iterations = 120 if args.iterations is None else args.iterations
        payload = _duct_step_gate(nx=args.nx, ny=args.ny, nz=args.nz,
            iterations=iterations, num_devices=args.num_devices)
    elif args.benchmark_kind == "extruded_solve":
        iterations = 120 if args.iterations is None else args.iterations
        record = benchmark_extruded_inductionless_solve(
            nx=args.nx,
            ny=args.ny,
            nz=args.nz,
            max_steps=iterations,
            repeats=args.repeats,
            num_devices=args.num_devices,
            profile_dir=args.profile_dir,
            restart_path=args.restart,
        )
    elif args.benchmark_kind == "extruded3d":
        iterations = 120 if args.iterations is None else args.iterations
        record = benchmark_sharded_extruded_operator(
            nx=args.nx,
            ny=args.ny,
            nz=args.nz,
            iterations=iterations,
            repeats=args.repeats,
            num_devices=args.num_devices,
        )
    if args.benchmark_kind not in {"duct_step_gate", "matched_b2_smoke"}:
        payload = {**record.__dict__}
    payload.update(platform=args.platform, source_commit=args.source_commit,
        source_fingerprint=_source_fingerprint(),
        precision="float64" if jax.config.jax_enable_x64 else "float32",
        cpu_affinity=(sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity") else None),
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


# Campaign orchestration stays beside the fingerprinted worker.  The public
# example intentionally contains only a small local calibration.
_MONITOR_SAMPLE_SECONDS = 1.0
_MONITOR_POSTFLIGHT_SECONDS = 15.0
_ADMISSION_SAMPLE_SECONDS = 60.0


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    """Publish one evidence record without exposing a partial JSON file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        json.dump(payload, stream, indent=2)
        candidate = Path(stream.name)
    os.replace(candidate, path)


def _monitor_summary(
    *, backend: str, num_devices: int, raw_path: Path,
    violations: set[str], timed_out: bool, sample_times: list[float],
    worker_elapsed: float, monitor_start: float, worker_start: float,
    worker_end: float, monitor_end: float,
) -> dict[str, object]:
    """Reduce one raw monitor stream to the compact promotion contract."""

    max_gap = max((b - a for a, b in zip(sample_times, sample_times[1:])), default=0.0)
    if max_gap > 2.0:
        violations.add("sampling_gap_above_2_seconds")
    return {
        "schema_version": 2, "scope": "continuous-and-postflight",
        "backend": backend, "num_devices": num_devices,
        "verified": not violations and not timed_out,
        "raw_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest(),
        "sample_period_seconds": _MONITOR_SAMPLE_SECONDS,
        "max_sample_gap_seconds": max_gap,
        "monitored_worker_seconds": worker_elapsed,
        "postflight_seconds": monitor_end - worker_end,
        "violation_count": len(violations),
        "monitor_started_unix_seconds": monitor_start,
        "worker_started_unix_seconds": worker_start,
        "worker_ended_unix_seconds": worker_end,
        "monitor_ended_unix_seconds": monitor_end,
    }


def _environment_record(path: Path, rung: dict[str, object]) -> dict[str, object]:
    """Bind a freshly collected admission record to the launched worker."""

    raw = path.read_bytes()
    return {
        "resource_environment_verified": rung.get("verified") is True,
        "resource_environment_sha256": hashlib.sha256(raw).hexdigest(),
        "resource_environment": rung,
    }


def _cpu_ticks(cpus: tuple[int, ...]) -> dict[int, tuple[int, int]]:
    """Return total and idle Linux scheduler ticks for selected CPUs."""

    rows = {
        int(parts[0][3:]): tuple(map(int, parts[1:]))
        for line in Path("/proc/stat").read_text().splitlines()
        if (parts := line.split())
        and parts[0].startswith("cpu")
        and parts[0][3:].isdigit()
    }
    return {cpu: (sum(rows[cpu]), rows[cpu][3] + rows[cpu][4]) for cpu in cpus}


def _collect_cpu_admission(
    path: Path, *, num_devices: int, source_commit: str
) -> dict[str, object]:
    """Observe an idle Linux cpuset immediately before a sustained rung."""

    missing = [name for name in ("ps", "taskset") if shutil.which(name) is None]
    if missing or not hasattr(os, "sched_getaffinity"):
        detail = ", ".join(missing) or "Linux affinity control"
        raise RuntimeError(f"Sustained CPU scaling requires {detail}")
    available = tuple(sorted(os.sched_getaffinity(0)))
    affinity = available[: 2 * num_devices]
    if len(affinity) != 2 * num_devices:
        raise RuntimeError(f"CPU rung {num_devices} requires {2 * num_devices} CPUs")

    started, before = time.monotonic(), _cpu_ticks(affinity)
    time.sleep(_ADMISSION_SAMPLE_SECONDS)
    after = _cpu_ticks(affinity)
    utilization = []
    for cpu in affinity:
        total = after[cpu][0] - before[cpu][0]
        idle = after[cpu][1] - before[cpu][1]
        utilization.append(100.0 * (1.0 - idle / max(total, 1)))
    maximum = max(utilization)
    rung = {
        "num_devices": num_devices,
        "affinity_cpus": list(affinity),
        "allocated_cpu_count": len(affinity),
        "max_cpu_utilization_percent": maximum,
        "admission_ended_unix_seconds": time.time(),
        "verified": maximum <= 5.0,
    }
    _atomic_json(
        path,
        {
            "backend": "cpu",
            "host": os.uname().nodename,
            "source_commit": source_commit,
            "sample_seconds": time.monotonic() - started,
            "rungs": {str(num_devices): rung},
        },
    )
    if not rung["verified"]:
        raise RuntimeError("CPU admission observed more than 5% utilization")
    return rung


def _forced_cpu_environment(count: int) -> dict[str, str]:
    """Select a bounded CPU device mesh without discarding safe XLA flags."""

    environment = os.environ.copy()
    excluded = (
        "--xla_force_host_platform_device_count=",
        "--xla_cpu_multi_thread_eigen=",
        "intra_op_parallelism_threads=",
    )
    flags = [
        flag for flag in shlex.split(environment.get("XLA_FLAGS", ""))
        if not flag.startswith(excluded)
    ]
    flags += [f"--xla_force_host_platform_device_count={count}",
        "--xla_cpu_multi_thread_eigen=false", "intra_op_parallelism_threads=1"]
    environment.update(
        JAX_PLATFORMS="cpu",
        JAX_ENABLE_X64="true",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        XLA_FLAGS=" ".join(flags),
        OMP_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        NUMEXPR_NUM_THREADS="1",
    )
    return environment


def _run_monitored_cpu_worker(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    raw_path: Path,
    num_devices: int,
    expected_affinity: tuple[int, ...],
    timeout_seconds: float,
) -> tuple[int, dict[str, object], bool]:
    """Run a CPU worker while checking affinity, swap, and foreign work."""

    started = time.monotonic()
    monitor_start = time.time()
    process = subprocess.Popen(command, cwd=cwd, env=env)
    worker_start = time.time()
    violations: set[str] = set()
    sample_times: list[float] = []
    baseline_swapouts: int | None = None
    timed_out = False
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    def sample(stream, phase: str) -> None:
        nonlocal baseline_swapouts
        now, current = time.monotonic(), []
        evidence: dict[str, object] = {
            "unix_seconds": time.time(),
            "phase": phase,
            "worker_pid": process.pid,
        }
        try:
            output = subprocess.run(
                ["ps", "-Ao", "pid=,ppid=,%cpu=,comm="],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            rows = [line.strip().split(maxsplit=3) for line in output.splitlines()]
            processes = [
                (int(row[0]), int(row[1]), float(row[2]), row[3])
                for row in rows
                if len(row) == 4
            ]
            if not processes:
                raise RuntimeError("empty process probe")
            owned = {process.pid, os.getpid()}
            while children := {
                pid
                for pid, parent, _, _ in processes
                if parent in owned and pid not in owned
            }:
                owned.update(children)
            foreign = [
                (pid, cpu, name)
                for pid, _, cpu, name in processes
                if pid not in owned and cpu > 25.0
            ]
            evidence["foreign_processes"] = foreign
            if foreign:
                current.append("foreign_process_above_25_percent_cpu")
            if sys.platform == "darwin":
                vm = subprocess.run(
                    ["vm_stat"], check=True, capture_output=True, text=True
                ).stdout
                row = next(line for line in vm.splitlines() if line.startswith("Swapouts:"))
                swapouts = int(row.split(":", 1)[1].strip().rstrip("."))
            else:
                row = next(
                    line
                    for line in Path("/proc/vmstat").read_text().splitlines()
                    if line.startswith("pswpout ")
                )
                swapouts = int(row.split()[1])
            baseline_swapouts = (
                swapouts if baseline_swapouts is None else baseline_swapouts
            )
            evidence["swapout_increase"] = swapouts - baseline_swapouts
            if swapouts > baseline_swapouts:
                current.append("swapout_increase")
            affinity = None
            if process.poll() is None and hasattr(os, "sched_getaffinity"):
                affinity = sorted(os.sched_getaffinity(process.pid))
                if tuple(affinity) != expected_affinity:
                    current.append("worker_affinity_escape")
            evidence["worker_affinity"] = affinity
        except Exception as error:  # Fail closed on unavailable system evidence.
            evidence["probe_error"] = f"{type(error).__name__}: {error}"
            current.append("probe_error")
        violations.update(current)
        evidence["violations"] = current
        stream.write(json.dumps(evidence, sort_keys=True) + "\n")
        stream.flush()
        sample_times.append(now)

    with raw_path.open("w") as stream:
        next_sample = started
        while process.poll() is None:
            now = time.monotonic()
            if now - started >= timeout_seconds:
                timed_out = True
                process.kill()
                process.wait()
                break
            if now < next_sample:
                time.sleep(min(next_sample - now, 0.1))
                continue
            sample(stream, "worker")
            next_sample += _MONITOR_SAMPLE_SECONDS
        worker_end = time.time()
        worker_elapsed = time.monotonic() - started
        postflight_started = time.monotonic()
        while time.monotonic() - postflight_started < _MONITOR_POSTFLIGHT_SECONDS:
            sample(stream, "postflight")
            time.sleep(_MONITOR_SAMPLE_SECONDS)
    monitor_end = time.time()
    summary = _monitor_summary(
        backend="cpu", num_devices=num_devices, raw_path=raw_path,
        violations=violations, timed_out=timed_out, sample_times=sample_times,
        worker_elapsed=worker_elapsed, monitor_start=monitor_start,
        worker_start=worker_start, worker_end=worker_end, monitor_end=monitor_end)
    return int(process.returncode), summary, timed_out


def _gpu_snapshot(
    visible_devices: tuple[str, ...],
) -> tuple[list[dict[str, str]], float, int, int]:
    """Read selected GPU identity, utilization, and foreign process counts."""

    probe = (
        "nvidia-smi --query-gpu=index,uuid,pci.bus_id,utilization.gpu "
        "--format=csv,noheader,nounits; echo __CONTEXTS__; "
        "nvidia-smi --query-compute-apps=gpu_uuid,pid "
        "--format=csv,noheader,nounits; echo __PROCESSES__; "
        "ps -eo pid=,pcpu=,comm="
    )
    output = subprocess.run(
        ["sh", "-c", probe],
        check=True,
        capture_output=True,
        text=True,
        timeout=10.0,
    ).stdout
    gpu_text, context_text = output.split("__CONTEXTS__\n", 1)
    context_text, process_text = context_text.split("__PROCESSES__\n", 1)
    rows = [
        [part.strip() for part in line.split(",")]
        for line in gpu_text.splitlines()
        if line.strip()
    ]
    selected = [row for row in rows if row[0] in visible_devices]
    if len(selected) != len(visible_devices):
        raise RuntimeError("GPU admission could not identify every selected device")
    identities = [
        {"uuid": row[1], "pci_bus_id": row[2]} for row in selected
    ]
    maximum = max(float(row[3]) for row in selected)
    selected_uuids = {row[1] for row in selected}
    contexts = sum(
        line.split(",", 1)[0].strip() in selected_uuids
        for line in context_text.splitlines()
        if line.strip()
    )
    foreign_cpu = sum(
        float(parts[1]) > 25.0
        for line in process_text.splitlines()
        if len(parts := line.split(maxsplit=2)) == 3
    )
    return identities, maximum, contexts, foreign_cpu


def _collect_gpu_admission(
    path: Path,
    *,
    visible_devices: str,
    num_devices: int,
    source_commit: str,
) -> dict[str, object]:
    """Observe an idle, identity-stable GPU allocation before one rung."""

    devices = tuple(visible_devices.split(","))
    started = time.monotonic()
    deadline = started + _ADMISSION_SAMPLE_SECONDS
    snapshots = []
    while True:
        snapshots.append(_gpu_snapshot(devices))
        if time.monotonic() >= deadline:
            break
        time.sleep(min(1.0, deadline - time.monotonic()))
    identities = snapshots[0][0]
    if any(snapshot[0] != identities for snapshot in snapshots):
        raise RuntimeError("GPU identity changed during admission")
    maximum = max(snapshot[1] for snapshot in snapshots)
    contexts = max(snapshot[2] for snapshot in snapshots)
    foreign_cpu = max(snapshot[3] for snapshot in snapshots)
    rung = {
        "num_devices": num_devices,
        "visible_devices": list(devices),
        "gpu_identities": identities,
        "foreign_compute_process_count": contexts,
        "foreign_cpu_process_count": foreign_cpu,
        "max_gpu_utilization_percent": maximum,
        "admission_ended_unix_seconds": time.time(),
        "verified": contexts == foreign_cpu == 0 and maximum <= 5.0,
    }
    _atomic_json(
        path,
        {
            "backend": "gpu",
            "host": os.uname().nodename,
            "source_commit": source_commit,
            "sample_seconds": time.monotonic() - started,
            "rungs": {str(num_devices): rung},
        },
    )
    if not rung["verified"]:
        raise RuntimeError("GPU admission observed utilization or foreign work")
    return rung


def _run_monitored_gpu_worker(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    raw_path: Path,
    num_devices: int,
    environment: dict[str, object],
    timeout_seconds: float,
) -> tuple[int, dict[str, object], bool]:
    """Run a remote worker while rejecting shared-host contamination."""

    def normalize_pci(value: object) -> str:
        return ":".join(str(value).lower().split(":")[-2:])

    expected = {
        str(index): (identity["uuid"], normalize_pci(identity["pci_bus_id"]))
        for index, identity in zip(
            environment["visible_devices"],
            environment["gpu_identities"],
            strict=True,
        )
    }
    selected_uuids = {identity[0] for identity in expected.values()}
    monitor_start, started = time.time(), time.monotonic()
    process = subprocess.Popen(command, cwd=cwd, env=env)
    worker_start = time.time()
    violations: set[str] = set()
    sample_times: list[float] = []
    timed_out = False
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    def sample(stream, phase: str) -> None:
        now, current = time.monotonic(), []
        evidence: dict[str, object] = {"unix_seconds": time.time(), "phase": phase}
        probe = (
            "nvidia-smi --query-gpu=index,uuid,pci.bus_id,utilization.gpu,memory.used "
            "--format=csv,noheader,nounits; echo __CONTEXTS__; "
            "nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory "
            "--format=csv,noheader,nounits; echo __PROCESSES__; "
            "ps -eo pid=,ppid=,pcpu=,comm="
        )
        try:
            output = subprocess.run(
                ["sh", "-c", probe],
                check=True,
                capture_output=True,
                text=True,
                timeout=1.0,
            ).stdout
            gpu_text, context_text = output.split("__CONTEXTS__\n", 1)
            context_text, process_text = context_text.split("__PROCESSES__\n", 1)
            lines = gpu_text.splitlines()
            gpus = [
                [part.strip() for part in line.split(",")]
                for line in lines
                if line.strip()
            ]
            observed = {
                row[0]: (row[1], normalize_pci(row[2])) for row in gpus
            }
            if any(observed.get(index) != identity for index, identity in expected.items()):
                current.append("gpu_identity_remap")
            process_rows = [
                (int(row[0]), int(row[1]), float(row[2]), row[3])
                for line in process_text.splitlines()
                if len(row := line.strip().split(maxsplit=3)) == 4
            ]
            if not process_rows:
                raise RuntimeError("missing remote worker or process inventory")
            owned = {process.pid, os.getpid()}
            while children := {
                pid
                for pid, parent, _, _ in process_rows
                if parent in owned and pid not in owned
            }:
                owned.update(children)
            contexts = [
                [part.strip() for part in line.split(",", 3)]
                for line in context_text.splitlines()
                if line.strip()
            ]
            selected = [row for row in contexts if row[0] in selected_uuids]
            foreign_gpu = [row for row in selected if int(row[1]) not in owned]
            foreign_cpu = [
                (pid, cpu, name)
                for pid, _, cpu, name in process_rows
                if pid not in owned and cpu > 25.0
            ]
            if foreign_gpu or (phase == "postflight" and selected):
                current.append("foreign_or_postflight_gpu_context")
            if foreign_cpu:
                current.append("foreign_process_above_25_percent_cpu")
            if phase == "postflight" and any(
                float(row[3]) > 5.0 for row in gpus if row[0] in expected
            ):
                current.append("postflight_gpu_utilization_above_5_percent")
            evidence.update(
                worker_pid=process.pid,
                gpus=gpus,
                selected_contexts=selected,
                foreign_gpu_contexts=foreign_gpu,
                foreign_processes=foreign_cpu,
            )
        except Exception as error:  # Fail closed on unavailable host evidence.
            evidence["probe_error"] = f"{type(error).__name__}: {error}"
            current.append("probe_error")
        violations.update(current)
        evidence["violations"] = current
        stream.write(json.dumps(evidence, sort_keys=True) + "\n")
        stream.flush()
        sample_times.append(now)

    with raw_path.open("w") as stream:
        next_sample = started
        while process.poll() is None:
            now = time.monotonic()
            if now - started >= timeout_seconds:
                timed_out = True
                process.terminate()
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                break
            if now < next_sample:
                time.sleep(min(next_sample - now, 0.1))
                continue
            sample(stream, "worker")
            next_sample += _MONITOR_SAMPLE_SECONDS
        worker_end = time.time()
        worker_elapsed = time.monotonic() - started
        postflight_started = time.monotonic()
        while time.monotonic() - postflight_started < _MONITOR_POSTFLIGHT_SECONDS:
            sample(stream, "postflight")
            time.sleep(_MONITOR_SAMPLE_SECONDS)
    monitor_end = time.time()
    summary = _monitor_summary(
        backend="gpu", num_devices=num_devices, raw_path=raw_path,
        violations=violations, timed_out=timed_out, sample_times=sample_times,
        worker_elapsed=worker_elapsed, monitor_start=monitor_start,
        worker_start=worker_start, worker_end=worker_end, monitor_end=monitor_end)
    return int(process.returncode), summary, timed_out


def _matched_worker_command(
    *,
    python_executable: str,
    num_devices: int,
    iterations: int,
    repeats: int,
    output: str,
    matched_input: str,
    evaluator: str,
    source_commit: str,
    minimum_warm_seconds: float,
    platform_name: str,
) -> list[str]:
    """Build one isolated matched-B2 worker command."""

    return [
        python_executable, str(Path(__file__).resolve()),
        "--benchmark-kind", "matched_b2_smoke",
        "--platform", platform_name,
        "--num-devices", str(num_devices),
        "--iterations", str(iterations),
        "--repeats", str(repeats),
        "--minimum-warm-seconds", str(minimum_warm_seconds),
        "--matched-input", matched_input,
        "--evaluator", evaluator,
        "--source-commit", source_commit,
        "--output", output,
    ]


def _finalize_record(
    path: Path,
    environment: dict[str, object],
    monitoring: dict[str, object] | None,
) -> dict[str, object]:
    """Attach admission and monitoring evidence to a worker record."""

    record = json.loads(path.read_text())
    if monitoring is not None:
        monitoring["source_fingerprint"] = record.get("source_fingerprint")
        record["resource_monitoring"] = monitoring
    record.update(environment)
    path.write_text(json.dumps(record, indent=2) + "\n")
    return record


def _run_campaign(
    *,
    backend: str,
    repo_root: Path,
    out_dir: Path,
    counts: tuple[int, ...],
    matched_input: Path,
    evaluator: Path,
    iterations: int,
    repeats: int,
    python_executable: str,
    source_commit: str,
    sustained: bool,
    timeout_seconds: float,
    admission_base: Path,
) -> list[dict[str, object]]:
    """Run one fixed-work CPU or GPU ladder serially on the current host."""

    if backend not in {"cpu", "gpu"}:
        raise ValueError(f"Unsupported campaign backend: {backend}")
    available_gpus = []
    if backend == "gpu":
        available_gpus = [
            line.strip()
            for line in subprocess.run(
                ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
                check=True, capture_output=True, text=True,
            ).stdout.splitlines()
            if line.strip()
        ]
    records = []
    for count in counts:
        environment: dict[str, object] = {"resource_environment_verified": False}
        affinity: tuple[int, ...] = ()
        monitor_path = None
        visible = ""
        if backend == "gpu":
            if count > len(available_gpus):
                raise ValueError(
                    f"Requested {count} GPUs; host exposes {len(available_gpus)}"
                )
            visible = ",".join(available_gpus[-count:])
        if sustained:
            admission = admission_base.with_name(
                f"{admission_base.stem}-{count}{admission_base.suffix}"
            )
            rung = (
                _collect_cpu_admission(
                    admission, num_devices=count, source_commit=source_commit
                )
                if backend == "cpu"
                else _collect_gpu_admission(
                    admission, visible_devices=visible,
                    num_devices=count, source_commit=source_commit,
                )
            )
            environment = _environment_record(admission, rung)
            if backend == "cpu":
                affinity = tuple(rung["affinity_cpus"])
            monitor_path = out_dir / f"{backend}_{count}.monitor.jsonl"

        output = out_dir / f"{backend}_{count}.json"
        command = _matched_worker_command(
            python_executable=python_executable,
            num_devices=count,
            iterations=iterations,
            repeats=repeats,
            output=str(output),
            matched_input=str(matched_input),
            evaluator=str(evaluator),
            source_commit=source_commit,
            minimum_warm_seconds=120.0 if sustained else 0.0,
            platform_name=backend.upper(),
        )
        if affinity:
            command = ["taskset", "--cpu-list", ",".join(map(str, affinity)), *command]
        worker_environment = (
            _forced_cpu_environment(count)
            if backend == "cpu"
            else os.environ | {
                "CUDA_VISIBLE_DEVICES": visible,
                "JAX_PLATFORMS": "cuda",
                "JAX_ENABLE_X64": "true",
                "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            }
        )
        worker_environment["PYTHONPATH"] = str(repo_root)
        monitoring = None
        if monitor_path is None:
            subprocess.run(
                command,
                check=True,
                cwd=repo_root,
                env=worker_environment,
                timeout=timeout_seconds,
            )
        else:
            monitor = (
                _run_monitored_cpu_worker if backend == "cpu"
                else _run_monitored_gpu_worker
            )
            options = dict(
                cwd=repo_root, env=worker_environment, raw_path=monitor_path,
                num_devices=count, timeout_seconds=timeout_seconds,
            )
            if backend == "cpu":
                options["expected_affinity"] = affinity
            else:
                options["environment"] = environment["resource_environment"]
            returncode, monitoring, timed_out = monitor(command, **options)
            if timed_out:
                raise subprocess.TimeoutExpired(command, timeout_seconds)
            if returncode:
                raise subprocess.CalledProcessError(returncode, command)
        record = _finalize_record(output, environment, monitoring)
        records.append(record)
        if not record.get("validation_passed", False):
            raise RuntimeError(f"Matched B2 {backend.upper()} gate failed: {output}")
    rows = summarize_strong_scaling_records(records)["rows"]
    if not all(row["physics_equivalent"] for row in rows):
        raise RuntimeError(f"Matched B2 {backend.upper()} topology changed the solution")
    return records


def _campaign_main(argv: list[str]) -> int:
    """Run the debug or monitored matched-B2 CPU/GPU campaign."""

    parser = argparse.ArgumentParser(description="Run the matched-B2 scaling campaign.")
    parser.add_argument("--output", type=Path, default=Path("artifacts/strong_scaling"))
    parser.add_argument("--sustained", action="store_true")
    parser.add_argument("--backend", choices=("cpu", "gpu", "both"), default="cpu")
    parser.add_argument("--cpu-counts", default="1,2,4")
    parser.add_argument("--gpu-counts", default="1,2")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[1]
    args.output.mkdir(parents=True, exist_ok=True)
    source_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    shape = (256, 67, 67) if args.sustained else (8, 7, 7)
    iterations = {"cpu": 32 if args.sustained else 2,
        "gpu": 96 if args.sustained else 2}
    timeout = 1800.0 if args.sustained else 180.0

    from scripts.run_freemhd_parity_suite import (
        materialize_matched_b2_evaluator,
        materialize_matched_b2_lmx_input,
    )

    evaluator = args.output / "matched_b2_evaluator.json"
    materialize_matched_b2_evaluator(evaluator)
    records: list[dict[str, object]] = []
    selected = ("cpu", "gpu") if args.backend == "both" else (args.backend,)
    for backend in selected:
        matched_input = args.output / f"matched_b2_{backend}_input.json"
        materialize_matched_b2_lmx_input(
            matched_input, solver_shape=shape, executed_steps=iterations[backend]
        )
        records.extend(
            _run_campaign(
                backend=backend, repo_root=repo_root, out_dir=args.output,
                counts=tuple(map(int, getattr(args, f"{backend}_counts").split(","))),
                matched_input=matched_input,
                evaluator=evaluator,
                iterations=iterations[backend], repeats=4,
                python_executable=args.python,
                source_commit=source_commit,
                sustained=args.sustained,
                timeout_seconds=timeout,
                admission_base=args.output / f"{backend}-admission.json",
            )
        )

    diagnostics = summarize_strong_scaling_records(records)
    table = write_strong_scaling_summary_table(
        records, args.output / "strong_scaling_table.csv"
    )
    summary = {
        "records": records,
        "table": table.name,
        "diagnostics": diagnostics,
        "source_commit": source_commit,
    }
    (args.output / "strong_scaling_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Dispatch a single worker or the explicit ``--campaign`` launcher."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["--campaign"]:
        return _campaign_main(arguments[1:])
    return _worker_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
