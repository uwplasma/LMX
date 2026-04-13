from io import StringIO
from pathlib import Path

import jax.numpy as jnp
import pytest

from lmx.cases import make_hartmann_case
from lmx.config import LoggingSpec
from lmx.core import Diagnostics, MHDState, Solution
from lmx.physics import build_material_fields
from lmx.runtime_logging import RestartLogInfo, SolverStepRecord, StreamingSolverLogger, default_log_path
from lmx.solvers import _build_mesh


pytestmark = pytest.mark.unit


def test_streaming_solver_logger_prints_live_solver_sections():
    stream = StringIO()
    logger = StreamingSolverLogger(LoggingSpec(step_stride=1), stream=stream)
    case = make_hartmann_case(ha=5.0, ny=8, nz=8)
    mesh = _build_mesh(case)
    materials = build_material_fields(case, mesh)
    shape = mesh.yz_shape
    logger.emit_header(
        case=case,
        mesh=mesh,
        materials=materials,
        mode="steady",
        potential_solver=case.time_stepper.potential_solver,
        target_mean_velocity=None,
        reference_mean_velocity=None,
        restart=RestartLogInfo(enabled=False),
    )
    logger.emit_step(
        SolverStepRecord(
            step_index=1,
            time=0.1,
            dt=0.1,
            u_max=1.0,
            mean_velocity=0.5,
            current_max=0.2,
            face_current_max=0.18,
            emf_max=0.12,
            lorentz_max=0.08,
            face_lorentz_max=0.07,
            residual=1e-6,
            potential_residual=1e-5,
            potential_iterations=12.0,
            linear_residual=1e-6,
            linear_iterations=8.0,
            applied_forcing=1.0,
            pressure_proxy=0.4,
            current_scaled_pressure_proxy=0.3,
            raw_update_max=0.02,
            limiter_scale=1.0,
            limited_fraction=0.0,
            courant_like=0.05,
            ohmic_power=0.01,
            volumetric_flow_rate=0.9,
            mean_current_magnitude=0.11,
            lorentz_power=0.02,
            div_current_max=1e-8,
            charge_balance_residual=2e-9,
            gauge_residual=1e-10,
            interface_current_residual=1e-8,
            potential_initial_residual=2e-5,
            linear_initial_residual=3e-6,
        )
    )
    logger.emit_footer(
            Solution(
                mesh=mesh,
                state=MHDState(
                u=jnp.ones(shape),
                phi=jnp.zeros(shape),
                jy=jnp.zeros(shape),
                jz=jnp.zeros(shape),
                lorentz_x=jnp.zeros(shape),
                time=0.1,
                residual=1e-6,
            ),
            diagnostics=Diagnostics(
                residual_history=jnp.asarray([1e-3, 1e-6]),
                courant_like=jnp.asarray([0.05]),
                ohmic_power=jnp.asarray([0.01]),
                time_history=jnp.asarray([0.1]),
                u_max_history=jnp.asarray([1.0]),
                mean_velocity_history=jnp.asarray([0.5]),
                applied_forcing_history=jnp.asarray([1.0]),
                pressure_proxy_history=jnp.asarray([0.4]),
                current_scaled_pressure_proxy_history=jnp.asarray([0.3]),
                raw_update_max_history=jnp.asarray([0.02]),
                limiter_scale_history=jnp.asarray([1.0]),
                limited_fraction_history=jnp.asarray([0.0]),
                current_max_history=jnp.asarray([0.2]),
                face_current_max_history=jnp.asarray([0.18]),
                emf_max_history=jnp.asarray([0.12]),
                lorentz_max_history=jnp.asarray([0.08]),
                face_lorentz_max_history=jnp.asarray([0.07]),
                potential_residual_history=jnp.asarray([1e-5]),
                potential_iterations_history=jnp.asarray([12.0]),
                linear_residual_history=jnp.asarray([1e-6]),
                linear_iterations_history=jnp.asarray([8.0]),
                volumetric_flow_rate_history=jnp.asarray([0.9]),
                mean_current_magnitude_history=jnp.asarray([0.11]),
                lorentz_power_history=jnp.asarray([0.02]),
                div_current_max_history=jnp.asarray([1e-8]),
                charge_balance_residual_history=jnp.asarray([2e-9]),
                gauge_residual_history=jnp.asarray([1e-10]),
                interface_current_residual_history=jnp.asarray([1e-8]),
            ),
            case_name=case.name,
        )
    )

    text = stream.getvalue()
    assert "LMX Solver Run" in text
    assert "Create mesh for case" in text
    assert "Time =" in text
    assert "smoothSolver: potE" in text
    assert "smoothSolver: U" in text
    assert "Initial residual" in text
    assert "currentScaledPressureProxy" in text
    assert "chargeBalanceResidual" in text
    assert "MHD integrals" in text
    assert "MHD conservation" in text
    assert "rawUpdateMax" in text
    assert "limitedFraction" in text
    assert "steadySolver" in text
    assert "ExecutionTime" in text


def _sample_record(step_index: int = 1) -> SolverStepRecord:
    return SolverStepRecord(
        step_index=step_index,
        time=0.1 * step_index,
        dt=0.1,
        u_max=1.0,
        mean_velocity=0.5,
        current_max=0.2,
        face_current_max=0.18,
        emf_max=0.12,
        lorentz_max=0.08,
        face_lorentz_max=0.07,
        residual=1e-6,
        potential_residual=1e-5,
        potential_iterations=12.0,
        linear_residual=1e-6,
        linear_iterations=8.0,
        applied_forcing=1.0,
        pressure_proxy=0.4,
        current_scaled_pressure_proxy=0.3,
        raw_update_max=0.02,
        limiter_scale=1.0,
        limited_fraction=0.0,
        courant_like=0.05,
        ohmic_power=0.01,
        volumetric_flow_rate=0.9,
        mean_current_magnitude=0.11,
        lorentz_power=0.02,
        div_current_max=1e-8,
        charge_balance_residual=2e-9,
        gauge_residual=1e-10,
        interface_current_residual=1e-8,
        potential_initial_residual=2e-5,
        linear_initial_residual=3e-6,
    )


def test_streaming_solver_logger_respects_disable_stride_and_legacy_sections():
    disabled_stream = StringIO()
    disabled_logger = StreamingSolverLogger(LoggingSpec(enabled=False), stream=disabled_stream)
    disabled_logger.emit_step(_sample_record())
    assert disabled_stream.getvalue() == ""

    step_stream = StringIO()
    extra_stream = StringIO()
    logger = StreamingSolverLogger(LoggingSpec(step_stride=2, print_footer=False), stream=step_stream)
    logger.add_stream(extra_stream)
    case = make_hartmann_case(ha=5.0, ny=8, nz=8)
    case = case.__class__(
        **{
            **case.__dict__,
            "solver": case.solver.__class__(**{**case.solver.__dict__, "kind": "legacy_reduced"}),
        }
    )
    mesh = _build_mesh(case)
    materials = build_material_fields(case, mesh)
    logger.emit_header(
        case=case,
        mesh=mesh,
        materials=materials,
        mode="steady",
        potential_solver=case.time_stepper.potential_solver,
        target_mean_velocity=None,
        reference_mean_velocity=None,
        restart=RestartLogInfo(enabled=True, path="restart.npz", start_time=0.2, reset_histories=False),
    )
    logger.emit_step(_sample_record(step_index=2))
    logger.emit_footer(
        Solution(
            mesh=mesh,
            state=MHDState(
                u=jnp.ones(mesh.yz_shape),
                phi=jnp.zeros(mesh.yz_shape),
                jy=jnp.zeros(mesh.yz_shape),
                jz=jnp.zeros(mesh.yz_shape),
                    lorentz_x=jnp.zeros(mesh.yz_shape),
                    time=0.2,
                    residual=1e-6,
                ),
                diagnostics=Diagnostics(
                    residual_history=jnp.asarray([1e-6]),
                    courant_like=jnp.asarray([0.0]),
                    ohmic_power=jnp.asarray([0.0]),
                ),
                case_name=case.name,
            )
        )

    text = step_stream.getvalue()
    assert "Legacy velocity controls" in text
    assert "Current reconstruction" in text
    assert "Restart controls" in text
    assert "Time =" not in text
    assert "Final time" not in text
    assert extra_stream.getvalue() == text


def test_default_log_path_appends_case_log_name(tmp_path: Path):
    assert default_log_path(tmp_path, "demo_case") == tmp_path / "demo_case.log"


def test_streaming_solver_logger_normal_verbosity_suppresses_detailed_sections():
    stream = StringIO()
    logger = StreamingSolverLogger(LoggingSpec(verbosity="normal"), stream=stream)
    case = make_hartmann_case(ha=5.0, ny=8, nz=8)
    mesh = _build_mesh(case)
    materials = build_material_fields(case, mesh)

    logger.emit_header(
        case=case,
        mesh=mesh,
        materials=materials,
        mode="steady",
        potential_solver=case.time_stepper.potential_solver,
        target_mean_velocity=None,
        reference_mean_velocity=None,
        restart=RestartLogInfo(enabled=False),
    )
    logger.emit_step(_sample_record())

    text = stream.getvalue()
    assert "LMX Solver Run" in text
    assert "Read region properties" not in text
    assert "Read boundary conditions" not in text
    assert "MHD integrals" not in text
    assert "MHD conservation" not in text
    assert "MHD limiter" not in text
    assert "steadySolver" in text


def test_streaming_solver_logger_debug_verbosity_prints_debug_line():
    stream = StringIO()
    logger = StreamingSolverLogger(LoggingSpec(verbosity="debug"), stream=stream)
    logger.emit_step(_sample_record())
    text = stream.getvalue()
    assert "MHD debug" in text
    assert "faceCurrentRatio" in text
    assert "faceLorentzRatio" in text
