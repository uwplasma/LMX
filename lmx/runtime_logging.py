from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import time
from typing import TextIO

import jax.numpy as jnp

from .config import LoggingSpec
from .core import Solution
from .physics import MaterialFields
from .specs import CaseSpec


@dataclass(frozen=True)
class RestartLogInfo:
    enabled: bool = False
    path: str | None = None
    start_time: float = 0.0
    reset_histories: bool = True


@dataclass(frozen=True)
class SolverStepRecord:
    step_index: int
    time: float
    dt: float
    u_max: float
    mean_velocity: float
    current_max: float
    face_current_max: float
    emf_max: float
    lorentz_max: float
    face_lorentz_max: float
    residual: float
    potential_residual: float
    potential_iterations: float
    applied_forcing: float
    pressure_proxy: float
    current_scaled_pressure_proxy: float
    courant_like: float
    ohmic_power: float


class StreamingSolverLogger:
    def __init__(self, config: LoggingSpec | None = None, *, stream: TextIO | None = None) -> None:
        self.config = config or LoggingSpec()
        self.streams: list[TextIO] = [stream or sys.stdout]
        self._start_time = time.perf_counter()

    def add_stream(self, stream: TextIO) -> None:
        self.streams.append(stream)

    def _write(self, line: str = "") -> None:
        for stream in self.streams:
            print(line, file=stream, flush=self.config.flush)

    def emit_header(
        self,
        *,
        case: CaseSpec,
        mesh,
        materials: MaterialFields,
        mode: str,
        potential_solver: str,
        target_mean_velocity: float | None,
        reference_mean_velocity: float | None,
        restart: RestartLogInfo | None = None,
    ) -> None:
        if not self.config.enabled:
            return
        width = 92
        if self.config.banner:
            self._write("=" * width)
            self._write(f"{'LMX Solver Run':^{width}}")
            self._write("=" * width)
        self._write(f"Create time")
        self._write(f"Create mesh for case        : {case.name}")
        self._write(f"Solve mode                  : {mode}")
        self._write(f"Geometry                    : {case.geometry.kind}")
        self._write(f"Cells (nx, ny, nz)          : ({mesh.nx}, {mesh.ny}, {mesh.nz})")
        self._write(f"Domain (L, W, H)            : ({case.geometry.length:.6e}, {case.geometry.width:.6e}, {case.geometry.height:.6e})")
        self._write(f"Time controls               : dt={case.time_stepper.dt:.6e}, endTime={case.time_stepper.t_final:.6e}, maxSteps={case.time_stepper.max_steps}")
        self._write(f"Velocity controls           : outerIterations={case.time_stepper.outer_iterations}, relaxation={case.time_stepper.relaxation:.6e}, limiter={case.time_stepper.velocity_update_limiter}, updateLimit={case.time_stepper.velocity_update_limit:.6e}")
        self._write(f"Potential controls          : solver={potential_solver}, iterations={case.time_stepper.potential_iterations}, tolerance={case.time_stepper.potential_tolerance}, omega={case.time_stepper.potential_relaxation:.6e}")
        self._write(f"Current reconstruction      : {case.time_stepper.current_reconstruction}")
        self._write(f"Magnetic field              : kind={case.magnetic_field.kind}, value={case.magnetic_field.value}, rampStart={case.magnetic_field.ramp_start:.6e}, rampDuration={case.magnetic_field.ramp_duration:.6e}")
        self._write(f"Flow forcing                : explicit={case.forcing:.6e}, initialVelocity={case.initial_velocity:.6e}, targetMeanVelocity={target_mean_velocity}, referenceMeanVelocity={reference_mean_velocity}")
        if restart is not None and restart.enabled:
            self._write(
                f"Restart controls            : source={restart.path}, startTime={restart.start_time:.6e}, "
                f"resetHistories={restart.reset_histories}"
            )
        if self.config.print_regions:
            self._write("Read region properties")
            for region in case.regions:
                self._write(
                    "  "
                    f"{region.name:<18s} kind={region.kind:<5s} sigma={region.conductivity:.6e} "
                    f"rho={float(region.density or 1.0):.6e} nu={float(region.viscosity or 1.0):.6e}"
                )
            fluid_fraction = float(jnp.mean(materials.fluid_mask.astype(float)))
            self._write(
                "  "
                f"material ranges             sigma=[{float(jnp.min(materials.conductivity)):.6e}, {float(jnp.max(materials.conductivity)):.6e}] "
                f"rho=[{float(jnp.min(materials.density)):.6e}, {float(jnp.max(materials.density)):.6e}] "
                f"nu=[{float(jnp.min(materials.viscosity)):.6e}, {float(jnp.max(materials.viscosity)):.6e}] "
                f"fluidFraction={fluid_fraction:.6f}"
            )
        if self.config.print_boundaries:
            self._write("Read boundary conditions")
            for boundary in case.boundary_conditions:
                self._write(
                    "  "
                    f"{boundary.name:<18s} kind={boundary.kind:<22s} axis={boundary.axis or '-':<2s} "
                    f"side={boundary.side or '-':<11s} region={boundary.region or '-':<18s} value={boundary.value}"
                )
        self._write("-" * width)

    def emit_step(self, record: SolverStepRecord) -> None:
        if not self.config.enabled:
            return
        if record.step_index > 1 and (record.step_index - 1) % max(self.config.step_stride, 1) != 0:
            return
        elapsed = time.perf_counter() - self._start_time
        self._write(f"Time = {record.time:.6e}")
        self._write(
            "smoothSolver: potE             "
            f"Final residual = {record.potential_residual:.6e}, No Iterations {int(record.potential_iterations)}"
        )
        self._write(
            "MHD predictor                  "
            f"max|U| = {record.u_max:.6e}, mean(U) = {record.mean_velocity:.6e}, "
            f"CourantLike = {record.courant_like:.6e}"
        )
        self._write(
            "MHD electromagnetics           "
            f"max|J| = {record.current_max:.6e}, max|J_face| = {record.face_current_max:.6e}, "
            f"max|UxB| = {record.emf_max:.6e}, max|JxB| = {record.lorentz_max:.6e}, "
            f"max|JxB|_face = {record.face_lorentz_max:.6e}"
        )
        self._write(
            "MHD forcing                    "
            f"appliedForcing = {record.applied_forcing:.6e}, pressureProxy = {record.pressure_proxy:.6e}, "
            f"currentScaledPressureProxy = {record.current_scaled_pressure_proxy:.6e}, "
            f"OhmicPower = {record.ohmic_power:.6e}"
        )
        self._write(
            "steadySolver                   "
            f"velocity residual = {record.residual:.6e}"
        )
        self._write(f"ExecutionTime = {elapsed:.3f} s")
        self._write("")

    def emit_footer(self, solution: Solution) -> None:
        if not self.config.enabled or not self.config.print_footer:
            return
        elapsed = time.perf_counter() - self._start_time
        self._write("-" * 92)
        self._write(f"End")
        self._write(f"Final time                    : {solution.state.time:.6e}")
        self._write(f"Final residual                : {solution.state.residual:.6e}")
        self._write(f"Output case                   : {solution.case_name}")
        self._write(f"ExecutionTime                 : {elapsed:.3f} s")


def default_log_path(out_dir: str | Path, case_name: str) -> Path:
    out_dir = Path(out_dir)
    return out_dir / f"{case_name}.log"
