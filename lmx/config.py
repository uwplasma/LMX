from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - only used on Python < 3.11
    import tomli as tomllib

import jax.numpy as jnp

from .physics import MaterialFields
from .specs import (
    BoundaryCondition,
    CaseSpec,
    FringingSpec,
    GeometrySpec,
    MagneticFieldSpec,
    OutputSpec,
    RegionSpec,
    Solution,
    SolverConfig,
    TimeStepperConfig,
)


@dataclass(frozen=True)
class LoggingSpec:
    enabled: bool = True
    verbosity: str = "detailed"
    banner: bool = True
    print_regions: bool = True
    print_boundaries: bool = True
    print_footer: bool = True
    flush: bool = True
    step_stride: int = 1

    @classmethod
    def from_user_controls(
        cls,
        *,
        enabled: bool = True,
        verbose: bool | None = None,
        verbosity: str | None = None,
        banner: bool = True,
        print_regions: bool = True,
        print_boundaries: bool = True,
        print_footer: bool = True,
        flush: bool = True,
        step_stride: int = 1,
    ) -> "LoggingSpec":
        if verbose is not None:
            enabled = enabled and bool(verbose)
            if verbosity is None:
                verbosity = "detailed" if verbose else "quiet"
        normalized = str(verbosity or "detailed").lower()
        if normalized not in {"quiet", "normal", "detailed", "debug"}:
            raise ValueError(f"Unsupported logging verbosity {normalized!r}")
        if normalized == "quiet":
            enabled = False
        return cls(
            enabled=enabled,
            verbosity=normalized,
            banner=banner,
            print_regions=print_regions,
            print_boundaries=print_boundaries,
            print_footer=print_footer,
            flush=flush,
            step_stride=step_stride,
        )

    def verbosity_rank(self) -> int:
        levels = {"quiet": 0, "normal": 1, "detailed": 2, "debug": 3}
        return levels.get(self.verbosity, 2)

    def is_enabled(self) -> bool:
        return bool(self.enabled) and self.verbosity_rank() > 0


@dataclass(frozen=True)
class RestartSpec:
    enabled: bool = False
    path: Path | None = None
    reset_histories: bool = True
    write_restart: bool = False
    restart_filename: str | None = None


@dataclass(frozen=True)
class RunConfig:
    case: CaseSpec
    logging: LoggingSpec = field(default_factory=LoggingSpec)
    restart: RestartSpec = field(default_factory=RestartSpec)
    fringing: FringingSpec = field(default_factory=FringingSpec)
    input_path: Path | None = None


def _load_toml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("rb") as handle:
        return tomllib.load(handle)


def _require(mapping: dict[str, Any], key: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required TOML key '{key}'")
    return mapping[key]


def _optional_tuple(
    mapping: dict[str, Any], key: str, *, length: int | None = None, cast=float
) -> tuple[Any, ...] | None:
    if key not in mapping:
        return None
    values = tuple(cast(value) for value in mapping[key])
    if length is not None and len(values) != length:
        raise ValueError(f"TOML key '{key}' must have length {length}")
    return values


def _parse_regions(entries: list[dict[str, Any]]) -> tuple[RegionSpec, ...]:
    return tuple(
        RegionSpec(
            name=str(_require(entry, "name")),
            kind=str(_require(entry, "kind")),
            conductivity=float(_require(entry, "conductivity")),
            density=None if entry.get("density") is None else float(entry["density"]),
            viscosity=None if entry.get("viscosity") is None else float(entry["viscosity"]),
            wall_thickness=None if entry.get("wall_thickness") is None else float(entry["wall_thickness"]),
        )
        for entry in entries
    )


def _parse_boundary_value(value: Any) -> float | tuple[float, float, float] | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, list):
        values = tuple(float(component) for component in value)
        if len(values) != 3:
            raise ValueError("Boundary-condition vector values must have length 3")
        return values
    raise ValueError(f"Unsupported boundary-condition value {value!r}")


def _parse_boundaries(entries: list[dict[str, Any]]) -> tuple[BoundaryCondition, ...]:
    return tuple(
        BoundaryCondition(
            name=str(_require(entry, "name")),
            kind=str(_require(entry, "kind")),
            value=_parse_boundary_value(entry.get("value")),
            region=None if entry.get("region") is None else str(entry["region"]),
            axis=None if entry.get("axis") is None else str(entry["axis"]),
            side=None if entry.get("side") is None else str(entry["side"]),
        )
        for entry in entries
    )


def load_run_config(path: str | Path) -> RunConfig:
    input_path = Path(path).resolve()
    root = _load_toml(input_path)

    case_table = root.get("case", {})
    geometry_table = root.get("geometry", {})
    field_table = root.get("magnetic_field", {})
    solver_table = root.get("solver", {})
    time_table = root.get("time_stepper", {})
    output_table = root.get("output", {})
    logging_table = root.get("logging", {})
    restart_table = root.get("restart", {})
    fringing_table = root.get("fringing", {})
    regions_table = root.get("regions", [])
    boundaries_table = root.get("boundary_conditions", [])

    if field_table.get("kind") == "analytic":
        raise ValueError(
            "TOML input does not support analytic magnetic-field callables; use the Python API for that case"
        )

    geometry = GeometrySpec(
        kind=str(_require(geometry_table, "kind")),
        width=float(_require(geometry_table, "width")),
        height=float(_require(geometry_table, "height")),
        length=float(geometry_table.get("length", 1.0)),
        nx=int(geometry_table.get("nx", 1)),
        ny=int(_require(geometry_table, "ny")),
        nz=int(_require(geometry_table, "nz")),
        radius=None if geometry_table.get("radius") is None else float(geometry_table["radius"]),
        nr=None if geometry_table.get("nr") is None else int(geometry_table["nr"]),
        ntheta=None if geometry_table.get("ntheta") is None else int(geometry_table["ntheta"]),
        wall_thickness=_optional_tuple(geometry_table, "wall_thickness", length=4, cast=float)
        or (0.0, 0.0, 0.0, 0.0),
        wall_cells=_optional_tuple(geometry_table, "wall_cells", length=4, cast=int) or (0, 0, 0, 0),
        target_ha=None if geometry_table.get("target_ha") is None else float(geometry_table["target_ha"]),
        target_side_layer=None
        if geometry_table.get("target_side_layer") is None
        else float(geometry_table["target_side_layer"]),
    )

    magnetic_field = MagneticFieldSpec(
        kind=str(_require(field_table, "kind")),
        value=_optional_tuple(field_table, "value", length=3, cast=float),
        fn=None,
        table_path=None
        if field_table.get("table_path") is None
        else str((input_path.parent / str(field_table["table_path"])).resolve()),
        ramp_start=float(field_table.get("ramp_start", 0.0)),
        ramp_duration=float(field_table.get("ramp_duration", 0.0)),
    )

    time_stepper = TimeStepperConfig(
        dt=float(_require(time_table, "dt")),
        t_final=float(_require(time_table, "t_final")),
        max_steps=int(_require(time_table, "max_steps")),
        outer_iterations=int(time_table.get("outer_iterations", 2)),
        potential_iterations=int(time_table.get("potential_iterations", 400)),
        potential_tolerance=None
        if time_table.get("potential_tolerance") is None
        else float(time_table["potential_tolerance"]),
        potential_relaxation=float(time_table.get("potential_relaxation", 1.0)),
        potential_solver=str(time_table.get("potential_solver", "auto")),
        current_reconstruction=str(time_table.get("current_reconstruction", "cell_centered")),
        post_update_potential_refresh=bool(time_table.get("post_update_potential_refresh", False)),
        steady_tolerance=float(time_table.get("steady_tolerance", 1e-8)),
        steady_potential_tolerance=None
        if time_table.get("steady_potential_tolerance") is None
        else float(time_table["steady_potential_tolerance"]),
        relaxation=float(time_table.get("relaxation", 0.35)),
        velocity_update_limit=float(time_table.get("velocity_update_limit", 1e-3)),
        velocity_update_limiter=str(time_table.get("velocity_update_limiter", "global_scale")),
        checkpoint_stride=int(time_table.get("checkpoint_stride", 1)),
    )

    solver_mode = str(solver_table.get("mode", "steady"))
    if solver_mode not in {"steady", "transient"}:
        raise ValueError(f"Unsupported solve mode {solver_mode!r}")
    solver_kind = str(solver_table.get("kind", "fully_developed_inductionless"))
    if solver_kind not in {
        "fully_developed_inductionless",
        "extruded_inductionless",
    }:
        raise ValueError(f"Unsupported solver kind {solver_kind!r}")
    solver = SolverConfig(
        kind=solver_kind,
        mode=solver_mode,
        preconditioner=str(solver_table.get("preconditioner", "jacobi")),
        time_scheme=str(solver_table.get("time_scheme", "implicit_euler")),
        coupling_iterations=int(solver_table.get("coupling_iterations", 12)),
        coupling_tolerance=float(solver_table.get("coupling_tolerance", 1e-8)),
        coupling_acceleration=str(solver_table.get("coupling_acceleration", "none")),
        coupling_min_relaxation=float(solver_table.get("coupling_min_relaxation", 0.05)),
        coupling_max_relaxation=float(solver_table.get("coupling_max_relaxation", 100.0)),
        coupling_history_depth=int(solver_table.get("coupling_history_depth", 6)),
        coupling_regularization=float(solver_table.get("coupling_regularization", 1.0e-8)),
        coupling_damping=float(solver_table.get("coupling_damping", 1.0)),
    )
    output_dir = output_table.get("directory")
    if output_dir is not None:
        output_dir = str((input_path.parent / str(output_dir)).resolve())

    output = OutputSpec(
        directory=output_dir,
        write_paraview=bool(output_table.get("write_paraview", True)),
        write_csv_profiles=bool(output_table.get("write_csv_profiles", True)),
        write_npz=bool(output_table.get("write_npz", True)),
        write_json_summary=bool(output_table.get("write_json_summary", True)),
        write_plots=bool(output_table.get("write_plots", False)),
        copy_input_file=bool(output_table.get("copy_input_file", True)),
        write_stride=int(output_table.get("write_stride", 1)),
    )

    case = CaseSpec(
        name=str(_require(case_table, "name")),
        geometry=geometry,
        regions=_parse_regions(regions_table),
        magnetic_field=magnetic_field,
        boundary_conditions=_parse_boundaries(boundaries_table),
        time_stepper=time_stepper,
        solver=solver,
        output=output,
        forcing=float(case_table.get("forcing", 1.0)),
        initial_velocity=float(case_table.get("initial_velocity", 0.0)),
        reference_pressure_gradient=float(case_table.get("reference_pressure_gradient", -1.0)),
        reference_phi_cell=_optional_tuple(case_table, "reference_phi_cell", length=2, cast=int) or (0, 0),
        notes=str(case_table.get("notes", "")),
    )

    logging = LoggingSpec.from_user_controls(
        enabled=bool(logging_table.get("enabled", True)),
        verbose=None if logging_table.get("verbose") is None else bool(logging_table.get("verbose")),
        verbosity=None if logging_table.get("verbosity") is None else str(logging_table.get("verbosity")),
        banner=bool(logging_table.get("banner", True)),
        print_regions=bool(logging_table.get("print_regions", True)),
        print_boundaries=bool(logging_table.get("print_boundaries", True)),
        print_footer=bool(logging_table.get("print_footer", True)),
        flush=bool(logging_table.get("flush", True)),
        step_stride=int(logging_table.get("step_stride", 1)),
    )
    restart_enabled = bool(restart_table.get("enabled", False))
    restart_path = restart_table.get("path")
    restart = RestartSpec(
        enabled=restart_enabled,
        path=None if restart_path is None else (input_path.parent / str(restart_path)).resolve(),
        reset_histories=bool(restart_table.get("reset_histories", True)),
        write_restart=bool(restart_table.get("write_restart", restart_enabled)),
        restart_filename=None
        if restart_table.get("restart_filename") is None
        else str(restart_table["restart_filename"]),
    )
    fringing = FringingSpec(
        enabled=bool(fringing_table.get("enabled", solver.kind == "extruded_inductionless")),
        entry_center=float(fringing_table.get("entry_center", 0.25 * geometry.length)),
        exit_center=float(fringing_table.get("exit_center", 0.75 * geometry.length)),
        transition_width=float(fringing_table.get("transition_width", max(0.05, 0.1 * geometry.length))),
        axis=str(fringing_table.get("axis", "z")),
    )

    return RunConfig(
        case=case,
        logging=logging,
        restart=restart,
        fringing=fringing,
        input_path=input_path,
    )


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
    linear_residual: float
    linear_iterations: float
    applied_forcing: float
    pressure_proxy: float
    current_scaled_pressure_proxy: float
    raw_update_max: float
    limiter_scale: float
    limited_fraction: float
    courant_like: float
    ohmic_power: float
    volumetric_flow_rate: float
    mean_current_magnitude: float
    lorentz_power: float
    div_current_max: float
    gauge_residual: float
    interface_current_residual: float
    charge_balance_residual: float = 0.0
    potential_initial_residual: float = 0.0
    linear_initial_residual: float = 0.0


class StreamingSolverLogger:
    def __init__(self, config: LoggingSpec | None = None, *, stream: TextIO | None = None) -> None:
        self.config = config or LoggingSpec()
        self.streams: list[TextIO] = [stream or sys.stdout]
        self._start_time = time.perf_counter()
        self._max_steps: int | None = None
        self._target_final_time: float | None = None
        self._mode: str = "steady"

    def add_stream(self, stream: TextIO) -> None:
        self.streams.append(stream)

    def _write(self, line: str = "") -> None:
        for stream in self.streams:
            print(line, file=stream, flush=self.config.flush)

    def _verbosity_rank(self) -> int:
        return self.config.verbosity_rank() if hasattr(self.config, "verbosity_rank") else 2

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
        if not self.config.is_enabled():
            return
        self._max_steps = int(case.time_stepper.max_steps)
        self._target_final_time = float(case.time_stepper.t_final)
        self._mode = mode
        width = 92
        if self.config.banner:
            self._write("=" * width)
            self._write(f"{'LMX Solver Run':^{width}}")
            self._write("=" * width)
        self._write("Create time")
        self._write(f"Create mesh for case        : {case.name}")
        solver = getattr(case, "solver", None)
        solver_kind = getattr(solver, "kind", "fully_developed_inductionless")
        self._write(f"Solve mode                  : {mode}")
        self._write(f"Solver kind                 : {solver_kind}")
        self._write(f"Geometry                    : {case.geometry.kind}")
        self._write(f"Cells (nx, ny, nz)          : ({mesh.nx}, {mesh.ny}, {mesh.nz})")
        self._write(
            f"Domain (L, W, H)            : ({case.geometry.length:.6e}, {case.geometry.width:.6e}, {case.geometry.height:.6e})"
        )
        self._write(
            f"Time controls               : dt={case.time_stepper.dt:.6e}, endTime={case.time_stepper.t_final:.6e}, maxSteps={case.time_stepper.max_steps}"
        )
        if solver is not None:
            self._write(
                f"Solver controls             : linearSolver=solvax_pcg, preconditioner={solver.preconditioner}, "
                f"timeScheme={solver.time_scheme}, couplingIterations={solver.coupling_iterations}, "
                f"couplingTolerance={solver.coupling_tolerance:.6e}"
            )
        self._write(
            f"Potential controls          : solver={potential_solver}, iterations={case.time_stepper.potential_iterations}, tolerance={case.time_stepper.potential_tolerance}, omega={case.time_stepper.potential_relaxation:.6e}"
        )
        self._write(
            f"Magnetic field              : kind={case.magnetic_field.kind}, value={case.magnetic_field.value}, rampStart={case.magnetic_field.ramp_start:.6e}, rampDuration={case.magnetic_field.ramp_duration:.6e}"
        )
        self._write(
            f"Flow forcing                : explicit={case.forcing:.6e}, initialVelocity={case.initial_velocity:.6e}, targetMeanVelocity={target_mean_velocity}, referenceMeanVelocity={reference_mean_velocity}"
        )
        if restart is not None and restart.enabled:
            self._write(
                f"Restart controls            : source={restart.path}, startTime={restart.start_time:.6e}, "
                f"resetHistories={restart.reset_histories}"
            )
        if self._verbosity_rank() >= 2 and self.config.print_regions:
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
        if self._verbosity_rank() >= 2 and self.config.print_boundaries:
            self._write("Read boundary conditions")
            for boundary in case.boundary_conditions:
                self._write(
                    "  "
                    f"{boundary.name:<18s} kind={boundary.kind:<22s} axis={boundary.axis or '-':<2s} "
                    f"side={boundary.side or '-':<11s} region={boundary.region or '-':<18s} value={boundary.value}"
                )
        self._write("-" * width)

    def _progress_fraction(self, record: SolverStepRecord) -> float | None:
        candidates: list[float] = []
        if self._max_steps is not None and self._max_steps > 0:
            candidates.append(record.step_index / float(self._max_steps))
        if self._target_final_time is not None and self._target_final_time > 0.0:
            candidates.append(record.time / float(self._target_final_time))
        if not candidates:
            return None
        return min(max(max(candidates), 0.0), 1.0)

    @staticmethod
    def _format_seconds(seconds: float | None) -> str:
        if seconds is None or not math.isfinite(float(seconds)):
            return "unknown"
        total = max(float(seconds), 0.0)
        minutes, secs = divmod(total, 60.0)
        hours, minutes = divmod(minutes, 60.0)
        if hours >= 1.0:
            return f"{int(hours):02d}:{int(minutes):02d}:{secs:04.1f}"
        return f"{int(minutes):02d}:{secs:04.1f}"

    def emit_step(self, record: SolverStepRecord) -> None:
        if not self.config.is_enabled():
            return
        if record.step_index > 1 and (record.step_index - 1) % max(self.config.step_stride, 1) != 0:
            return
        elapsed = time.perf_counter() - self._start_time
        progress = self._progress_fraction(record)
        average_step = elapsed / max(record.step_index, 1)
        estimated_total = elapsed / progress if progress is not None and progress > 0.0 else None
        remaining = estimated_total - elapsed if estimated_total is not None else None
        self._write(f"Time = {record.time:.6e}")
        self._write(
            "smoothSolver: potE             "
            f"Initial residual = {record.potential_initial_residual:.6e}, "
            f"Final residual = {record.potential_residual:.6e}, No Iterations {int(record.potential_iterations)}"
        )
        self._write(
            "smoothSolver: U                "
            f"Initial residual = {record.linear_initial_residual:.6e}, "
            f"Final residual = {record.linear_residual:.6e}, No Iterations {int(record.linear_iterations)}"
        )
        self._write(
            "MHD predictor                  "
            f"max|U| = {record.u_max:.6e}, mean(U) = {record.mean_velocity:.6e}, "
            f"CourantLike = {record.courant_like:.6e}"
        )
        if self._verbosity_rank() >= 2:
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
                "MHD integrals                  "
                f"Q = {record.volumetric_flow_rate:.6e}, mean|J| = {record.mean_current_magnitude:.6e}, "
                f"LorentzPower = {record.lorentz_power:.6e}"
            )
            self._write(
                "MHD conservation               "
                f"max|divJ| = {record.div_current_max:.6e}, gaugeResidual = {record.gauge_residual:.6e}, "
                f"chargeBalanceResidual = {record.charge_balance_residual:.6e}, "
                f"interfaceCurrentResidual = {record.interface_current_residual:.6e}"
            )
            self._write(
                "MHD limiter                    "
                f"rawUpdateMax = {record.raw_update_max:.6e}, limiterScale = {record.limiter_scale:.6e}, "
                f"limitedFraction = {record.limited_fraction:.6e}"
            )
        if self._verbosity_rank() >= 3:
            face_current_ratio = record.face_current_max / max(record.current_max, 1e-12)
            face_lorentz_ratio = record.face_lorentz_max / max(record.lorentz_max, 1e-12)
            self._write(
                "MHD debug                      "
                f"step = {record.step_index:d}, dt = {record.dt:.6e}, "
                f"faceCurrentRatio = {face_current_ratio:.6e}, faceLorentzRatio = {face_lorentz_ratio:.6e}"
            )
        self._write(f"steadySolver                   velocity residual = {record.residual:.6e}")
        if progress is None:
            self._write(
                "Progress                       "
                f"step = {record.step_index:d}, avgStepWallTime = {average_step:.3f} s"
            )
        else:
            self._write(
                "Progress                       "
                f"step = {record.step_index:d}/{self._max_steps if self._max_steps is not None else '-'}, "
                f"complete = {100.0 * progress:5.1f} %, avgStepWallTime = {average_step:.3f} s, "
                f"remaining ≈ {self._format_seconds(remaining)}, total ≈ {self._format_seconds(estimated_total)}"
            )
        self._write(f"ExecutionTime = {elapsed:.3f} s")
        self._write("")

    def emit_footer(self, solution: Solution) -> None:
        if not self.config.is_enabled() or not self.config.print_footer:
            return
        elapsed = time.perf_counter() - self._start_time
        self._write("-" * 92)
        self._write("End")
        self._write(f"Final time                    : {solution.state.time:.6e}")
        self._write(f"Final residual                : {solution.state.residual:.6e}")
        self._write(f"Output case                   : {solution.case_name}")
        self._write(f"ExecutionTime                 : {elapsed:.3f} s")


def default_log_path(out_dir: str | Path, case_name: str) -> Path:
    out_dir = Path(out_dir)
    return out_dir / f"{case_name}.log"
