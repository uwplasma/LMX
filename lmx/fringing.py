from __future__ import annotations

from dataclasses import dataclass, replace

import jax.numpy as jnp

from .cases import make_shercliff_case
from .core import Solution
from .specs import CaseSpec, MagneticFieldSpec
from .solvers import solve_steady
from .validation import validation_summary


@dataclass(frozen=True)
class FringingProfile:
    x: jnp.ndarray
    field_scale: jnp.ndarray
    axis: str


def smooth_fringing_profile(
    *,
    length: float,
    nx: int,
    entry_center: float,
    exit_center: float,
    transition_width: float,
    peak_scale: float = 1.0,
    axis: str = "z",
) -> FringingProfile:
    if axis not in {"x", "y", "z"}:
        raise ValueError(f"Unsupported magnetic axis {axis!r}")
    x = jnp.linspace(0.0, length, nx)
    width = max(float(transition_width), 1.0e-6)
    rise = 0.5 * (1.0 + jnp.tanh((x - entry_center) / width))
    fall = 0.5 * (1.0 - jnp.tanh((x - exit_center) / width))
    return FringingProfile(x=x, field_scale=peak_scale * rise * fall, axis=axis)


def _constant_field_on_axis(axis: str, magnitude: float) -> tuple[float, float, float]:
    if axis == "x":
        return (magnitude, 0.0, 0.0)
    if axis == "y":
        return (0.0, magnitude, 0.0)
    if axis == "z":
        return (0.0, 0.0, magnitude)
    raise ValueError(f"Unsupported magnetic axis {axis!r}")


def clone_case_with_field(case: CaseSpec, *, axis: str, magnitude: float, suffix: str | None = None) -> CaseSpec:
    magnetic_field = replace(case.magnetic_field, kind="constant", value=_constant_field_on_axis(axis, magnitude))
    name = case.name if suffix is None else f"{case.name}_{suffix}"
    return replace(case, name=name, magnetic_field=magnetic_field)


def build_square_duct_fringing_benchmark(
    *,
    ha_peak: float = 20.0,
    width: float = 2.0,
    height: float = 2.0,
    ny: int = 48,
    nz: int = 48,
    length: float = 6.0,
    nx_stations: int = 21,
    entry_center: float = 1.5,
    exit_center: float = 4.5,
    transition_width: float = 0.35,
) -> tuple[CaseSpec, FringingProfile]:
    base_case = make_shercliff_case(ha=ha_peak, width=width, height=height, ny=ny, nz=nz)
    base_case = replace(
        base_case,
        time_stepper=replace(
            base_case.time_stepper,
            max_steps=min(base_case.time_stepper.max_steps, 80),
            potential_iterations=min(base_case.time_stepper.potential_iterations, 80),
            steady_tolerance=1.0e-6,
        ),
        solver=replace(
            base_case.solver,
            coupling_iterations=min(base_case.solver.coupling_iterations, 8),
            coupling_tolerance=1.0e-7,
        ),
    )
    profile = smooth_fringing_profile(
        length=length,
        nx=nx_stations,
        entry_center=entry_center,
        exit_center=exit_center,
        transition_width=transition_width,
        peak_scale=1.0,
        axis="z",
    )
    return base_case, profile


def run_fringing_station_sweep(
    base_case: CaseSpec,
    profile: FringingProfile,
    *,
    solver=solve_steady,
) -> list[dict[str, float]]:
    if base_case.magnetic_field.kind != "constant" or base_case.magnetic_field.value is None:
        raise ValueError("Fringing station sweep requires a constant-field base case")

    base_magnitude = max(abs(float(component)) for component in base_case.magnetic_field.value)
    history: list[dict[str, float]] = []
    previous_state = None
    for index, (x_value, scale) in enumerate(zip(profile.x, profile.field_scale, strict=True)):
        station_case = clone_case_with_field(
            base_case,
            axis=profile.axis,
            magnitude=base_magnitude * float(scale),
            suffix=f"station{index:03d}",
        )
        solution: Solution = solver(station_case, initial_state=previous_state)
        metrics = validation_summary(solution, station_case.name, ha=base_case.geometry.target_ha)
        history.append(
            {
                "x": float(x_value),
                "field_scale": float(scale),
                "u_max": float(metrics["u_max"]),
                "mean_velocity": float(metrics["mean_velocity"]),
                "volumetric_flow_rate": float(metrics["volumetric_flow_rate"]),
                "current_scaled_pressure_proxy": float(metrics["current_scaled_pressure_proxy"]),
                "residual": float(solution.state.residual),
            }
        )
        previous_state = solution.state
    return history
