from __future__ import annotations

from .specs import (
    BoundaryCondition,
    CaseSpec,
    GeometrySpec,
    MagneticFieldSpec,
    OutputSpec,
    RegionSpec,
    TimeStepperConfig,
)


def _ha_to_b(ha: float, length_scale: float, conductivity: float, density: float, viscosity: float) -> float:
    return ha / (length_scale * ((conductivity / (density * viscosity)) ** 0.5))


def _wall_conductivity_from_conductance_ratio(
    *,
    wall_conductance_ratio: float,
    fluid_conductivity: float,
    wall_thickness: float,
    hartmann_half_spacing: float,
) -> float:
    if wall_thickness <= 0.0:
        raise ValueError("wall_thickness must be positive when deriving wall conductivity from conductance ratio")
    if hartmann_half_spacing <= 0.0:
        raise ValueError("hartmann_half_spacing must be positive when deriving wall conductivity from conductance ratio")
    return wall_conductance_ratio * fluid_conductivity * hartmann_half_spacing / wall_thickness


def _hunt_short_transient_controls(ha: float) -> TimeStepperConfig:
    if ha <= 20.0:
        return TimeStepperConfig(
            dt=0.002,
            t_final=1.0,
            max_steps=500,
            outer_iterations=6,
            potential_iterations=400,
            relaxation=0.08,
            velocity_update_limit=2e-3,
        )
    if ha <= 100.0:
        return TimeStepperConfig(
            dt=0.002,
            t_final=1.0,
            max_steps=500,
            outer_iterations=4,
            potential_iterations=400,
            relaxation=0.1,
            velocity_update_limit=1e-3,
        )
    return TimeStepperConfig(
        dt=0.002,
        t_final=1.0,
        max_steps=500,
        outer_iterations=3,
        potential_iterations=400,
        relaxation=0.1,
        velocity_update_limit=1e-3,
    )


def make_hartmann_case(
    ha: float = 20.0,
    width: float = 2.0,
    height: float = 2.0,
    ny: int = 96,
    nz: int = 96,
    conductivity: float = 1.0,
    density: float = 1.0,
    viscosity: float = 1.0,
    output_dir: str | None = None,
) -> CaseSpec:
    bmag = _ha_to_b(ha, 0.5 * height, conductivity, density, viscosity)
    anchor = (ny // 2, nz // 2)
    return CaseSpec(
        name=f"hartmann_ha{int(ha)}",
        geometry=GeometrySpec(kind="rect_duct", width=width, height=height, ny=ny, nz=nz, target_ha=ha),
        regions=(RegionSpec("fluid", "fluid", conductivity, density, viscosity),),
        magnetic_field=MagneticFieldSpec(kind="constant", value=(0.0, 1.0 * bmag, 0.0)),
        boundary_conditions=(
            BoundaryCondition("walls", "no_slip"),
            BoundaryCondition("electric", "insulating"),
        ),
        time_stepper=TimeStepperConfig(dt=0.001, t_final=1.0, max_steps=400, potential_iterations=200, relaxation=0.1),
        output=OutputSpec(directory=output_dir),
        forcing=1.0,
        reference_pressure_gradient=-1.0,
        reference_phi_cell=anchor,
        notes="Planar Hartmann-like reference configuration for solver smoke tests.",
    )


def make_shercliff_case(
    ha: float = 20.0,
    width: float = 2.0,
    height: float = 2.0,
    ny: int = 96,
    nz: int = 96,
    conductivity: float = 1.0,
    density: float = 1.0,
    viscosity: float = 1.0,
    output_dir: str | None = None,
) -> CaseSpec:
    bmag = _ha_to_b(ha, 0.5 * width, conductivity, density, viscosity)
    anchor = (ny // 2, nz // 2)
    return CaseSpec(
        name=f"shercliff_ha{int(ha)}",
        geometry=GeometrySpec(kind="rect_duct", width=width, height=height, ny=ny, nz=nz, target_ha=ha),
        regions=(RegionSpec("fluid", "fluid", conductivity, density, viscosity),),
        magnetic_field=MagneticFieldSpec(kind="constant", value=(0.0, 0.0, 1.0 * bmag)),
        boundary_conditions=(
            BoundaryCondition("walls", "no_slip"),
            BoundaryCondition("electric", "insulating"),
        ),
        time_stepper=TimeStepperConfig(dt=0.001, t_final=1.5, max_steps=400, potential_iterations=225, relaxation=0.1),
        output=OutputSpec(directory=output_dir),
        forcing=1.0,
        reference_pressure_gradient=-1.0,
        reference_phi_cell=anchor,
        notes="All-insulating Shercliff-style duct. Analytical parity hooks are staged through validation utilities.",
    )


def make_hunt_case(
    ha: float = 20.0,
    width: float = 2.0,
    height: float = 2.0,
    ny: int = 72,
    nz: int = 72,
    wall_cells: int = 8,
    wall_thickness: float = 0.1,
    fluid_conductivity: float = 1.0,
    wall_conductance_ratio: float = 0.05,
    wall_conductivity: float | None = None,
    density: float = 1.0,
    viscosity: float = 1.0,
    output_dir: str | None = None,
) -> CaseSpec:
    bmag = _ha_to_b(ha, 0.5 * width, fluid_conductivity, density, viscosity)
    if wall_conductivity is None:
        wall_conductivity = _wall_conductivity_from_conductance_ratio(
            wall_conductance_ratio=wall_conductance_ratio,
            fluid_conductivity=fluid_conductivity,
            wall_thickness=wall_thickness,
            hartmann_half_spacing=0.5 * height,
        )
    anchor = ((ny + wall_cells * 0) // 2, (nz + 2 * wall_cells) // 2)
    controls = _hunt_short_transient_controls(ha)
    return CaseSpec(
        name=f"hunt_ha{int(ha)}",
        geometry=GeometrySpec(
            kind="layered_duct",
            width=width,
            height=height,
            ny=ny,
            nz=nz,
            wall_thickness=(0.0, 0.0, wall_thickness, wall_thickness),
            wall_cells=(0, 0, wall_cells, wall_cells),
            target_ha=ha,
        ),
        regions=(
            RegionSpec("fluid", "fluid", fluid_conductivity, density, viscosity),
            RegionSpec("conducting_wall", "solid", wall_conductivity, density, viscosity, wall_thickness),
        ),
        magnetic_field=MagneticFieldSpec(kind="constant", value=(0.0, 0.0, 1.0 * bmag)),
        boundary_conditions=(
            BoundaryCondition("walls", "no_slip"),
            BoundaryCondition("conducting_hartmann_walls", "conducting_wall", region="conducting_wall"),
        ),
        time_stepper=controls,
        output=OutputSpec(directory=output_dir),
        forcing=1.0,
        reference_pressure_gradient=-1.0,
        reference_phi_cell=anchor,
        notes=(
            "Hunt-style duct with explicit conducting Hartmann-wall layers. "
            f"Default wall conductance ratio c={wall_conductance_ratio:g}."
        ),
    )
