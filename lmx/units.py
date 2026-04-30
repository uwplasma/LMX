"""Unit and nondimensional-number helpers for LMX case setup.

LMX stores ``RegionSpec.viscosity`` as the kinematic viscosity ``nu`` in
``m^2/s``.  Dynamic viscosity ``mu`` in ``Pa s`` should be converted at input
boundaries with :func:`dynamic_to_kinematic_viscosity`.
"""

from __future__ import annotations

import math


MU0 = 4.0e-7 * math.pi


def dynamic_to_kinematic_viscosity(dynamic_viscosity: float, density: float) -> float:
    """Return kinematic viscosity ``nu = mu / rho``.

    Parameters are SI values: ``dynamic_viscosity`` in ``Pa s`` and ``density``
    in ``kg/m^3``.  The returned value is in ``m^2/s``.
    """

    if density <= 0.0:
        raise ValueError("density must be positive")
    if dynamic_viscosity < 0.0:
        raise ValueError("dynamic_viscosity must be non-negative")
    return float(dynamic_viscosity) / float(density)


def kinematic_to_dynamic_viscosity(kinematic_viscosity: float, density: float) -> float:
    """Return dynamic viscosity ``mu = rho nu`` in ``Pa s``."""

    if density <= 0.0:
        raise ValueError("density must be positive")
    if kinematic_viscosity < 0.0:
        raise ValueError("kinematic_viscosity must be non-negative")
    return float(kinematic_viscosity) * float(density)


def hartmann_number(
    *,
    magnetic_field: float,
    length_scale: float,
    conductivity: float,
    density: float,
    kinematic_viscosity: float,
) -> float:
    """Return ``Ha = B a sqrt(sigma / (rho nu))``."""

    _require_positive("length_scale", length_scale)
    _require_positive("conductivity", conductivity)
    _require_positive("density", density)
    _require_positive("kinematic_viscosity", kinematic_viscosity)
    return abs(float(magnetic_field)) * float(length_scale) * math.sqrt(float(conductivity) / (float(density) * float(kinematic_viscosity)))


def magnetic_field_from_hartmann(
    *,
    hartmann: float,
    length_scale: float,
    conductivity: float,
    density: float,
    kinematic_viscosity: float,
) -> float:
    """Return ``B`` from a target Hartmann number using LMX's ``nu`` convention."""

    _require_positive("length_scale", length_scale)
    _require_positive("conductivity", conductivity)
    _require_positive("density", density)
    _require_positive("kinematic_viscosity", kinematic_viscosity)
    return float(hartmann) / (float(length_scale) * math.sqrt(float(conductivity) / (float(density) * float(kinematic_viscosity))))


def reynolds_number(*, velocity: float, length_scale: float, kinematic_viscosity: float) -> float:
    """Return ``Re = U a / nu``."""

    _require_positive("length_scale", length_scale)
    _require_positive("kinematic_viscosity", kinematic_viscosity)
    return abs(float(velocity)) * float(length_scale) / float(kinematic_viscosity)


def interaction_parameter(
    *,
    magnetic_field: float,
    length_scale: float,
    conductivity: float,
    density: float,
    velocity: float,
) -> float:
    """Return ``N = sigma B^2 a / (rho U)``."""

    _require_positive("length_scale", length_scale)
    _require_positive("conductivity", conductivity)
    _require_positive("density", density)
    _require_positive("velocity", abs(velocity))
    return float(conductivity) * float(magnetic_field) ** 2 * float(length_scale) / (float(density) * abs(float(velocity)))


def magnetic_reynolds_number(
    *,
    velocity: float,
    length_scale: float,
    conductivity: float,
    magnetic_permeability: float = MU0,
) -> float:
    """Return ``Rm = mu0 sigma U a`` for the inductionless-scope audit."""

    _require_positive("length_scale", length_scale)
    _require_positive("conductivity", conductivity)
    _require_positive("magnetic_permeability", magnetic_permeability)
    return float(magnetic_permeability) * float(conductivity) * abs(float(velocity)) * float(length_scale)


def wall_conductance_ratio(
    *,
    wall_conductivity: float,
    wall_thickness: float,
    fluid_conductivity: float,
    length_scale: float,
) -> float:
    """Return thin-wall tangential conductance ratio ``c = sigma_w t_w/(sigma_f a)``."""

    _require_positive("wall_thickness", wall_thickness)
    _require_positive("fluid_conductivity", fluid_conductivity)
    _require_positive("length_scale", length_scale)
    if wall_conductivity < 0.0:
        raise ValueError("wall_conductivity must be non-negative")
    return float(wall_conductivity) * float(wall_thickness) / (float(fluid_conductivity) * float(length_scale))


def normal_leakage_ratio(
    *,
    coating_conductivity: float,
    coating_thickness: float,
    fluid_conductivity: float,
    length_scale: float,
) -> float:
    """Return normal shunt ratio ``g_perp = (sigma_c/t_c)/(sigma_f/a)``."""

    _require_positive("coating_thickness", coating_thickness)
    _require_positive("fluid_conductivity", fluid_conductivity)
    _require_positive("length_scale", length_scale)
    if coating_conductivity < 0.0:
        raise ValueError("coating_conductivity must be non-negative")
    return float(coating_conductivity) * float(length_scale) / (float(fluid_conductivity) * float(coating_thickness))


def _require_positive(name: str, value: float) -> None:
    if value <= 0.0:
        raise ValueError(f"{name} must be positive")
