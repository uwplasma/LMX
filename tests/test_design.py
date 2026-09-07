"""Fully developed duct design: linearity, exact drive elimination and gradients."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lmx
from lmx.design import (
    DuctResponse,
    drive_for_flow_rate,
    fixed_flow_hydraulic_power,
    fluid_cell_areas,
    hydraulic_power,
    linear_flow_response,
    pressure_drop,
    volumetric_flow_rate,
)

pytestmark = pytest.mark.unit

LENGTH = 2.5


def _case(ha: float = 5.0, ny: int = 12, nz: int = 12):
    return lmx.make_hartmann_case(ha=ha, ny=ny, nz=nz)


def _flow(case, drive, scale=1.0):
    velocity, *_ = lmx.solve_fully_developed_fields(case, forcing=drive, magnetic_field_scale=scale)
    return volumetric_flow_rate(case, velocity)


def test_fluid_areas_sum_to_the_open_cross_section():
    case = _case()
    areas = np.asarray(fluid_cell_areas(case))
    assert areas.shape == (case.geometry.ny, case.geometry.nz)
    assert np.all(areas >= 0.0)
    assert float(np.sum(areas)) == pytest.approx(case.geometry.width * case.geometry.height, rel=1e-12)


def test_flow_rate_is_linear_in_the_drive():
    case = _case()
    single = _flow(case, 1.0)
    double = _flow(case, 2.0)
    assert float(double) == pytest.approx(2.0 * float(single), rel=1e-10)
    assert abs(float(_flow(case, 0.0))) < 1e-12 * abs(float(single))


def test_eliminated_drive_hits_the_requested_throughput_exactly():
    case, target = _case(), 0.35
    drive = drive_for_flow_rate(case, target)
    assert float(_flow(case, drive)) == pytest.approx(target, rel=1e-10)


def test_the_analytic_drive_derivative_matches_automatic_differentiation():
    """``df/dQ = 1/G`` because the problem is linear in the drive."""
    case, target = _case(), 0.35
    response = linear_flow_response(case)
    analytic = 1.0 / float(response.flow_per_unit_drive)
    automatic = float(jax.grad(lambda value: drive_for_flow_rate(case, value))(target))
    assert automatic == pytest.approx(analytic, rel=1e-10)
    # And the forward relation is the same number seen from the other side.
    forward = float(jax.grad(lambda drive: _flow(case, drive))(1.0))
    assert forward == pytest.approx(float(response.flow_per_unit_drive), rel=1e-10)


def test_pressure_drop_and_power_follow_the_drive():
    drive, flow = 1.4, 0.6
    assert float(pressure_drop(drive, LENGTH)) == pytest.approx(drive * LENGTH, rel=1e-12)
    assert float(hydraulic_power(drive, flow, LENGTH)) == pytest.approx(drive * LENGTH * flow, rel=1e-12)
    with pytest.raises(ValueError, match="length must be positive"):
        pressure_drop(drive, 0.0)


def test_holding_throughput_costs_more_power_in_a_stronger_field():
    """Magnetic drag: the same throughput needs a larger drive as Ha grows."""
    case, target = _case(), 0.2
    powers = [
        float(fixed_flow_hydraulic_power(case, target, LENGTH, magnetic_field_scale=scale))
        for scale in (0.5, 1.0, 2.0)
    ]
    assert powers[0] < powers[1] < powers[2]
    assert all(value > 0.0 for value in powers)


def test_fixed_flow_power_is_differentiable_in_the_field_scale():
    case, target = _case(), 0.2

    def power(scale):
        return fixed_flow_hydraulic_power(case, target, LENGTH, magnetic_field_scale=scale)

    gradient = float(jax.grad(power)(1.0))
    step = 1.0e-4
    difference = (float(power(1.0 + step)) - float(power(1.0 - step))) / (2.0 * step)
    assert gradient == pytest.approx(difference, rel=1e-5)
    # Stronger field, more drag, more power.
    assert gradient > 0.0


def test_response_object_reports_its_field_and_converts_targets():
    case = _case()
    response = linear_flow_response(case, magnetic_field_scale=1.5)
    assert isinstance(response, DuctResponse)
    assert response.magnetic_field_scale == pytest.approx(1.5)
    assert float(response.drive_for(2.0)) == pytest.approx(
        2.0 / float(response.flow_per_unit_drive), rel=1e-12
    )


def test_flow_rate_rejects_a_mismatched_velocity():
    case = _case()
    with pytest.raises(ValueError, match="does not match the mesh"):
        volumetric_flow_rate(case, jnp.zeros((3, 3)))


def test_a_conducting_wall_costs_more_power_than_an_insulating_one():
    """Hunt versus Shercliff at fixed throughput: wall currents add drag."""
    target = 0.05
    insulating = lmx.make_shercliff_case(ha=5.0, ny=12, nz=12)
    conducting = lmx.make_hunt_case(ha=5.0, ny=12, nz=12, wall_cells=2, insulator_cells=2)
    insulating_power = float(fixed_flow_hydraulic_power(insulating, target, LENGTH))
    conducting_power = float(fixed_flow_hydraulic_power(conducting, target, LENGTH))
    assert conducting_power > insulating_power
