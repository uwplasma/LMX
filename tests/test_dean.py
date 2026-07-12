from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lmx.dean import (
    DeanVelocityPoint,
    bayat_rezai_dean_velocity,
    bayat_rezai_lateral_reynolds,
    compare_dean_velocity_points,
    dean_number_from_reynolds,
    dean_secondary_flow_field,
    dean_velocity_reference_rows,
    write_dean_literature_validation_plots,
)


pytestmark = pytest.mark.unit


def test_bayat_rezai_dean_velocity_correlation_matches_dimensionless_form():
    dean = np.asarray([2.73, 6.82, 20.0])
    velocity = bayat_rezai_dean_velocity(
        dean, kinematic_viscosity=1.0e-6, largest_channel_dimension=150.0e-6
    )
    lateral_re = bayat_rezai_lateral_reynolds(dean)

    assert np.all(velocity > 0.0)
    assert np.allclose(velocity * 150.0e-6 / 1.0e-6, lateral_re)
    assert dean_number_from_reynolds(100.0, 0.01) == pytest.approx(10.0)


def test_dean_secondary_flow_field_is_scaled_and_finite():
    y = np.linspace(-1.0, 1.0, 41)
    z = np.linspace(-1.0, 1.0, 41)
    field = dean_secondary_flow_field(y, z, tube_radius=1.0, target_rms_velocity=0.02)

    assert field["rms_velocity"] == pytest.approx(0.02)
    assert field["peak_velocity"] > field["rms_velocity"]
    assert np.all(np.isfinite(field["speed"]))
    assert np.all(np.asarray(field["speed"])[np.asarray(field["mask"]) < 0.5] == 0.0)


def test_dean_literature_validation_rows_and_plots(tmp_path: Path):
    points = [
        DeanVelocityPoint(2.73, 0.15, 1.0e-6, 150.0e-6),
        DeanVelocityPoint(6.82, 0.74, 1.0e-6, 150.0e-6),
    ]
    comparison = compare_dean_velocity_points(points)
    rows = dean_velocity_reference_rows([2.73, 6.82])
    plots = write_dean_literature_validation_plots(comparison, tmp_path)

    assert comparison["validation_pass"] is True
    assert len(rows) == 2
    assert rows[0]["source"] == "Bayat & Rezai, Sci. Rep. 2017, Eq. 8"
    assert [path.suffix for path in plots] == [".png", ".pdf"]
    assert all(path.exists() and path.stat().st_size > 0 for path in plots)
