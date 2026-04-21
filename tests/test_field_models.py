import pytest

from lmx.field_models import (
    cross_section_divergence_metrics,
    make_localized_divergence_free_obstacle_field,
    make_divergence_free_cross_section_field,
    sample_cross_section_field,
)


pytestmark = pytest.mark.unit


def test_divergence_free_cross_section_field_has_small_discrete_divergence():
    field_fn = make_divergence_free_cross_section_field(width=2.0, height=1.5, base_bz=10.0, perturbation=0.1)
    metrics = cross_section_divergence_metrics(field_fn, width=2.0, height=1.5, ny=61, nz=61)
    assert metrics["max_abs_divergence"] < 0.2
    assert metrics["rms_divergence"] < 0.05


def test_sample_cross_section_field_returns_expected_shape():
    field_fn = make_divergence_free_cross_section_field(width=2.0, height=1.0, base_bz=8.0, perturbation=0.1)
    y, z, field = sample_cross_section_field(field_fn, width=2.0, height=1.0, ny=21, nz=25)
    assert y.shape == (21,)
    assert z.shape == (25,)
    assert field.shape == (21, 25, 3)


def test_localized_divergence_free_obstacle_field_has_small_discrete_divergence():
    field_fn = make_localized_divergence_free_obstacle_field(width=2.0, height=2.0, base_bz=10.0)
    metrics = cross_section_divergence_metrics(field_fn, width=2.0, height=2.0, ny=61, nz=61)
    assert metrics["max_abs_divergence"] < 0.2
    assert metrics["rms_divergence"] < 0.05
