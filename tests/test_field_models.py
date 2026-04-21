import pytest
import numpy as np

from lmx.field_models import (
    cross_section_divergence_metrics,
    load_tabulated_field,
    make_localized_divergence_free_obstacle_field,
    make_divergence_free_cross_section_field,
    sample_cross_section_field,
    sample_tabulated_cross_section_field,
    sample_tabulated_field_volume,
    write_tabulated_field_npz,
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


def test_tabulated_field_npz_round_trip_and_sampling(tmp_path):
    field_fn = make_divergence_free_cross_section_field(width=2.0, height=1.0, base_bz=8.0, perturbation=0.1)
    y, z, field = sample_cross_section_field(field_fn, width=2.0, height=1.0, ny=21, nz=25)
    path = write_tabulated_field_npz(
        tmp_path / "field.npz",
        y=y,
        z=z,
        bx=field[..., 0],
        by=field[..., 1],
        bz=field[..., 2],
    )
    payload = load_tabulated_field(path)
    assert set(payload) == {"y", "z", "bx", "by", "bz"}
    sampled = sample_tabulated_cross_section_field(path, y=field[..., 0] * 0.0 + y[:, None], z=field[..., 0] * 0.0 + z[None, :])
    assert sampled.shape == field.shape
    assert abs(float(sampled[..., 2].mean()) - float(field[..., 2].mean())) < 1.0e-8


def test_tabulated_field_volume_sampling_supports_3d_npz(tmp_path):
    x = np.linspace(0.0, 1.0, 5)
    y = np.linspace(-1.0, 1.0, 7)
    z = np.linspace(-0.5, 0.5, 9)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    bx = 0.1 * xx
    by = yy * zz
    bz = 1.0 + 0.2 * xx - 0.1 * yy
    path = write_tabulated_field_npz(tmp_path / "field3d.npz", x=x, y=y, z=z, bx=bx, by=by, bz=bz)
    sampled = sample_tabulated_field_volume(path, x=xx, y=yy, z=zz)
    assert sampled.shape == xx.shape + (3,)
    assert sampled[..., 0] == pytest.approx(bx)
