import importlib.util

import pytest
import numpy as np

from lmx.field_models import (
    cross_section_divergence_metrics,
    load_tabulated_field,
    load_wham_coil_model_script,
    make_localized_divergence_free_obstacle_field,
    make_divergence_free_cross_section_field,
    sample_cross_section_field,
    sample_tabulated_cross_section_field,
    sample_tabulated_field_volume,
    sample_wham_mirror_axis_profile,
    sample_wham_mirror_field,
    tabulated_cross_section_reconstruction_metrics,
    tabulated_field_quality_metrics,
    wham_mirror_station_scale,
    write_tabulated_field_npz,
    write_wham_mirror_field_npz,
)


pytestmark = pytest.mark.unit


def _skip_without_magpylib_jax() -> None:
    if importlib.util.find_spec("magpylib_jax") is None:
        pytest.skip("magpylib_jax is optional and is not installed in the bounded CI environment")


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
    quality = tabulated_field_quality_metrics(path)
    assert quality["dimension"] == 2
    assert quality["axis_monotonic"] is True
    assert quality["validation_pass"] is True
    assert quality["interpolation_node_linf_error"] < 1.0e-12


def test_tabulated_cross_section_reconstruction_metrics_compare_solver_points(tmp_path):
    field_fn = make_divergence_free_cross_section_field(width=2.0, height=1.0, base_bz=8.0, perturbation=0.1)
    y, z, field = sample_cross_section_field(field_fn, width=2.0, height=1.0, ny=41, nz=41)
    path = write_tabulated_field_npz(
        tmp_path / "field.npz",
        y=y,
        z=z,
        bx=field[..., 0],
        by=field[..., 1],
        bz=field[..., 2],
    )
    solver_y = np.linspace(-0.95, 0.95, 13)
    solver_z = np.linspace(-0.45, 0.45, 11)
    metrics = tabulated_cross_section_reconstruction_metrics(
        path,
        reference_field_fn=field_fn,
        y=solver_y,
        z=solver_z,
    )
    assert metrics["sample_count"] == 13 * 11
    assert metrics["relative_l2_error"] < 1.0e-3
    assert metrics["validation_pass"] is True


def test_tabulated_field_volume_sampling_supports_3d_npz(tmp_path):
    x = np.linspace(0.0, 1.0, 5)
    y = np.linspace(-1.0, 1.0, 7)
    z = np.linspace(-0.5, 0.5, 9)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    bx = np.sin(yy)
    by = -0.25 * zz
    bz = 1.0 + 0.25 * yy
    path = write_tabulated_field_npz(tmp_path / "field3d.npz", x=x, y=y, z=z, bx=bx, by=by, bz=bz)
    sampled = sample_tabulated_field_volume(path, x=xx, y=yy, z=zz)
    assert sampled.shape == xx.shape + (3,)
    assert sampled[..., 0] == pytest.approx(bx)
    quality = tabulated_field_quality_metrics(path)
    assert quality["dimension"] == 3
    assert quality["axis_names"] == "x,y,z"
    assert quality["validation_pass"] is True
    assert quality["normalized_magnitude_max"] == pytest.approx(1.0)


def test_wham_mirror_axis_profile_is_symmetric():
    _skip_without_magpylib_jax()
    x = np.linspace(-0.4, 0.4, 9)
    profile = np.asarray(sample_wham_mirror_axis_profile(x, coil_separation=1.2, radial_loops=4, axial_loops=2), dtype=float)
    assert profile.shape == (9,)
    assert np.all(np.isfinite(profile))
    assert profile == pytest.approx(profile[::-1], rel=1.0e-6, abs=1.0e-8)


def test_wham_mirror_station_scale_is_normalized():
    _skip_without_magpylib_jax()
    x = np.linspace(-0.5, 0.5, 11)
    scale = np.asarray(wham_mirror_station_scale(x, coil_separation=1.3, radial_loops=4, axial_loops=2), dtype=float)
    assert scale.shape == (11,)
    assert np.max(scale) == pytest.approx(1.0)
    assert np.min(scale) >= 0.0


def test_write_wham_mirror_field_npz_round_trip(tmp_path):
    _skip_without_magpylib_jax()
    x = np.linspace(-0.3, 0.3, 5)
    y = np.linspace(-0.1, 0.1, 4)
    z = np.linspace(-0.1, 0.1, 4)
    path = write_wham_mirror_field_npz(
        tmp_path / "wham_field.npz",
        x=x,
        y=y,
        z=z,
        coil_separation=1.2,
        radial_loops=3,
        axial_loops=2,
    )
    payload = load_tabulated_field(path)
    assert set(payload) == {"x", "y", "z", "bx", "by", "bz"}
    sampled = np.asarray(sample_wham_mirror_field(*np.meshgrid(x, y, z, indexing="ij"), coil_separation=1.2, radial_loops=3, axial_loops=2))
    assert sampled.shape == (5, 4, 4, 3)

    solver_x = np.linspace(0.0, 0.6, 5)
    shifted_path = write_wham_mirror_field_npz(
        tmp_path / "wham_shifted_field.npz",
        x=solver_x,
        y=y,
        z=z,
        coil_separation=1.2,
        radial_loops=3,
        axial_loops=2,
        x_offset=-0.3,
    )
    shifted_payload = load_tabulated_field(shifted_path)
    shifted_reference = np.asarray(
        sample_wham_mirror_field(
            *np.meshgrid(solver_x - 0.3, y, z, indexing="ij"),
            coil_separation=1.2,
            radial_loops=3,
            axial_loops=2,
        )
    )
    assert shifted_payload["x"][0] == pytest.approx(0.0)
    assert shifted_payload["x"][-1] == pytest.approx(0.6)
    assert shifted_payload["bz"] == pytest.approx(shifted_reference[..., 2])


def test_load_wham_coil_model_script_extracts_reduced_loop_parameters(tmp_path):
    script = tmp_path / "coil_model.py"
    script.write_text(
        "dz_HF = 14.3e-3 * 8\n"
        "r_in_HF = 0.5*86e-3\n"
        "r_out_HF = 0.5*730e-3\n"
        "nz = 8\n"
        "nr = 310\n"
        "I_coil = 2000 * 17.0 / 17.51\n"
        "HF1.position = (0,0,-0.98)\n"
        "HF2 = HF1.copy(position=(0,0,0.98))\n",
        encoding="utf-8",
    )

    params = load_wham_coil_model_script(script, radial_loops=31, axial_loops=4)

    assert params["coil_separation"] == pytest.approx(1.96)
    assert params["inner_radius"] == pytest.approx(0.043)
    assert params["outer_radius"] == pytest.approx(0.365)
    assert params["source_radial_loops"] == 310
    assert params["source_axial_loops"] == 8
    assert params["radial_loops"] == 31
    assert params["axial_loops"] == 4
    assert params["current_scale"] == pytest.approx(params["source_ampere_turns"] / (31 * 4))
