from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from lmx.cases import make_hartmann_case, make_hunt_case
from lmx.core import Diagnostics, MHDState, Solution
from lmx.plotting import (
    _add_fluid_outline,
    _format_time_with_units,
    _movie_field_stack,
    _plot_field,
    _prepare_plot_output,
    _preferred_animation_writer,
    _save_figure_pair,
    write_autodiff_plots,
    write_bent_pipe_overview_plots,
    write_case_overview_plots,
    write_cross_section_field_plots,
    write_freemhd_observable_parity_plots,
    write_geometry_gallery_plots,
    write_geometry_preview_plots,
    write_interface_verification_plots,
    write_freemhd_parity_plots,
    write_magnetic_obstacle_benchmark_plots,
    write_magnetic_obstacle_regime_plots,
    write_magnetic_obstacle_schematic_plots,
    write_operator_verification_plots,
    write_strong_scaling_plots,
    write_tabulated_field_reconstruction_plots,
    write_wham_mirror_overview_plots,
    write_transient_movies,
)
from lmx.field_models import write_tabulated_field_npz
from lmx.fringing import (
    build_bent_pipe_extruded_problem,
    build_magnetic_obstacle_rect_extruded_problem,
    solve_extruded_inductionless,
)
from lmx.solvers import _build_mesh


pytestmark = pytest.mark.unit


def _sample_solution(case) -> Solution:
    mesh = _build_mesh(case)
    y, z = jnp.meshgrid(mesh.y_centers, mesh.z_centers, indexing="ij")
    u = 1.0 - 0.2 * y**2 - 0.3 * z**2
    phi = y - z
    zeros = jnp.zeros_like(u)
    diagnostics = Diagnostics(
        residual_history=jnp.asarray([1.0e-2, 1.0e-4]),
        courant_like=jnp.asarray([0.1, 0.05]),
        ohmic_power=jnp.asarray([0.2, 0.1]),
        time_history=jnp.asarray([0.0, 0.1]),
        u_max_history=jnp.asarray([0.8, 0.9]),
        mean_velocity_history=jnp.asarray([0.5, 0.55]),
        applied_forcing_history=jnp.asarray([1.0, 1.0]),
        pressure_proxy_history=jnp.asarray([0.2, 0.18]),
        current_scaled_pressure_proxy_history=jnp.asarray([0.15, 0.13]),
        raw_update_max_history=jnp.asarray([0.05, 0.02]),
        limiter_scale_history=jnp.asarray([1.0, 1.0]),
        limited_fraction_history=jnp.asarray([0.0, 0.0]),
        current_max_history=jnp.asarray([0.4, 0.35]),
        face_current_max_history=jnp.asarray([0.38, 0.33]),
        emf_max_history=jnp.asarray([0.25, 0.2]),
        lorentz_max_history=jnp.asarray([0.18, 0.16]),
        face_lorentz_max_history=jnp.asarray([0.16, 0.14]),
        potential_residual_history=jnp.asarray([1.0e-3, 1.0e-4]),
        potential_iterations_history=jnp.asarray([8.0, 6.0]),
        linear_residual_history=jnp.asarray([1.0e-2, 1.0e-5]),
        linear_iterations_history=jnp.asarray([12.0, 8.0]),
        volumetric_flow_rate_history=jnp.asarray([0.9, 1.0]),
        mean_current_magnitude_history=jnp.asarray([0.2, 0.18]),
        lorentz_power_history=jnp.asarray([0.1, 0.09]),
        div_current_max_history=jnp.asarray([1.0e-6, 5.0e-7]),
        charge_balance_residual_history=jnp.asarray([1.0e-7, 8.0e-8]),
        gauge_residual_history=jnp.asarray([1.0e-8, 5.0e-9]),
        interface_current_residual_history=jnp.asarray([1.0e-6, 8.0e-7]),
    )
    return Solution(
        mesh=mesh,
        state=MHDState(
            u=u,
            phi=phi,
            jy=zeros,
            jz=zeros,
            lorentz_x=zeros,
            time=0.1,
            residual=1.0e-5,
        ),
        diagnostics=diagnostics,
        case_name=case.name,
    )


def _sample_frames(case, *, signed: bool) -> list[dict[str, object]]:
    mesh = _build_mesh(case)
    y, z = np.meshgrid(
        np.asarray(mesh.y_centers), np.asarray(mesh.z_centers), indexing="ij"
    )
    if signed:
        base = 0.3 * np.sin(y) - 0.2 * np.cos(z)
    else:
        base = 0.2 + 0.1 * (1.0 - 0.2 * y**2 - 0.3 * z**2)
    frames = []
    for idx, scale in enumerate((1.0, 1.5, 2.0)):
        frames.append(
            {
                "time": 1e-4 * idx,
                "u": base * scale,
                "phi": np.zeros_like(base),
                "jy": np.zeros_like(base),
                "jz": np.zeros_like(base),
                "lorentz_x": np.zeros_like(base),
                "fluid_mask": np.ones_like(base, dtype=bool),
                "residual": 0.0,
                "potential_residual": 0.0,
                "potential_iterations": 1.0,
                "face_current_max": 0.0,
                "emf_max": 0.0,
                "face_lorentz_max": 0.0,
                "mean_velocity": float(np.mean(base)),
                "applied_forcing": 0.0,
                "pressure_proxy": 0.0,
                "mesh": mesh,
            }
        )
    return frames


def _assert_figure_pair(outputs: list[Path], out_dir: Path, stem: str) -> None:
    expected = [out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"]
    assert outputs == expected
    assert all(path.exists() for path in expected)


@pytest.fixture(autouse=True)
def _stub_repeated_figure_pair_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    def save(
        fig,
        out_dir: Path,
        stem: str,
        *,
        dpi: int | None = None,
        tight: bool = True,
    ) -> list[Path]:
        paths = [out_dir / f"{stem}.png", out_dir / f"{stem}.pdf"]
        for path in paths:
            path.write_bytes(b"plot")
        plt.close(fig)
        return paths

    monkeypatch.setattr("lmx.plotting._save_figure_pair", save)


def test_figure_pair_owner_writes_real_png_and_pdf(tmp_path: Path):
    out_dir = _prepare_plot_output(tmp_path)
    fig, ax = plt.subplots()
    ax.plot([0.0, 1.0], [0.0, 1.0])
    png, pdf = _save_figure_pair(fig, out_dir, "formats")
    assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert pdf.read_bytes().startswith(b"%PDF")


@pytest.mark.parametrize(
    ("options", "expected"),
    [({}, {"bbox_inches": "tight"}), ({"dpi": 185}, {"bbox_inches": "tight", "dpi": 185}), ({"tight": False}, {})],
)
def test_figure_pair_owner_preserves_save_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    options: dict[str, object],
    expected: dict[str, object],
):
    calls = []
    fig = SimpleNamespace(savefig=lambda path, **kwargs: calls.append((path, kwargs)))
    monkeypatch.setattr(plt, "close", lambda figure: None)

    paths = _save_figure_pair(fig, tmp_path, "options", **options)

    assert paths == [tmp_path / "options.png", tmp_path / "options.pdf"]
    assert calls == [(path, expected) for path in paths]


def test_write_case_overview_plots_writes_overview_and_diagnostics(tmp_path: Path):
    solution = _sample_solution(make_hartmann_case(ha=5.0, ny=8, nz=8))
    outputs = write_case_overview_plots(
        solution,
        tmp_path,
        case_title="Hartmann demo",
        y_reference_coordinate=jnp.asarray([-1.0, 0.0, 1.0]),
        y_reference_values=jnp.asarray([0.0, 1.0, 0.0]),
        z_reference_coordinate=jnp.asarray([-1.0, 0.0, 1.0]),
        z_reference_values=jnp.asarray([0.0, 1.0, 0.0]),
    )
    _assert_figure_pair(outputs[:2], tmp_path, "overview")
    _assert_figure_pair(outputs[2:], tmp_path, "diagnostics")


def test_write_case_overview_plots_skips_diagnostics_when_no_time_history(
    tmp_path: Path,
):
    solution = _sample_solution(make_hartmann_case(ha=5.0, ny=8, nz=8))
    solution = Solution(
        mesh=solution.mesh,
        state=solution.state,
        diagnostics=Diagnostics(
            residual_history=jnp.asarray([]),
            courant_like=jnp.asarray([]),
            ohmic_power=jnp.asarray([]),
            time_history=jnp.asarray([]),
        ),
        case_name=solution.case_name,
    )
    outputs = write_case_overview_plots(solution, tmp_path, case_title="No diagnostics")
    _assert_figure_pair(outputs, tmp_path, "overview")
    assert not (tmp_path / "diagnostics.png").exists()


def test_write_geometry_preview_plots_writes_rectangular_and_pipe_outputs(
    tmp_path: Path,
):
    rect_mesh = _build_mesh(make_hartmann_case(ha=5.0, ny=8, nz=8))
    rect_outputs = write_geometry_preview_plots(
        rect_mesh, tmp_path / "rect", case_title="Rectangular geometry"
    )
    _assert_figure_pair(rect_outputs, tmp_path / "rect", "geometry_preview")

    pipe_mesh = rect_mesh.__class__(
        **{
            **rect_mesh.__dict__,
            "geometry": "pipe_ogrid",
            "point_coordinates": jnp.asarray(
                [
                    [
                        [[0.0, -0.5, -0.5], [0.0, -0.5, 0.5]],
                        [[0.0, 0.5, -0.5], [0.0, 0.5, 0.5]],
                    ],
                    [
                        [[1.0, -0.5, -0.5], [1.0, -0.5, 0.5]],
                        [[1.0, 0.5, -0.5], [1.0, 0.5, 0.5]],
                    ],
                ]
            ),
        }
    )
    pipe_outputs = write_geometry_preview_plots(
        pipe_mesh, tmp_path / "pipe", case_title="Pipe preview"
    )
    _assert_figure_pair(pipe_outputs, tmp_path / "pipe", "geometry_preview")


def test_write_geometry_gallery_plots_writes_panel_outputs(tmp_path: Path):
    rect_mesh = _build_mesh(make_hartmann_case(ha=5.0, ny=8, nz=8))
    layered_mesh = _build_mesh(make_hunt_case(ha=20.0, ny=6, nz=6, wall_cells=1))
    pipe_mesh = rect_mesh.__class__(
        **{
            **rect_mesh.__dict__,
            "geometry": "pipe_ogrid",
            "point_coordinates": jnp.asarray(
                [
                    [
                        [[0.0, -0.5, -0.5], [0.0, -0.5, 0.5]],
                        [[0.0, 0.5, -0.5], [0.0, 0.5, 0.5]],
                    ],
                    [
                        [[1.0, -0.5, -0.5], [1.0, -0.5, 0.5]],
                        [[1.0, 0.5, -0.5], [1.0, 0.5, 0.5]],
                    ],
                ]
            ),
        }
    )
    outputs = write_geometry_gallery_plots(
        [
            ("Rectangular duct", rect_mesh, rect_mesh.fluid_mask),
            ("Layered duct", layered_mesh, layered_mesh.fluid_mask),
            ("Mapped pipe O-grid", pipe_mesh, pipe_mesh.fluid_mask),
        ],
        tmp_path,
        title="Geometry gallery",
    )
    _assert_figure_pair(outputs, tmp_path, "geometry_gallery")


def test_plot_field_handles_negative_only_field():
    solution = _sample_solution(make_hartmann_case(ha=5.0, ny=8, nz=8))
    fig, ax = plt.subplots()
    try:
        _plot_field(
            ax,
            solution,
            -jnp.abs(solution.state.u),
            title="Negative field",
            cmap="RdBu_r",
        )
        assert ax.get_title() == "Negative field"
    finally:
        plt.close(fig)


def test_movie_field_stack_supports_raw_and_bulk_deviation():
    frames = _sample_frames(
        make_hunt_case(ha=20.0, ny=4, nz=4, wall_cells=1), signed=False
    )
    raw_fields, raw_peaks, raw_label, raw_colorbar = _movie_field_stack(
        frames, field_mode="raw"
    )
    dev_fields, dev_peaks, dev_label, dev_colorbar = _movie_field_stack(
        frames, field_mode="bulk_deviation"
    )

    assert len(raw_fields) == len(frames)
    assert len(dev_fields) == len(frames)
    assert raw_label == "Velocity u"
    assert raw_colorbar == "u"
    assert dev_label == "Velocity deviation"
    assert dev_colorbar == "u - <u>_fluid"
    assert raw_peaks[0] > 0.0
    assert dev_peaks[0] > 0.0


def test_movie_field_stack_rejects_unknown_mode():
    frames = _sample_frames(make_hartmann_case(ha=5.0, ny=4, nz=4), signed=True)
    with pytest.raises(ValueError, match="Unsupported field_mode"):
        _movie_field_stack(frames, field_mode="nope")


def test_write_transient_movies_returns_empty_list_for_empty_frames(tmp_path: Path):
    assert write_transient_movies([], tmp_path, case_title="empty") == []


def test_write_freemhd_parity_plots_writes_outputs(tmp_path: Path):
    records = [
        {
            "case_kind": "shercliff",
            "freemhd_execution_seconds": 35.0,
            "lmx_execution_seconds": 2.5,
            "u_max_abs_diff": 1.0e-3,
            "y_l2_error": 5.0e-3,
            "z_l2_error": 3.0e-3,
            "freemhd_u_max_history": {"time": [0.0, 1.0e-5], "value": [0.0, 1.0]},
            "lmx_u_max_history": {"time": [0.0, 1.0e-5], "value": [0.0, 0.98]},
            "y_profile": {
                "coordinate": np.linspace(-1.0, 1.0, 5).tolist(),
                "simulated": np.array([0.0, 0.8, 1.0, 0.8, 0.0]).tolist(),
                "reference": np.array([0.0, 0.82, 1.0, 0.82, 0.0]).tolist(),
            },
            "z_profile": {
                "coordinate": np.linspace(-1.0, 1.0, 5).tolist(),
                "simulated": np.array([0.0, 0.75, 1.0, 0.75, 0.0]).tolist(),
                "reference": np.array([0.0, 0.77, 1.0, 0.77, 0.0]).tolist(),
            },
        },
        {
            "case_kind": "hunt",
            "freemhd_execution_seconds": 36.0,
            "lmx_execution_seconds": 2.8,
            "u_max_abs_diff": 2.0e-3,
            "y_l2_error": 8.0e-3,
            "z_l2_error": 6.0e-3,
            "freemhd_u_max_history": {"time": [0.0, 1.0e-5], "value": [0.0, 1.0]},
            "lmx_u_max_history": {"time": [0.0, 1.0e-5], "value": [0.0, 0.97]},
            "y_profile": {
                "coordinate": np.linspace(-1.0, 1.0, 5).tolist(),
                "simulated": np.array([0.0, 0.9, 0.8, 0.9, 0.0]).tolist(),
                "reference": np.array([0.0, 0.92, 0.78, 0.92, 0.0]).tolist(),
            },
            "z_profile": {
                "coordinate": np.linspace(-1.0, 1.0, 5).tolist(),
                "simulated": np.array([0.0, 0.7, 1.0, 0.7, 0.0]).tolist(),
                "reference": np.array([0.0, 0.72, 1.0, 0.72, 0.0]).tolist(),
            },
        },
    ]
    outputs = write_freemhd_parity_plots(records, tmp_path, case_title="Parity demo")
    _assert_figure_pair(outputs, tmp_path, "freemhd_closed_channel_parity")


def test_write_freemhd_observable_parity_plots_writes_outputs(tmp_path: Path):
    cut = {
        "coordinate": np.linspace(-1.0, 1.0, 5).tolist(),
        "simulated": np.array([0.0, 0.9, 1.0, 0.9, 0.0]).tolist(),
        "reference": np.array([0.0, 0.92, 1.0, 0.92, 0.0]).tolist(),
        "l2_error": 5.0e-3,
        "linf_error": 2.0e-2,
    }
    records = [
        {
            "case_kind": "shercliff",
            "observables": {
                name: {"y": {**cut}, "z": {**cut}, "peak_ratio": 0.98}
                for name in ("velocity", "potential", "current", "lorentz")
            },
        },
        {
            "case_kind": "hunt",
            "observables": {
                name: {"y": {**cut}, "z": {**cut}, "peak_ratio": 1.02}
                for name in ("velocity", "potential", "current", "lorentz")
            },
        },
    ]
    outputs = write_freemhd_observable_parity_plots(
        records, tmp_path, case_title="Observable parity"
    )
    _assert_figure_pair(
        outputs, tmp_path, "freemhd_closed_channel_observable_parity"
    )


def test_preferred_animation_writer_selects_one_backend(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(
        "matplotlib.animation.writers.list", lambda: ["imagemagick", "pillow"]
    )
    assert _preferred_animation_writer() == ("gif", "pillow")
    monkeypatch.setattr("matplotlib.animation.writers.list", lambda: ["imagemagick"])
    assert _preferred_animation_writer() == ("gif", "imagemagick")


def test_small_plot_helpers_cover_units_and_safe_fallback(monkeypatch):
    assert [_format_time_with_units(value) for value in (1.0e-4, 0.1, 2.0)] == [
        "100.0 μs",
        "100.0 ms",
        "2.00 s",
    ]
    monkeypatch.setattr("matplotlib.animation.writers.list", lambda: [])
    assert _preferred_animation_writer() == ("gif", "pillow")

    def reject_contour(*args, **kwargs):
        raise ValueError("constant mask")

    mesh = SimpleNamespace(y_centers=[0.0, 1.0], z_centers=[0.0, 1.0])
    _add_fluid_outline(SimpleNamespace(), mesh, None)
    _add_fluid_outline(SimpleNamespace(contour=reject_contour), mesh, np.ones((2, 2)))


def test_write_transient_movies_writes_posters_and_stubbed_gifs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    frames = _sample_frames(
        make_hunt_case(ha=20.0, ny=4, nz=4, wall_cells=1), signed=True
    )

    saved = []

    def fake_save(self, filename, *args, **kwargs):
        self._draw_was_started = True
        Path(filename).write_bytes(b"gif")
        saved.append(Path(filename))

    monkeypatch.setattr("matplotlib.animation.FuncAnimation.save", fake_save)

    outputs = write_transient_movies(
        frames,
        tmp_path,
        case_title="Hunt startup",
        fps=4,
        field_mode="bulk_deviation",
        output_stem="hunt_demo",
    )

    expected = {
        tmp_path / "hunt_demo_2d_poster.png",
        tmp_path / "hunt_demo_2d_poster.pdf",
        tmp_path / "hunt_demo_3d_poster.png",
        tmp_path / "hunt_demo_3d_poster.pdf",
        tmp_path / "hunt_demo_2d.gif",
        tmp_path / "hunt_demo_3d.gif",
    }
    assert expected.issubset(set(outputs))
    assert saved == [tmp_path / "hunt_demo_2d.gif", tmp_path / "hunt_demo_3d.gif"]
    for path in expected:
        assert path.exists()


def test_write_transient_movies_handles_positive_only_normalized_path_and_contour_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    frames = _sample_frames(
        make_hunt_case(ha=20.0, ny=4, nz=4, wall_cells=1), signed=False
    )

    def fake_save(self, filename, *args, **kwargs):
        self._draw_was_started = True
        self._func(0)
        self._func(1)
        Path(filename).write_bytes(b"gif")

    monkeypatch.setattr("matplotlib.animation.FuncAnimation.save", fake_save)

    outputs = write_transient_movies(
        frames,
        tmp_path,
        case_title="Positive startup",
        fps=4,
        field_mode="raw",
        output_stem="positive_demo",
    )

    assert (tmp_path / "positive_demo_2d.gif") in outputs
    assert (tmp_path / "positive_demo_3d.gif") in outputs


def test_write_strong_scaling_plots_writes_png_and_pdf(tmp_path: Path):
    records = [
        {"platform": "CPU", "num_devices": 1, "mean_seconds": 4.0},
        {"platform": "CPU", "num_devices": 2, "mean_seconds": 2.4},
        {"platform": "GPU", "num_devices": 1, "warm_seconds": 1.8},
        {"platform": "GPU", "num_devices": 2, "mean_seconds": 1.0},
    ]
    outputs = write_strong_scaling_plots(
        records, tmp_path, case_title="Scaling", resource_label="Worker count"
    )

    _assert_figure_pair(outputs, tmp_path, "strong_scaling")


def test_write_autodiff_plots_writes_png_and_pdf(tmp_path: Path):
    sensitivity_scan = [
        {"hartmann_number": 2.0, "mean_velocity": 0.4, "d_mean_velocity_d_ha": -0.03},
        {"hartmann_number": 10.0, "mean_velocity": 0.2, "d_mean_velocity_d_ha": -0.01},
    ]
    optimization_history = [
        {"iteration": 0.0, "hartmann_number": 4.0, "loss": 1.0e-1, "gradient": -0.02},
        {"iteration": 1.0, "hartmann_number": 6.0, "loss": 2.0e-2, "gradient": -0.01},
    ]
    outputs = write_autodiff_plots(
        sensitivity_scan,
        optimization_history,
        tmp_path,
        case_title="Autodiff",
        target_parameter=8.0,
    )

    _assert_figure_pair(outputs, tmp_path, "autodiff_summary")


def test_write_operator_verification_plots_writes_png_and_pdf(tmp_path: Path):
    records = [
        {
            "resolution": 16.0,
            "max_spacing": 0.125,
            "gradient_y_l2_error": 2.0e-2,
            "gradient_z_l2_error": 2.2e-2,
            "laplacian_l2_error": 5.0e-2,
        },
        {
            "resolution": 32.0,
            "max_spacing": 0.0625,
            "gradient_y_l2_error": 5.0e-3,
            "gradient_z_l2_error": 5.5e-3,
            "laplacian_l2_error": 1.3e-2,
        },
        {
            "resolution": 64.0,
            "max_spacing": 0.03125,
            "gradient_y_l2_error": 1.3e-3,
            "gradient_z_l2_error": 1.4e-3,
            "laplacian_l2_error": 3.3e-3,
        },
    ]
    outputs = write_operator_verification_plots(
        records, tmp_path, case_title="Operator verification"
    )

    _assert_figure_pair(outputs, tmp_path, "operator_verification")


def test_write_interface_verification_plots_writes_png_and_pdf(tmp_path: Path):
    records = [
        {
            "resolution": 24.0,
            "max_spacing": 0.0833,
            "profile_l2_error": 1.0e-6,
            "flux_error": 2.0e-6,
        },
        {
            "resolution": 48.0,
            "max_spacing": 0.0417,
            "profile_l2_error": 2.5e-7,
            "flux_error": 5.0e-7,
        },
        {
            "resolution": 96.0,
            "max_spacing": 0.0208,
            "profile_l2_error": 6.0e-8,
            "flux_error": 1.2e-7,
        },
    ]
    profile = {
        "y": np.linspace(-1.0, 1.0, 9),
        "u_exact": np.linspace(0.0, 1.0, 9),
        "u_numeric": np.linspace(0.0, 1.0, 9) + 1.0e-3,
    }
    outputs = write_interface_verification_plots(
        records, profile, tmp_path, case_title="Interface verification"
    )

    _assert_figure_pair(outputs, tmp_path, "interface_verification")


def test_write_cross_section_field_plots_writes_png_and_pdf(tmp_path: Path):
    y = np.linspace(-1.0, 1.0, 9)
    z = np.linspace(-1.0, 1.0, 11)
    yy, zz = np.meshgrid(y, z, indexing="ij")
    field = np.stack([np.zeros_like(yy), 0.2 * yy, 1.0 + 0.1 * zz], axis=-1)
    outputs = write_cross_section_field_plots(
        y=y, z=z, field=field, out_dir=tmp_path, title="Field preview"
    )
    _assert_figure_pair(outputs, tmp_path, "field_preview")


def test_write_tabulated_field_reconstruction_plots_writes_png_and_pdf(tmp_path: Path):
    y = np.linspace(-1.0, 1.0, 9)
    z = np.linspace(-1.0, 1.0, 11)
    yy, zz = np.meshgrid(y, z, indexing="ij")
    reference = np.stack([np.zeros_like(yy), 0.2 * yy, 1.0 + 0.1 * zz], axis=-1)
    sampled = reference + 1.0e-4
    outputs = write_tabulated_field_reconstruction_plots(
        y=y,
        z=z,
        reference_field=reference,
        tabulated_field=sampled,
        out_dir=tmp_path,
    )
    _assert_figure_pair(outputs, tmp_path, "tabulated_field_reconstruction")


def test_write_bent_pipe_overview_plots_writes_png_and_pdf(tmp_path: Path):
    bent_problem = build_bent_pipe_extruded_problem(
        ha_peak=4.0, bend_radius=4.0, bend_angle=1.0, nx_stations=4, nr=4, ntheta=12
    )
    geometry = bent_problem.case.geometry
    radius = float(geometry.radius)
    r = radius * (np.arange(geometry.nr) + 0.5) / geometry.nr
    theta = 2.0 * np.pi * (np.arange(geometry.ntheta) + 0.5) / geometry.ntheta
    x = np.linspace(0.0, float(geometry.length), geometry.nx)
    base = 1.0 - (r[:, None] / radius) ** 2
    shape = (geometry.nx, geometry.nr, geometry.ntheta)
    bent_u = np.broadcast_to(base * (1.0 + 0.05 * np.cos(theta)), shape)
    bundle = SimpleNamespace(
        u=bent_u,
        x=x,
        y=r,
        z=theta,
        field_scale=np.linspace(0.2, 1.0, geometry.nx),
        mean_velocity=np.mean(bent_u, axis=(1, 2)),
        charge_balance_residual=np.geomspace(1.0e-4, 1.0e-7, geometry.nx),
    )
    bent_solution = SimpleNamespace(problem=bent_problem, bundle=bundle)
    straight_solution = SimpleNamespace(
        bundle=SimpleNamespace(**{**vars(bundle), "u": np.broadcast_to(base, bent_u.shape)})
    )

    outputs = write_bent_pipe_overview_plots(
        bent_solution, tmp_path, straight_solution=straight_solution
    )
    _assert_figure_pair(outputs, tmp_path, "bent_pipe_overview")


def test_write_magnetic_obstacle_schematic_plots_writes_png_and_pdf(tmp_path: Path):
    problem = build_magnetic_obstacle_rect_extruded_problem(
        base_bz=4.0, ny=6, nz=6, nx_stations=5
    )
    problem = replace(
        problem,
        case=replace(
            problem.case,
            time_stepper=replace(
                problem.case.time_stepper, max_steps=3, potential_iterations=8
            ),
            solver=replace(problem.case.solver, coupling_iterations=3),
        ),
    )
    solution = solve_extruded_inductionless(problem)
    reference_problem = replace(
        problem,
        profile=replace(
            problem.profile, field_scale=jnp.zeros_like(problem.profile.field_scale)
        ),
    )
    reference_solution = solve_extruded_inductionless(reference_problem)
    outputs = write_magnetic_obstacle_schematic_plots(
        solution, reference_solution, tmp_path
    )
    _assert_figure_pair(outputs, tmp_path, "magnetic_obstacle_schematic")

    benchmark_outputs = write_magnetic_obstacle_benchmark_plots(
        solution, reference_solution, tmp_path, case_title="Magnetic obstacle"
    )
    assert all(path.exists() for path in benchmark_outputs)


def test_write_wham_mirror_overview_plots_writes_png_and_pdf(tmp_path: Path):
    x = np.linspace(-0.4, 0.4, 9)
    y = np.linspace(-0.2, 0.2, 7)
    z = np.linspace(-0.5, 0.5, 11)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    bx = np.zeros_like(xx)
    by = 0.2 * yy
    bz = 1.0 + 0.4 * np.exp(-((zz / 0.25) ** 2))
    table_path = write_tabulated_field_npz(
        tmp_path / "wham_test_field.npz", x=x, y=y, z=z, bx=bx, by=by, bz=bz
    )

    bundle = SimpleNamespace(
        x=np.linspace(-0.4, 0.4, 5),
        field_scale=np.asarray([0.3, 0.7, 1.0, 0.7, 0.3]),
        mean_velocity=np.asarray([1.0, 0.98, 0.96, 0.98, 1.0]),
        current_scaled_pressure_proxy=np.asarray([0.1, 0.2, 0.35, 0.2, 0.1]),
        p=np.linspace(0.0, 0.2, 5)[:, None, None] + np.zeros((5, 4, 12)),
        u=np.broadcast_to(np.linspace(0.6, 1.0, 4)[:, None], (5, 4, 12)),
    )
    solution = SimpleNamespace(bundle=bundle)
    autodiff_summary = {
        "reference_separation": 1.96,
        "separation_sweep": [1.6, 1.8, 2.0, 2.2],
        "pressure_drop_curve": [3.1, 3.4, 3.6, 3.7],
        "sensitivity_curve": [0.7, 0.4, 0.2, 0.0],
    }

    outputs = write_wham_mirror_overview_plots(
        solution,
        table_path=table_path,
        pipe_radius=0.2,
        coil_separation=1.0,
        out_dir=tmp_path,
        case_title="WHAM overview",
        autodiff_summary=autodiff_summary,
    )
    _assert_figure_pair(outputs, tmp_path, "wham_mirror_overview")


def test_write_magnetic_obstacle_regime_plots_writes_png_and_pdf(tmp_path: Path):
    records = [
        {
            "base_bz": 20.0,
            "forcing": 0.5,
            "peak_velocity_deficit_ratio": 1.0e-2,
            "pressure_excess_proxy": 2.0e-2,
            "current_proxy_peak": 5.0e-1,
            "y_l2_distortion": 1.5e-1,
            "z_l2_distortion": 1.3e-1,
        },
        {
            "base_bz": 20.0,
            "forcing": 1.0,
            "peak_velocity_deficit_ratio": 2.0e-2,
            "pressure_excess_proxy": 3.0e-2,
            "current_proxy_peak": 7.0e-1,
            "y_l2_distortion": 1.8e-1,
            "z_l2_distortion": 1.6e-1,
        },
        {
            "base_bz": 40.0,
            "forcing": 0.5,
            "peak_velocity_deficit_ratio": 3.0e-2,
            "pressure_excess_proxy": 5.0e-2,
            "current_proxy_peak": 1.2,
            "y_l2_distortion": 2.0e-1,
            "z_l2_distortion": 1.9e-1,
        },
        {
            "base_bz": 40.0,
            "forcing": 1.0,
            "peak_velocity_deficit_ratio": 4.0e-2,
            "pressure_excess_proxy": 7.0e-2,
            "current_proxy_peak": 1.7,
            "y_l2_distortion": 2.4e-1,
            "z_l2_distortion": 2.2e-1,
        },
    ]
    outputs = write_magnetic_obstacle_regime_plots(
        records, tmp_path, case_title="Obstacle regime scan"
    )
    _assert_figure_pair(outputs, tmp_path, "magnetic_obstacle_regime_scan")
