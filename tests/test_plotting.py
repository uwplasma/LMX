from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from lmx.cases import make_hartmann_case, make_hunt_case
from lmx.core import Diagnostics, MHDState, Solution
from lmx.plotting import (
    _movie_field_stack,
    _plot_field,
    write_autodiff_plots,
    write_case_overview_plots,
    write_strong_scaling_plots,
    write_transient_movies,
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
    y, z = np.meshgrid(np.asarray(mesh.y_centers), np.asarray(mesh.z_centers), indexing="ij")
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
    assert (tmp_path / "overview.png").exists()
    assert (tmp_path / "overview.pdf").exists()
    assert (tmp_path / "diagnostics.png").exists()
    assert (tmp_path / "diagnostics.pdf").exists()
    assert len(outputs) == 4


def test_write_case_overview_plots_skips_diagnostics_when_no_time_history(tmp_path: Path):
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
    assert outputs == [tmp_path / "overview.png", tmp_path / "overview.pdf"]
    assert not (tmp_path / "diagnostics.png").exists()


def test_plot_field_handles_negative_only_field():
    solution = _sample_solution(make_hartmann_case(ha=5.0, ny=8, nz=8))
    fig, ax = plt.subplots()
    try:
        _plot_field(ax, solution, -jnp.abs(solution.state.u), title="Negative field", cmap="RdBu_r")
        assert ax.get_title() == "Negative field"
    finally:
        plt.close(fig)


def test_movie_field_stack_supports_raw_and_bulk_deviation():
    frames = _sample_frames(make_hunt_case(ha=20.0, ny=4, nz=4, wall_cells=1), signed=False)
    raw_fields, raw_peaks, raw_label, raw_colorbar = _movie_field_stack(frames, field_mode="raw")
    dev_fields, dev_peaks, dev_label, dev_colorbar = _movie_field_stack(frames, field_mode="bulk_deviation")

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


@pytest.mark.filterwarnings("ignore:Animation was deleted without rendering anything:UserWarning")
def test_write_transient_movies_writes_posters_and_stubbed_gifs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    frames = _sample_frames(make_hunt_case(ha=20.0, ny=4, nz=4, wall_cells=1), signed=True)

    def fake_save(self, filename, *args, **kwargs):
        Path(filename).write_bytes(b"gif")

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
    for path in expected:
        assert path.exists()


@pytest.mark.filterwarnings("ignore:Animation was deleted without rendering anything:UserWarning")
def test_write_transient_movies_handles_positive_only_normalized_path_and_contour_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    frames = _sample_frames(make_hunt_case(ha=20.0, ny=4, nz=4, wall_cells=1), signed=False)

    def fake_save(self, filename, *args, **kwargs):
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
        {"platform": "GPU", "num_devices": 1, "mean_seconds": 1.8},
        {"platform": "GPU", "num_devices": 2, "mean_seconds": 1.0},
    ]
    outputs = write_strong_scaling_plots(records, tmp_path, case_title="Scaling")

    assert outputs == [tmp_path / "strong_scaling.png", tmp_path / "strong_scaling.pdf"]
    assert outputs[0].exists()
    assert outputs[1].exists()


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

    assert outputs == [tmp_path / "autodiff_summary.png", tmp_path / "autodiff_summary.pdf"]
    assert outputs[0].exists()
    assert outputs[1].exists()
