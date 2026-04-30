from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from lmx.q2d import (
    build_q2d_decay_case,
    build_q2d_forced_case,
    build_q2d_turbulence_decay_case,
    build_q2d_wall_bounded_forced_case,
    build_q2d_wall_driven_cavity_case,
    compare_q2d_wall_driven_observables,
    q2d_energy_spectrum,
    q2d_modal_energy_budget,
    q2d_turbulence_observables,
    q2d_turbulence_readiness_metrics,
    q2d_wall_driven_cavity_observables,
    q2d_wall_driven_cavity_observables_on_grid,
    solve_q2d_decay,
    solve_q2d_forced,
    solve_q2d_turbulence_decay,
    solve_q2d_wall_bounded_forced,
    solve_q2d_wall_driven_cavity,
    validate_q2d_decay_solution,
    validate_q2d_decay_energy_budget,
    validate_q2d_forced_solution,
    validate_q2d_forced_energy_budget,
    validate_q2d_turbulence_decay_observables,
    validate_q2d_wall_bounded_forced_solution,
    validate_q2d_wall_bounded_energy_budget,
    write_q2d_decay_plots,
    write_q2d_forced_plots,
    write_q2d_turbulence_decay_movie,
    write_q2d_turbulence_observable_plots,
    write_q2d_wall_bounded_forced_plots,
    write_q2d_wall_driven_comparison_plots,
)


pytestmark = pytest.mark.unit


def test_q2d_decay_validation_passes_on_baseline_case():
    case = build_q2d_decay_case(nx=48, ny=48, dt=2.5e-4, t_final=0.04)
    solution = solve_q2d_decay(case)
    validation = validate_q2d_decay_solution(case, solution)
    energy_budget = validate_q2d_decay_energy_budget(case, solution)
    assert validation["l2_error"] < 5.0e-2
    assert validation["validation_pass"] is True
    assert energy_budget["relative_budget_l2"] < 6.0e-2
    assert energy_budget["validation_pass"] is True


def test_write_q2d_decay_plots_writes_png_and_pdf(tmp_path: Path):
    case = build_q2d_decay_case(nx=32, ny=32, dt=5.0e-4, t_final=0.02)
    solution = solve_q2d_decay(case)
    outputs = write_q2d_decay_plots(case, solution, tmp_path)
    assert outputs == [tmp_path / "q2d_decay_overview.png", tmp_path / "q2d_decay_overview.pdf"]
    assert outputs[0].exists()
    assert outputs[1].exists()


def test_q2d_forced_validation_passes_on_baseline_case():
    case = build_q2d_forced_case(nx=48, ny=48, dt=2.5e-4, t_final=0.08)
    solution = solve_q2d_forced(case)
    validation = validate_q2d_forced_solution(case, solution)
    energy_budget = validate_q2d_forced_energy_budget(case, solution)
    assert validation["l2_error"] < 7.0e-2
    assert validation["validation_pass"] is True
    assert energy_budget["relative_budget_l2"] < 6.0e-2
    assert energy_budget["validation_pass"] is True


def test_write_q2d_forced_plots_writes_png_and_pdf(tmp_path: Path):
    case = build_q2d_forced_case(nx=32, ny=32, dt=5.0e-4, t_final=0.04)
    solution = solve_q2d_forced(case)
    outputs = write_q2d_forced_plots(case, solution, tmp_path)
    assert outputs == [tmp_path / "q2d_forced_overview.png", tmp_path / "q2d_forced_overview.pdf"]
    assert outputs[0].exists()
    assert outputs[1].exists()


def test_q2d_wall_bounded_validation_passes_on_baseline_case():
    case = build_q2d_wall_bounded_forced_case(nx=48, ny=48, dt=2.5e-4, t_final=0.08)
    solution = solve_q2d_wall_bounded_forced(case)
    validation = validate_q2d_wall_bounded_forced_solution(case, solution)
    energy_budget = validate_q2d_wall_bounded_energy_budget(case, solution)
    assert validation["l2_error"] < 8.0e-2
    assert validation["validation_pass"] is True
    assert energy_budget["relative_budget_l2"] < 6.0e-2
    assert energy_budget["validation_pass"] is True


def test_write_q2d_wall_bounded_plots_writes_png_and_pdf(tmp_path: Path):
    case = build_q2d_wall_bounded_forced_case(nx=32, ny=32, dt=5.0e-4, t_final=0.04)
    solution = solve_q2d_wall_bounded_forced(case)
    outputs = write_q2d_wall_bounded_forced_plots(case, solution, tmp_path)
    assert outputs == [tmp_path / "q2d_wall_bounded_overview.png", tmp_path / "q2d_wall_bounded_overview.pdf"]
    assert outputs[0].exists()
    assert outputs[1].exists()


def test_write_q2d_turbulence_observable_plots_writes_png_and_pdf(tmp_path: Path):
    case = build_q2d_wall_bounded_forced_case(nx=32, ny=32, dt=5.0e-4, t_final=0.04)
    solution = solve_q2d_wall_bounded_forced(case)
    outputs = write_q2d_turbulence_observable_plots(
        solution.field,
        tmp_path,
        lx=case.lx,
        ly=case.ly,
        viscosity=case.viscosity,
        hartmann_friction=case.hartmann_friction,
    )
    assert outputs == [tmp_path / "q2d_turbulence_observables.png", tmp_path / "q2d_turbulence_observables.pdf"]
    assert outputs[0].exists()
    assert outputs[1].exists()


def test_q2d_turbulence_decay_movie_and_observable_gate(tmp_path: Path):
    case = build_q2d_turbulence_decay_case(nx=24, ny=24, dt=2.0e-3, t_final=1.0, frame_count=8)
    solution = solve_q2d_turbulence_decay(case)
    validation = validate_q2d_turbulence_decay_observables(case, solution)
    assert solution.frames.shape[0] == 8
    assert validation["validation_pass"] is True
    assert validation["turnover_count"] > 0.02
    assert validation["max_courant"] < 0.45
    assert validation["research_grade_turbulence_validation_pass"] is False
    outputs = write_q2d_turbulence_decay_movie(solution, tmp_path, fps=4)
    assert outputs == [tmp_path / "q2d_turbulence_decay.gif", tmp_path / "q2d_turbulence_decay_poster.png"]
    assert outputs[0].exists()
    assert outputs[1].exists()


def test_q2d_wall_driven_cavity_comparison_gate_and_plot(tmp_path: Path):
    case = build_q2d_wall_driven_cavity_case(nx=25, ny=17, dt=1.0e-3, t_final=0.02, frame_count=5)
    solution = solve_q2d_wall_driven_cavity(case)
    observables = q2d_wall_driven_cavity_observables(case, solution)
    reference = {
        "speed_mean": observables["speed_mean"] * 1.01,
        "speed_rms": observables["speed_rms"] * 0.99,
        "uy_mean": observables["uy_mean"] * 1.02,
        "vorticity_peak": observables["vorticity_peak"] * 0.98,
    }

    comparison = compare_q2d_wall_driven_observables(observables, reference, relative_tolerance=0.05)
    outputs = write_q2d_wall_driven_comparison_plots(case, solution, comparison, tmp_path)

    assert observables["validation_pass"] is True
    assert comparison["matched_parity"] is True
    assert comparison["passed_observable_count"] == 4
    assert outputs == [tmp_path / "q2d_lmx_q2dmhdfoam_lid_driven_parity.png", tmp_path / "q2d_lmx_q2dmhdfoam_lid_driven_parity.pdf"]
    assert outputs[0].exists()
    assert outputs[1].exists()


def test_q2d_wall_driven_cavity_external_grid_observables():
    case = build_q2d_wall_driven_cavity_case(nx=25, ny=17, dt=1.0e-3, t_final=0.02, frame_count=5)
    solution = solve_q2d_wall_driven_cavity(case)

    observables = q2d_wall_driven_cavity_observables_on_grid(
        case,
        solution,
        x=np.linspace(0.002, 0.038, 9),
        y=np.asarray([-0.018, -0.008, 0.004, 0.016]),
        y_widths=np.asarray([0.004, 0.012, 0.012, 0.012]),
    )
    comparison = compare_q2d_wall_driven_observables(
        observables,
        {
            "speed_mean": observables["speed_mean"],
            "speed_rms": observables["speed_rms"],
            "vorticity_peak": observables["vorticity_peak"],
        },
        relative_tolerance=1.0e-12,
        observable_keys=("speed_mean", "speed_rms", "vorticity_peak"),
    )

    assert observables["sample_count"] == 36
    assert observables["validation_pass"] is True
    assert observables["sampling"] == "interpolated_external_cell_grid"
    assert comparison["matched_parity"] is True
    assert comparison["passed_observable_count"] == 3


def test_q2d_energy_spectrum_identifies_single_mode():
    x = np.linspace(0.0, 2.0, 64, endpoint=False)
    y = np.linspace(0.0, 2.0, 64, endpoint=False)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    field = np.sin(2.0 * np.pi * xx / 2.0) * np.sin(2.0 * np.pi * yy / 2.0)

    spectrum = q2d_energy_spectrum(field, lx=2.0, ly=2.0, bins=12)

    assert len(spectrum["wavenumber"]) == 12
    assert len(spectrum["counts"]) == 12
    assert max(spectrum["energy"]) > 0.0
    assert sum(spectrum["counts"]) == field.size


def test_q2d_turbulence_observables_report_sommeria_moreau_readiness():
    case = build_q2d_wall_bounded_forced_case(nx=40, ny=40, dt=5.0e-4, t_final=0.04)
    solution = solve_q2d_wall_bounded_forced(case)

    metrics = q2d_turbulence_observables(
        solution.field,
        lx=case.lx,
        ly=case.ly,
        viscosity=case.viscosity,
        hartmann_friction=case.hartmann_friction,
    )

    assert metrics["kinetic_energy"] > 0.0
    assert metrics["fluctuation_kinetic_energy"] > 0.0
    assert metrics["enstrophy_proxy"] > 0.0
    assert metrics["spectrum_peak_wavenumber"] > 0.0
    assert np.isfinite(metrics["spectrum_log_slope"])
    assert 0.0 <= metrics["high_wavenumber_energy_fraction"] <= 1.0
    assert metrics["validation_status"] == "spectral_observables_available_no_turbulent_reference"
    assert metrics["research_grade_turbulence_validation_pass"] is False


def test_q2d_turbulence_metrics_reject_non_2d_fields():
    with pytest.raises(ValueError, match="2D field"):
        q2d_energy_spectrum(np.zeros((4, 4, 2)), lx=1.0, ly=1.0)
    with pytest.raises(ValueError, match="2D field"):
        q2d_turbulence_readiness_metrics(
            np.zeros((4, 4, 2)),
            lx=1.0,
            ly=1.0,
            viscosity=0.01,
            hartmann_friction=2.0,
        )


def test_q2d_modal_energy_budget_rejects_malformed_inputs():
    with pytest.raises(ValueError, match="matching 1D"):
        q2d_modal_energy_budget(
            time=np.zeros((2, 2)),
            amplitude=np.zeros(4),
            decay_rate=1.0,
            mode_mean_square=0.25,
        )
    with pytest.raises(ValueError, match="at least three"):
        q2d_modal_energy_budget(
            time=np.asarray([0.0, 1.0]),
            amplitude=np.asarray([1.0, 0.5]),
            decay_rate=1.0,
            mode_mean_square=0.25,
        )
    with pytest.raises(ValueError, match="strictly increasing"):
        q2d_modal_energy_budget(
            time=np.asarray([0.0, 1.0, 1.0]),
            amplitude=np.asarray([1.0, 0.8, 0.7]),
            decay_rate=1.0,
            mode_mean_square=0.25,
        )
