from __future__ import annotations

from pathlib import Path

import pytest

from lmx.q2d import (
    build_q2d_decay_case,
    build_q2d_forced_case,
    build_q2d_wall_bounded_forced_case,
    solve_q2d_decay,
    solve_q2d_forced,
    solve_q2d_wall_bounded_forced,
    validate_q2d_decay_solution,
    validate_q2d_forced_solution,
    validate_q2d_wall_bounded_forced_solution,
    write_q2d_decay_plots,
    write_q2d_forced_plots,
    write_q2d_wall_bounded_forced_plots,
)


pytestmark = pytest.mark.unit


def test_q2d_decay_validation_passes_on_baseline_case():
    case = build_q2d_decay_case(nx=48, ny=48, dt=2.5e-4, t_final=0.04)
    solution = solve_q2d_decay(case)
    validation = validate_q2d_decay_solution(case, solution)
    assert validation["l2_error"] < 5.0e-2
    assert validation["validation_pass"] is True


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
    assert validation["l2_error"] < 7.0e-2
    assert validation["validation_pass"] is True


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
    assert validation["l2_error"] < 8.0e-2
    assert validation["validation_pass"] is True


def test_write_q2d_wall_bounded_plots_writes_png_and_pdf(tmp_path: Path):
    case = build_q2d_wall_bounded_forced_case(nx=32, ny=32, dt=5.0e-4, t_final=0.04)
    solution = solve_q2d_wall_bounded_forced(case)
    outputs = write_q2d_wall_bounded_forced_plots(case, solution, tmp_path)
    assert outputs == [tmp_path / "q2d_wall_bounded_overview.png", tmp_path / "q2d_wall_bounded_overview.pdf"]
    assert outputs[0].exists()
    assert outputs[1].exists()
