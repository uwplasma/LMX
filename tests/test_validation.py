from pathlib import Path

import jax.numpy as jnp
import pytest

from lmx.cases import make_hartmann_case, make_shercliff_case
from lmx.solvers import solve_steady
from lmx.validation import (
    closed_channel_validation,
    duct_profile_metrics,
    compare_with_freemhd,
    hartmann_analytic_profile,
    hartmann_validation,
    write_analytic_comparison,
    write_closed_channel_validation,
    write_metrics_json,
    write_validation_report,
)


pytestmark = pytest.mark.validation


def test_hartmann_profile_center_is_maximum():
    y = jnp.linspace(-1.0, 1.0, 101)
    profile = hartmann_analytic_profile(y, ha=10.0)
    assert float(profile[50]) >= float(profile[0])


def test_compare_with_freemhd_report(tmp_path: Path):
    case = make_hartmann_case()
    report = compare_with_freemhd(case, tmp_path)
    path = write_validation_report(report, tmp_path / "report.json")
    assert path.exists()


def test_hartmann_validation_writer(tmp_path: Path):
    case = make_hartmann_case(ha=5.0, ny=16, nz=16)
    solution = solve_steady(case)
    comparison = hartmann_validation(solution, ha=5.0)
    path = write_analytic_comparison(comparison, tmp_path / "analytic.json", axis_name="y")
    assert path.exists()


def test_duct_profile_metrics_writer(tmp_path: Path):
    case = make_hartmann_case(ha=5.0, ny=16, nz=16)
    solution = solve_steady(case)
    metrics = duct_profile_metrics(solution)
    path = write_metrics_json(metrics, tmp_path / "metrics.json")
    assert path.exists()


def test_closed_channel_validation_writer(tmp_path: Path):
    analytical_root = tmp_path / "ClosedChannel" / "AnalyticalSolutions"
    analytical_root.mkdir(parents=True)
    (analytical_root / "Shercliff_Analytical_Ha2_PresDrop1.0.txt").write_text(
        "r\tu1\tu2\n-1.0\t0.0\t0.0\n0.0\t1.0\t1.0\n1.0\t0.0\t0.0\n"
    )
    case = make_shercliff_case(ha=2.0, ny=12, nz=12)
    solution = solve_steady(case)
    comparison = closed_channel_validation(solution, "shercliff", 2, reference_root=tmp_path / "ClosedChannel")
    path = write_closed_channel_validation(comparison, tmp_path / "closed_channel_validation.json")
    assert path.exists()
