from pathlib import Path

import jax.numpy as jnp
import pytest

from lmx.cases import make_hartmann_case, make_shercliff_case
from lmx.core import Diagnostics, MHDState, Solution
from lmx.mesh import generate_rect_duct_mesh
from lmx.solvers import solve_steady
from lmx.validation import (
    closed_channel_validation,
    compare_with_freemhd,
    duct_profile_metrics,
    hartmann_analytic_profile,
    hartmann_validation,
    processed_slice_validation,
    write_analytic_comparison,
    write_closed_channel_validation,
    write_metrics_json,
    write_processed_slice_validation,
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


def test_hartmann_ha20_validation_error_is_bounded():
    case = make_hartmann_case(ha=20.0, ny=16, nz=16)
    solution = solve_steady(case)
    comparison = hartmann_validation(solution, ha=20.0)
    assert comparison.l2_error < 0.05


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


def test_processed_slice_validation_writer(tmp_path: Path):
    mesh = generate_rect_duct_mesh(width=2.0, height=2.0, ny=5, nz=5)
    y, z = jnp.meshgrid(mesh.y_centers, mesh.z_centers, indexing="ij")
    u = 1.0 - 0.2 * y**2 - 0.3 * z**2
    zeros = jnp.zeros_like(u)
    solution = Solution(
        mesh=mesh,
        state=MHDState(
            u=u,
            phi=zeros,
            jy=zeros,
            jz=zeros,
            lorentz_x=zeros,
            time=0.0,
            residual=0.0,
        ),
        diagnostics=Diagnostics(
            residual_history=jnp.asarray([0.0]),
            courant_like=jnp.asarray([0.0]),
            ohmic_power=jnp.asarray([0.0]),
        ),
        case_name="shercliff_ha2",
    )
    closed_channel_root = tmp_path / "ClosedChannel"
    closed_channel_root.mkdir(parents=True)
    center_y = solution.state.u[:, solution.state.u.shape[1] // 2]
    center_z = solution.state.u[solution.state.u.shape[0] // 2, :]
    rows = ["Points:1,Points:2,U:0,potE"]
    for y_coord, value in zip(mesh.y_centers.tolist(), center_y.tolist()):
        rows.append(f"{y_coord},0.0,{value},0.0")
    for z_coord, value in zip(mesh.z_centers.tolist(), center_z.tolist()):
        if abs(z_coord) < 1e-12:
            continue
        rows.append(f"0.0,{z_coord},{value},0.0")
    (closed_channel_root / "shercliff_Ha2_XSlice1m_4s.csv").write_text("\n".join(rows))
    report = processed_slice_validation(solution, "shercliff", 2, reference_root=closed_channel_root)
    path = write_processed_slice_validation(report, tmp_path / "processed_slice_validation.json")
    assert report.y_profile.l2_error < 1e-12
    assert report.z_profile.l2_error < 1e-12
    assert path.exists()
