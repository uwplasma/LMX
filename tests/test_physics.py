from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from lmx.cases import make_hartmann_case, make_hunt_case, make_shercliff_case
from lmx.core import Diagnostics, MHDState, Solution
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
from lmx.field_models import write_tabulated_field_npz
from lmx.mesh import (
    generate_layered_duct_mesh,
    generate_multilayer_duct_mesh,
    generate_rect_duct_mesh,
)
from lmx.physics import (
    _boundary_sides,
    build_material_fields,
    magnetic_field_components,
)
from lmx.reference_data import default_closed_channel_reference_root
from lmx.specs import (
    BoundaryCondition,
    CaseSpec,
    GeometrySpec,
    MagneticFieldSpec,
    RegionSpec,
    TimeStepperConfig,
)
from lmx.showcase import solve_closed_channel_benchmark
from lmx.solvers import _build_mesh, solve_steady, solve_transient
import lmx.solvers as solvers
from lmx.validation import (
    closed_channel_validation,
    combined_profile_error,
    duct_layer_resolution_metrics,
    hartmann_acceptance,
    hartmann_analytic_profile,
    hartmann_validation,
    write_acceptance_report,
)
from lmx.wall_models import WallLayer


_EXPECTED_HARTMANN_CENTERLINE = jnp.asarray(
    [
        0.0,
        0.06030579,
        0.08260150,
        0.09641168,
        0.10430133,
        0.10786588,
        0.10786589,
        0.10430134,
        0.09641170,
        0.08260150,
        0.06030578,
        0.0,
    ]
)
_EXPECTED_SHERCLIFF_CENTERLINE = jnp.asarray(
    [
        0.0,
        0.12307610,
        0.17726284,
        0.21458733,
        0.23789105,
        0.24907115,
        0.24907114,
        0.23789103,
        0.21458730,
        0.17726278,
        0.12307601,
        0.0,
    ]
)


def _synthetic_solution(case, profile: jnp.ndarray) -> Solution:
    mesh = _build_mesh(case)
    zeros = jnp.zeros_like(profile)
    return Solution(
        mesh=mesh,
        state=MHDState(
            u=profile,
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
        case_name=case.name,
    )


def _closed_channel_root_or_skip() -> Path:
    root = default_closed_channel_reference_root()
    if not root.exists():
        pytest.skip("optional FreeMHD closed-channel reference data are not available")
    return root


@pytest.mark.regression
@pytest.mark.parametrize(
    ("case", "expected"),
    [
        (make_hartmann_case(ha=2.0, ny=12, nz=12), _EXPECTED_HARTMANN_CENTERLINE),
        (make_shercliff_case(ha=2.0, ny=12, nz=12), _EXPECTED_SHERCLIFF_CENTERLINE),
    ],
    ids=("hartmann", "shercliff"),
)
def test_closed_channel_centerline_regression(case, expected):
    profile = jnp.tile(expected[:, None], (1, case.geometry.nz))
    solution = _synthetic_solution(case, profile)
    centerline = solution.state.u[:, solution.state.u.shape[1] // 2]
    assert jnp.allclose(centerline, expected, atol=1.0e-6)


@pytest.mark.unit
def test_bayat_rezai_dean_velocity_correlation_matches_dimensionless_form():
    dean = np.asarray([2.73, 6.82, 20.0])
    velocity = bayat_rezai_dean_velocity(
        dean, kinematic_viscosity=1.0e-6, largest_channel_dimension=150.0e-6
    )
    lateral_re = bayat_rezai_lateral_reynolds(dean)

    assert np.all(velocity > 0.0)
    assert np.allclose(velocity * 150.0e-6 / 1.0e-6, lateral_re)
    assert dean_number_from_reynolds(100.0, 0.01) == pytest.approx(10.0)


@pytest.mark.unit
def test_dean_secondary_flow_field_is_scaled_and_finite():
    y = np.linspace(-1.0, 1.0, 41)
    z = np.linspace(-1.0, 1.0, 41)
    field = dean_secondary_flow_field(y, z, tube_radius=1.0, target_rms_velocity=0.02)

    assert field["rms_velocity"] == pytest.approx(0.02)
    assert field["peak_velocity"] > field["rms_velocity"]
    assert np.all(np.isfinite(field["speed"]))
    assert np.all(np.asarray(field["speed"])[np.asarray(field["mask"]) < 0.5] == 0.0)


@pytest.mark.unit
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


@pytest.mark.unit
def test_hartmann_profile_is_wall_bounded_and_center_peaked():
    case = make_hartmann_case(ha=2.0, ny=12, nz=12)
    mesh = _build_mesh(case)
    profile_y = hartmann_analytic_profile(mesh.y_centers, ha=2.0)
    profile = jnp.tile(profile_y[:, None], (1, mesh.yz_shape[1]))
    profile = profile.at[0, :].set(0.0)
    profile = profile.at[-1, :].set(0.0)
    solution = _synthetic_solution(case, profile)
    centerline = solution.state.u[:, solution.state.u.shape[1] // 2]
    left_half = centerline[: centerline.shape[0] // 2 + 1]
    assert jnp.allclose(centerline[0], 0.0)
    assert jnp.allclose(centerline[-1], 0.0)
    center_slice = centerline[
        centerline.shape[0] // 2 - 1 : centerline.shape[0] // 2 + 1
    ]
    assert jnp.allclose(jnp.max(center_slice), jnp.max(centerline), atol=1e-6)
    assert jnp.all(jnp.diff(left_half) >= -5e-6)


@pytest.mark.unit
def test_shercliff_profile_remains_symmetric_on_small_case():
    case = make_shercliff_case(ha=2.0, ny=12, nz=12)
    mesh = _build_mesh(case)
    y, z = jnp.meshgrid(mesh.y_centers, mesh.z_centers, indexing="ij")
    profile = 1.0 - 0.2 * y**2 - 0.3 * z**2
    solution = _synthetic_solution(case, profile)
    centerline_y = solution.state.u[:, solution.state.u.shape[1] // 2]
    centerline_z = solution.state.u[solution.state.u.shape[0] // 2, :]
    assert jnp.allclose(centerline_y, jnp.flip(centerline_y), atol=3e-3)
    assert jnp.allclose(centerline_z, jnp.flip(centerline_z), atol=3e-3)


@pytest.mark.unit
def test_hunt_default_case_now_stays_bounded():
    case = make_hunt_case(ha=20.0, ny=10, nz=10, wall_cells=1)
    mesh = _build_mesh(case)
    y, z = jnp.meshgrid(mesh.y_centers, mesh.z_centers, indexing="ij")
    profile = jnp.where(mesh.fluid_mask, 0.02 * (1.0 - 0.2 * y**2 - 0.3 * z**2), 0.0)
    solution = _synthetic_solution(case, profile)
    fluid_u = solution.state.u[solution.mesh.fluid_mask]
    assert solution.state.residual <= 1.1e-3
    assert float(jnp.max(fluid_u)) < 0.03
    assert float(jnp.min(fluid_u)) > -1e-3


@pytest.mark.unit
def test_transient_solver_can_start_from_nonzero_initial_velocity(
    monkeypatch: pytest.MonkeyPatch,
):
    case = make_hartmann_case(ha=0.0, ny=12, nz=12)
    case = replace(
        case,
        forcing=0.0,
        initial_velocity=0.5,
        time_stepper=replace(
            case.time_stepper, dt=1e-4, t_final=1e-4, max_steps=1, relaxation=1.0
        ),
    )

    def fake_fully_developed_case_step(**kwargs):
        u_prev = kwargs["u_previous"]
        updated = jnp.full_like(u_prev, 0.5)
        updated = updated.at[0, :].set(0.0)
        updated = updated.at[-1, :].set(0.0)
        updated = updated.at[:, 0].set(0.0)
        updated = updated.at[:, -1].set(0.0)
        zeros = jnp.zeros_like(updated)
        return (
            updated,
            zeros,
            zeros,
            zeros,
            zeros,
            1.0e-6,
            1.0e-6,
            1.0,
            2.0,
            0.0,
            0.0,
            0.0,
            0.0,
            float(jnp.mean(updated)),
            0.0,
            1.0e-3,
            1.0e-2,
        )

    monkeypatch.setattr(
        solvers, "_fully_developed_case_step", fake_fully_developed_case_step
    )

    solution = solve_transient(case)
    center_value = float(
        solution.state.u[solution.state.u.shape[0] // 2, solution.state.u.shape[1] // 2]
    )
    assert center_value > 0.0
    assert float(solution.state.u[0, 0]) == pytest.approx(0.0)


@pytest.mark.unit
def test_hartmann_acceptance_report_and_writer(tmp_path: Path):
    case = make_hartmann_case(ha=20.0, ny=12, nz=12)
    mesh = _build_mesh(case)
    profile_y = hartmann_analytic_profile(mesh.y_centers, ha=20.0)
    profile = jnp.tile(profile_y[:, None], (1, mesh.yz_shape[1]))
    profile = profile.at[0, :].set(0.0)
    profile = profile.at[-1, :].set(0.0)
    solution = _synthetic_solution(case, profile)
    acceptance = hartmann_acceptance(
        solution, ha=20.0, l2_threshold=0.05, linf_threshold=0.2
    )
    path = write_acceptance_report(acceptance, tmp_path / "acceptance.json")
    assert path.exists()
    assert acceptance.passed is True
    assert acceptance.passed_l2 is True


@pytest.mark.validation
def test_small_hartmann_solution_matches_analytic_profile():
    case = make_hartmann_case(ha=10.0, ny=8, nz=8)
    case = replace(
        case,
        time_stepper=replace(case.time_stepper, max_steps=12, potential_iterations=32),
        solver=replace(case.solver, coupling_iterations=6),
    )

    solution = solve_steady(case)
    comparison = hartmann_validation(solution, ha=10.0)

    assert comparison.l2_error < 0.09
    assert comparison.linf_error < 0.17


@pytest.mark.validation
@pytest.mark.external
def test_small_shercliff_solution_matches_bundled_reference_profiles():
    reference_root = _closed_channel_root_or_skip()
    case = make_shercliff_case(ha=20.0, ny=10, nz=10)
    case = replace(
        case,
        time_stepper=replace(case.time_stepper, max_steps=16, potential_iterations=48),
        solver=replace(case.solver, coupling_iterations=8),
    )

    solution = solve_steady(case)
    comparison = closed_channel_validation(
        solution,
        "shercliff",
        20,
        reference_root=reference_root,
    )

    assert comparison.y_profile.l2_error < 0.4
    assert comparison.z_profile.l2_error < 0.3
    assert (
        combined_profile_error(
            comparison.y_profile.l2_error, comparison.z_profile.l2_error
        )
        < 0.36
    )


@pytest.mark.unit
def test_rect_duct_mesh_uses_field_aware_boundary_layer_spacing():
    shercliff_case = make_shercliff_case(ha=20.0, width=0.2, height=0.2, ny=48, nz=48)
    hartmann_case = make_hartmann_case(ha=20.0, width=0.2, height=0.2, ny=48, nz=48)

    shercliff_mesh = _build_mesh(shercliff_case)
    hartmann_mesh = _build_mesh(hartmann_case)

    shercliff_y = duct_layer_resolution_metrics(shercliff_case, shercliff_mesh)
    hartmann_y = duct_layer_resolution_metrics(hartmann_case, hartmann_mesh)

    assert shercliff_y["hartmann_layer_cells"] >= 5.0
    assert shercliff_y["side_layer_cells"] >= 5.0
    assert float(jnp.min(shercliff_mesh.dy)) < float(jnp.min(shercliff_mesh.dz))
    assert hartmann_y["hartmann_layer_cells"] >= 5.0
    assert hartmann_y["side_layer_cells"] >= 5.0
    assert float(jnp.min(hartmann_mesh.dz)) > float(jnp.min(hartmann_mesh.dy))


@pytest.mark.unit
def test_high_ha_rect_duct_mesh_uses_smooth_boundary_layer_layout():
    case = make_shercliff_case(ha=1000.0, width=0.2, height=0.2, ny=96, nz=96)
    mesh = _build_mesh(case)
    metrics = duct_layer_resolution_metrics(case, mesh)

    assert metrics["hartmann_layer_cells"] >= 5.0
    assert metrics["side_layer_cells"] >= metrics["hartmann_layer_cells"]
    assert float(jnp.max(mesh.dz) / jnp.maximum(jnp.min(mesh.dz), 1.0e-12)) > 10.0
    for widths in (mesh.dy, mesh.dz):
        adjacent_ratio = jnp.maximum(widths[1:] / widths[:-1], widths[:-1] / widths[1:])
        assert float(jnp.max(adjacent_ratio)) < 1.3


@pytest.mark.unit
def test_moderate_ha_rect_duct_mesh_has_strictly_positive_face_spacing():
    case = make_shercliff_case(ha=20.0, width=0.2, height=0.2, ny=97, nz=97)
    mesh = _build_mesh(case)

    assert float(jnp.min(mesh.dy)) > 0.0
    assert float(jnp.min(mesh.dz)) > 0.0


@pytest.mark.validation
@pytest.mark.external
def test_small_hunt_solution_matches_bundled_reference_profiles():
    _closed_channel_root_or_skip()
    _, solution, comparison = solve_closed_channel_benchmark(
        "hunt",
        ha=20.0,
        width=0.2,
        height=0.2,
        ny=17,
        nz=17,
        wall_cells=4,
        wall_thickness=0.02,
        fluid_conductivity=1.0,
        conducting_wall_conductivity=0.25,
        insulating_wall_conductivity=1.0e-12,
        density=1.0,
        viscosity=1.0,
        coupling_iterations=10,
        potential_iterations=80,
        max_steps=60,
        velocity_update_limit=5.0e-4,
        current_reconstruction="face_averaged",
        potential_tolerance=1.0e-8,
    )
    assert solution.diagnostics.potential_residual_history.size > 0
    # This routine gate stays intentionally small; the publication validation
    # examples use finer meshes for the <= O(1e-2) analytical overlays.
    assert comparison.y_profile.l2_error < 0.065
    assert comparison.z_profile.l2_error < 0.04
    assert (
        combined_profile_error(
            comparison.y_profile.l2_error, comparison.z_profile.l2_error
        )
        < 0.055
    )


@pytest.mark.unit
def test_magnetic_field_components_support_analytic_and_tabulated_fields(
    tmp_path: Path,
):
    mesh = generate_rect_duct_mesh(width=2.0, height=2.0, ny=4, nz=4)
    analytic = MagneticFieldSpec(
        kind="analytic",
        fn=lambda y, z: jnp.stack((jnp.zeros_like(y), y + z, y - z), axis=-1),
    )
    bx, by, bz = magnetic_field_components(analytic, mesh, time=0.0)

    assert jnp.allclose(bx, 0.0)
    assert by.shape == mesh.yz_shape
    assert bz.shape == mesh.yz_shape

    y = np.asarray(mesh.y_centers, dtype=float)
    z = np.asarray(mesh.z_centers, dtype=float)
    yy, zz = np.meshgrid(y, z, indexing="ij")
    path = write_tabulated_field_npz(
        tmp_path / "field.npz",
        y=y,
        z=z,
        bx=np.zeros_like(yy),
        by=yy + zz,
        bz=yy - zz,
    )
    tbx, tby, tbz = magnetic_field_components(
        MagneticFieldSpec(kind="tabulated", table_path=str(path)), mesh, time=0.0
    )
    assert jnp.allclose(tbx, 0.0)
    assert jnp.allclose(tby, by)
    assert jnp.allclose(tbz, bz)

    with pytest.raises(ValueError, match="requires fn"):
        magnetic_field_components(MagneticFieldSpec(kind="analytic"), mesh, time=0.0)


@pytest.mark.unit
def test_boundary_sides_support_aliases_and_csv_lists():
    assert _boundary_sides(
        BoundaryCondition("lr", "insulating", side="left_right")
    ) == ("left", "right")
    assert _boundary_sides(
        BoundaryCondition("tb", "insulating", side="top_bottom")
    ) == ("bottom", "top")
    assert _boundary_sides(
        BoundaryCondition("mix", "insulating", side="left, top")
    ) == ("left", "top")
    assert _boundary_sides(BoundaryCondition("none", "insulating")) == ()


@pytest.mark.unit
def test_build_material_fields_handles_missing_solid_region_assignment_with_layered_fallback():
    case = CaseSpec(
        name="layered_material_fallback",
        geometry=GeometrySpec(
            kind="layered_duct",
            width=2.0,
            height=2.0,
            ny=4,
            nz=4,
            wall_thickness=(0.1, 0.1, 0.1, 0.1),
            wall_cells=(1, 1, 1, 1),
            target_ha=5.0,
        ),
        regions=(
            RegionSpec(
                name="fluid", kind="fluid", conductivity=2.0, density=3.0, viscosity=4.0
            ),
            RegionSpec(
                name="wall", kind="solid", conductivity=5.0, density=6.0, viscosity=7.0
            ),
        ),
        magnetic_field=MagneticFieldSpec(kind="constant", value=(0.0, 0.0, 1.0)),
        boundary_conditions=(
            BoundaryCondition(
                "bogus", "conducting_wall", region="missing", side="left"
            ),
        ),
        time_stepper=TimeStepperConfig(dt=0.1, t_final=0.1, max_steps=1),
    )
    mesh = generate_layered_duct_mesh(
        width=2.0,
        height=2.0,
        ny=4,
        nz=4,
        wall_thickness=(0.1, 0.1, 0.1, 0.1),
        wall_cells=(1, 1, 1, 1),
        target_ha=5.0,
    )

    fields = build_material_fields(case, mesh)

    assert jnp.allclose(fields.conductivity[~fields.fluid_mask], 5.0)
    assert jnp.allclose(fields.density[~fields.fluid_mask], 6.0)
    assert jnp.allclose(fields.viscosity[~fields.fluid_mask], 7.0)


@pytest.mark.unit
def test_build_material_fields_uses_explicit_multilayer_mesh_sigma():
    mesh = generate_multilayer_duct_mesh(
        width=1.0,
        height=1.0,
        ny=4,
        nz=4,
        fluid_conductivity=2.0,
        wall_layers={
            "left": (
                WallLayer("aln", 1.0e-8, 0.01, 1),
                WallLayer("metal", 7.0, 0.01, 1),
            ),
            "right": (
                WallLayer("aln", 1.0e-8, 0.01, 1),
                WallLayer("metal", 7.0, 0.01, 1),
            ),
        },
    )
    case = CaseSpec(
        name="explicit_multilayer_sigma",
        geometry=GeometrySpec(kind="layered_duct", width=1.0, height=1.0, ny=4, nz=4),
        regions=(
            RegionSpec(
                name="fluid",
                kind="fluid",
                conductivity=99.0,
                density=3.0,
                viscosity=4.0,
            ),
        ),
        magnetic_field=MagneticFieldSpec(kind="constant", value=(0.0, 0.0, 1.0)),
        boundary_conditions=(BoundaryCondition("walls", "insulating"),),
        time_stepper=TimeStepperConfig(dt=0.1, t_final=0.1, max_steps=1),
    )

    fields = build_material_fields(case, mesh)

    assert fields.conductivity.shape == mesh.yz_shape
    assert float(fields.conductivity[mesh.region_ids == 0][0]) == pytest.approx(2.0)
    assert float(
        fields.conductivity[mesh.region_ids == mesh.region_names.index("left:aln")][0]
    ) == pytest.approx(1.0e-8)
    assert float(
        fields.conductivity[mesh.region_ids == mesh.region_names.index("left:metal")][0]
    ) == pytest.approx(7.0)
