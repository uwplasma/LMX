from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lmx
import lmx.cases as cases_impl
from lmx.cases import (
    build_hartmann_autodiff_problem,
    hartmann_mean_velocity,
    hartmann_mean_velocity_gradients,
    make_hartmann_case,
    make_hunt_case,
    make_shercliff_case,
    solve_differentiable_hartmann,
    solve_steady,
    solve_transient,
)
from lmx.mesh import (
    generate_layered_duct_mesh,
    generate_multilayer_duct_mesh,
    generate_rect_duct_mesh,
    write_tabulated_field_npz,
)
from lmx.physics import (
    WallLayer,
    _boundary_sides,
    build_material_fields,
    magnetic_field_components,
)
from lmx.q2d import Q2DProblem, make_q2d_case, solve_q2d
from lmx.solvers import _build_mesh
from lmx.specs import (
    BoundaryCondition,
    CaseSpec,
    Diagnostics,
    GeometrySpec,
    MagneticFieldSpec,
    MHDState,
    RegionSpec,
    Solution,
    TimeStepperConfig,
)
from lmx.validation import (
    combined_profile_error,
    compare_normalized_profiles,
    compare_profiles_with_shared_scale,
    duct_layer_resolution_gate,
    duct_layer_resolution_metrics,
    extract_midplane_scalar_profile,
    hartmann_acceptance,
    hartmann_analytic_profile,
    hartmann_validation,
    validation_summary,
    write_acceptance_report,
    write_analytic_comparison,
    write_metrics_json,
    write_profile_csv,
)

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
    center_slice = centerline[centerline.shape[0] // 2 - 1 : centerline.shape[0] // 2 + 1]
    assert jnp.allclose(jnp.max(center_slice), jnp.max(centerline), atol=1e-6)
    assert jnp.all(jnp.diff(left_half) >= -5e-6)


@pytest.mark.unit
def test_validation_api_reports_profiles_metrics_and_artifacts(tmp_path: Path):
    case = make_hartmann_case(ha=5.0, ny=6, nz=6)
    solution = _synthetic_solution(case, jnp.ones((case.geometry.ny, case.geometry.nz)))
    solution = replace(
        solution,
        diagnostics=replace(
            solution.diagnostics,
            potential_residual_history=jnp.asarray([1.0e-7]),
        ),
    )
    metrics = validation_summary(solution, case.name, ha=5.0)
    assert "l2_error" not in validation_summary(solution, "shercliff")
    scalar = extract_midplane_scalar_profile(solution, solution.state.jy, axis="z", fluid_only=True)
    coordinate = jnp.asarray([-0.5, 0.0, 0.5])
    reference_coordinate = jnp.asarray([-1.0, 0.0, 1.0])
    normalized = compare_normalized_profiles(
        coordinate,
        jnp.asarray([0.0, 1.0, 0.0]),
        reference_coordinate,
        jnp.asarray([0.0, 2.0, 0.0]),
        simulated_boundary_values=(0.0, 0.0),
    )
    assert compare_normalized_profiles(
        reference_coordinate,
        normalized.simulated,
        reference_coordinate,
        normalized.reference,
    ).l2_error == pytest.approx(0.0)
    shared = compare_profiles_with_shared_scale(
        reference_coordinate,
        normalized.simulated,
        reference_coordinate,
        normalized.reference,
        coordinate_scale=1.0,
        value_scale=1.0,
        simulated_boundary_values=(0.0, 0.0),
    )
    assert compare_profiles_with_shared_scale(
        reference_coordinate,
        normalized.simulated,
        reference_coordinate,
        normalized.reference,
        coordinate_scale=1.0,
        value_scale=1.0,
    ).l2_error == pytest.approx(0.0)
    unsupported = replace(case, magnetic_field=replace(case.magnetic_field, value=(1.0, 0.0, 0.0)))

    assert metrics["potential_residual"] == pytest.approx(1.0e-7)
    assert metrics["linear_residual"] == pytest.approx(0.0)
    assert scalar["coordinate"].shape == scalar["value"].shape
    assert shared.l2_error == pytest.approx(0.0)
    assert combined_profile_error() == pytest.approx(0.0)
    assert combined_profile_error(3.0, 4.0) == pytest.approx(12.5**0.5)
    assert write_metrics_json(metrics, tmp_path / "metrics.json").exists()
    assert write_profile_csv(tmp_path / "profile.csv", scalar).exists()
    assert write_analytic_comparison(shared, tmp_path / "profile.json").exists()
    assert duct_layer_resolution_gate(case, solution.mesh)["layer_resolution_supported"]
    assert not duct_layer_resolution_gate(unsupported, solution.mesh)["layer_resolution_supported"]


@pytest.mark.unit
def test_validation_profiles_cover_walls_singletons_and_invalid_inputs():
    hunt = make_hunt_case(ha=5.0, ny=8, nz=8, wall_cells=1)
    hunt_mesh = _build_mesh(hunt)
    hunt_solution = _synthetic_solution(hunt, jnp.ones(hunt_mesh.yz_shape))
    profile = extract_midplane_scalar_profile(
        hunt_solution, hunt_solution.state.jy, axis="y", fluid_only=True
    )
    assert profile["coordinate"].size < hunt_mesh.y_centers.size
    assert duct_layer_resolution_gate(hunt, hunt_mesh)["layer_resolution_supported"]
    unsupported = replace(
        hunt,
        magnetic_field=replace(hunt.magnetic_field, kind="analytic", value=None),
    )
    assert not duct_layer_resolution_gate(unsupported, hunt_mesh)["layer_resolution_supported"]

    singleton = make_hartmann_case(ha=5.0, ny=3, nz=1)
    singleton_solution = _synthetic_solution(singleton, jnp.ones((3, 1)))
    for axis in ("y", "z"):
        result = extract_midplane_scalar_profile(singleton_solution, singleton_solution.state.jy, axis=axis)
        assert result["coordinate"].shape == result["value"].shape
    with pytest.raises(ValueError, match="Unsupported axis"):
        extract_midplane_scalar_profile(singleton_solution, singleton_solution.state.jy, axis="x")

    values = jnp.ones((1,))
    for name in ("coordinate_scale", "value_scale"):
        kwargs = {"coordinate_scale": 1.0, "value_scale": 1.0, name: 0.0}
        with pytest.raises(ValueError, match=f"{name} must be positive"):
            compare_profiles_with_shared_scale(values, values, values, values, **kwargs)


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
        time_stepper=replace(case.time_stepper, dt=1e-4, t_final=1e-4, max_steps=1, relaxation=1.0),
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

    monkeypatch.setattr(cases_impl, "_fully_developed_case_step", fake_fully_developed_case_step)

    solution = solve_transient(case)
    center_value = float(solution.state.u[solution.state.u.shape[0] // 2, solution.state.u.shape[1] // 2])
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
    acceptance = hartmann_acceptance(solution, ha=20.0, l2_threshold=0.05, linf_threshold=0.2)
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
    assert _boundary_sides(BoundaryCondition("lr", "insulating", side="left_right")) == ("left", "right")
    assert _boundary_sides(BoundaryCondition("tb", "insulating", side="top_bottom")) == ("bottom", "top")
    assert _boundary_sides(BoundaryCondition("mix", "insulating", side="left, top")) == ("left", "top")
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
            RegionSpec(name="fluid", kind="fluid", conductivity=2.0, density=3.0, viscosity=4.0),
            RegionSpec(name="wall", kind="solid", conductivity=5.0, density=6.0, viscosity=7.0),
        ),
        magnetic_field=MagneticFieldSpec(kind="constant", value=(0.0, 0.0, 1.0)),
        boundary_conditions=(BoundaryCondition("bogus", "conducting_wall", region="missing", side="left"),),
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


@pytest.fixture(scope="module")
def hartmann_problem():
    return build_hartmann_autodiff_problem(
        ny=12,
        nz=12,
        macro_iterations=3,
        potential_iterations=12,
        velocity_iterations=16,
    )


def test_differentiable_hartmann_solution_returns_finite_fields(hartmann_problem):
    u, phi = solve_differentiable_hartmann(hartmann_problem, forcing=1.0, hartmann_number=5.0)

    assert u.shape == (12, 12)
    assert phi.shape == (12, 12)
    assert jnp.isfinite(u).all()
    assert jnp.isfinite(phi).all()


def test_hartmann_mean_velocity_is_differentiable(hartmann_problem):
    value, gradient = jax.value_and_grad(
        lambda ha: hartmann_mean_velocity(hartmann_problem, forcing=1.0, hartmann_number=ha)
    )(5.0)

    assert jnp.isfinite(value)
    assert jnp.isfinite(gradient)


def test_profile_loss_gradient_step_reduces_objective(hartmann_problem):
    target_u, _ = solve_differentiable_hartmann(hartmann_problem, forcing=1.0, hartmann_number=9.0)
    target_profile = target_u[:, target_u.shape[1] // 2]

    def objective(ha):
        u, _ = solve_differentiable_hartmann(hartmann_problem, forcing=1.0, hartmann_number=ha)
        profile = u[:, u.shape[1] // 2]
        return jnp.mean(
            (profile / jnp.max(jnp.abs(profile)) - target_profile / jnp.max(jnp.abs(target_profile))) ** 2
        )

    loss0, grad0 = jax.value_and_grad(objective)(4.0)
    loss1 = objective(jnp.clip(4.0 - 2.0 * grad0, 0.5, 30.0))

    assert jnp.isfinite(loss0)
    assert jnp.isfinite(grad0)
    assert float(loss1) <= float(loss0)


def test_mean_velocity_gradients_match_finite_difference(hartmann_problem):
    autodiff = hartmann_mean_velocity_gradients(hartmann_problem, forcing=1.1, hartmann_number=5.0)
    delta = 1.0e-3

    def objective(forcing, ha):
        return hartmann_mean_velocity(hartmann_problem, forcing=forcing, hartmann_number=ha)

    finite_forcing = (objective(1.1 + delta, 5.0) - objective(1.1 - delta, 5.0)) / (2.0 * delta)
    finite_ha = (objective(1.1, 5.0 + delta) - objective(1.1, 5.0 - delta)) / (2.0 * delta)

    assert jnp.isfinite(autodiff["d_mean_velocity_d_forcing"])
    assert jnp.isfinite(autodiff["d_mean_velocity_d_ha"])
    assert float(jnp.abs(autodiff["d_mean_velocity_d_forcing"] - finite_forcing)) < 5.0e-2
    assert float(jnp.abs(autodiff["d_mean_velocity_d_ha"] - finite_ha)) < 5.0e-2


@pytest.mark.physics
def test_q2d_model_contract_refinement_and_failures():
    case = make_q2d_case(
        shape=(18, 18),
        length=(2.0 * np.pi, 3.0 * np.pi),
        mode=(2, 3),
        viscosity=0.02,
        hartmann_friction=0.3,
        dt=0.01,
        steps=4,
        history_stride=2,
    )
    result = lmx.solve(case)
    wave_number_squared = (2.0 * np.pi * 2 / case.length[0]) ** 2 + (2.0 * np.pi * 3 / case.length[1]) ** 2
    expected = case.initial_vorticity * jnp.exp(
        -(case.viscosity * wave_number_squared + case.hartmann_friction) * case.dt * case.steps
    )

    assert result.converged and result.status == "completed" and result.steps == 4
    assert result.fields is result
    assert result.vorticity == pytest.approx(expected, rel=2.0e-6, abs=2.0e-7)
    assert result.vorticity_history.shape == (3, 18, 18)
    assert result.frame_times == pytest.approx([0.0, 0.02, 0.04])
    assert result.diagnostics.energy_budget_residual < 2.0e-6
    assert result.diagnostics.max_divergence < 1.0e-6
    assert result.residual < 2.0e-6

    x = jnp.arange(18) * 2.0 * jnp.pi / 18
    initial = (jnp.sin(x[:, None]) * jnp.cos(2.0 * x[None, :])).astype(jnp.float32)
    forcing = 0.03 * jnp.cos(3.0 * x[:, None] - x[None, :])
    forced = solve_q2d(
        Q2DProblem(
            initial,
            forcing=forcing,
            viscosity=0.01,
            hartmann_friction=0.2,
            dt=0.002,
            steps=4,
        )
    )

    assert forced.vorticity_history.shape == (0, 18, 18)
    assert forced.vorticity.dtype == jnp.float32
    assert forced.frame_times.size == 0
    assert jnp.isfinite(forced.vorticity).all()
    assert forced.diagnostics.kinetic_energy_final > 0.0
    assert forced.diagnostics.enstrophy_final > 0.0

    def solve_on_grid(size):
        coordinate = jnp.arange(size) * 2.0 * jnp.pi / size
        x, y = coordinate[:, None], coordinate[None, :]
        vorticity = jnp.sin(x) * jnp.sin(y) + 0.4 * jnp.sin(2.0 * x + 0.2) * jnp.sin(3.0 * y)
        return np.asarray(
            solve_q2d(
                Q2DProblem(
                    vorticity,
                    viscosity=0.003,
                    hartmann_friction=0.04,
                    dt=0.005,
                    steps=4,
                )
            ).vorticity
        )

    coarse, medium, reference = (solve_on_grid(size) for size in (9, 18, 36))
    coarse_error = np.linalg.norm(coarse - reference[::4, ::4]) / np.linalg.norm(reference[::4, ::4])
    medium_error = np.linalg.norm(medium - reference[::2, ::2]) / np.linalg.norm(reference[::2, ::2])

    assert coarse_error < 5.0e-3
    assert medium_error < coarse_error * 0.1

    coordinate = 2.0 * jnp.pi * jnp.arange(16) / 16
    mode = jnp.sin(coordinate[:, None]) * jnp.sin(coordinate[None, :])
    parameters = jnp.asarray([1.0, 0.02, 0.1, 0.0, 2.0 * np.pi, 0.01], dtype=jnp.float32)

    def objective(values, checkpoint_size=None):
        vorticity, _, _ = lmx.evolve_q2d(
            values[0] * mode,
            forcing=values[3] * mode,
            length=(values[4], 2.0 * jnp.pi),
            viscosity=values[1],
            hartmann_friction=values[2],
            dt=values[5],
            steps=32,
            adjoint_checkpoint_size=checkpoint_size,
        )
        return jnp.mean(vorticity**2)

    value, gradient = jax.jit(jax.value_and_grad(objective))(parameters)
    time = parameters[5] * 32
    expected_gradient = jnp.asarray(
        [
            2.0 * value,
            -4.0 * time * value,
            -2.0 * time * value,
            gradient[3],
            4.0 * time * parameters[1] * value / parameters[4],
            -2.0 * 32 * (2.0 * parameters[1] + parameters[2]) * value,
        ]
    )
    assert gradient == pytest.approx(expected_gradient, rel=3.0e-6, abs=2.0e-7)
    perturbation = jnp.zeros_like(parameters).at[3].set(3.0e-2)
    finite_forcing = (
        jax.jit(objective)(parameters + perturbation) - jax.jit(objective)(parameters - perturbation)
    ) / 6.0e-2
    assert gradient[3] == pytest.approx(finite_forcing, rel=1.0e-4)
    direction = jnp.asarray([0.2, -0.4, 0.7, 0.1, -0.05, 0.3], dtype=parameters.dtype)
    tangent = jax.jvp(objective, (parameters,), (direction,))[1]
    pullback = jax.vjp(objective, parameters)[1](jnp.ones_like(value))[0]
    assert tangent == pytest.approx(jnp.vdot(pullback, direction), rel=2.0e-6)

    bounded = jax.jit(jax.value_and_grad(objective)).lower(parameters).compile()
    full_tape = jax.jit(jax.value_and_grad(lambda values: objective(values, 32))).lower(parameters).compile()
    assert bounded.memory_analysis().temp_size_in_bytes < 0.5 * full_tape.memory_analysis().temp_size_in_bytes

    unstable = solve_q2d(
        make_q2d_case(shape=(18, 18), amplitude=20.0, viscosity=0.0, hartmann_friction=0.0, dt=0.1, steps=4)
    )
    assert unstable.status == "courant_limit_exceeded"
    assert not unstable.converged

    invalid = (
        (lambda: Q2DProblem(jnp.zeros(4)), "2-D array"),
        (lambda: Q2DProblem(jnp.zeros((4, 4)), forcing=jnp.zeros((3, 4))), "must match"),
        (lambda: Q2DProblem(jnp.zeros((4, 4)), length=(0.0, 1.0)), "positive"),
        (lambda: Q2DProblem(jnp.zeros((4, 4)), viscosity=-1.0), "non-negative"),
        (lambda: Q2DProblem(jnp.zeros((4, 4)), dt=0.0), "dt and steps"),
        (lambda: Q2DProblem(jnp.zeros((4, 4)), adjoint_checkpoint_size=0), "checkpoint_size"),
        (lambda: make_q2d_case(mode=(0, 1)), "describe two axes"),
    )
    for action, message in invalid:
        with pytest.raises(ValueError, match=message):
            action()
