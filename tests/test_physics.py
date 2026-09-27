import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import lmhdx
import lmhdx.cases as cases_impl
from lmhdx.cases import (
    make_hartmann_case,
    make_hunt_case,
    make_shercliff_case,
    solve_fully_developed_fields,
    solve_steady,
    solve_transient,
)
from lmhdx.mesh import (
    generate_layered_duct_mesh,
    generate_multilayer_duct_mesh,
    generate_rect_duct_mesh,
    write_tabulated_field_npz,
)
from lmhdx.physics import (
    WallLayer,
    _boundary_sides,
    build_material_fields,
    magnetic_field_components,
)
from lmhdx.q2d import Q2DProblem, make_q2d_case, solve_q2d
from lmhdx.solvers import _build_mesh
from lmhdx.specs import (
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
from lmhdx.validation import (
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
def differentiable_hartmann_case():
    return make_hartmann_case(ha=5, ny=8, nz=8)


def test_differentiable_fields_match_the_production_steady_solution(differentiable_hartmann_case):
    fields = solve_fully_developed_fields(differentiable_hartmann_case)
    production = solve_steady(differentiable_hartmann_case)

    assert all(field.shape == (8, 8) and jnp.isfinite(field).all() for field in fields)
    assert jnp.linalg.norm(fields[0] - production.state.u) / jnp.linalg.norm(production.state.u) < 2e-6
    assert jnp.linalg.norm(fields[1] - production.state.phi) / jnp.linalg.norm(production.state.phi) < 2e-6


def test_layered_hunt_fields_and_implicit_gradient_match_independent_checks():
    case = make_hunt_case(ha=5, ny=6, nz=6, wall_cells=1, insulator_cells=1)
    fields = solve_fully_developed_fields(case)
    production = solve_steady(case)

    assert jnp.linalg.norm(fields[0] - production.state.u) / jnp.linalg.norm(production.state.u) < 2e-6
    assert jnp.linalg.norm(fields[1] - production.state.phi) / jnp.linalg.norm(production.state.phi) < 2e-6

    def response(scale):
        return jnp.mean(solve_fully_developed_fields(case, magnetic_field_scale=scale)[0])

    compiled = jax.jit(response)
    value, derivative = jax.jit(jax.value_and_grad(response))(1.0)
    delta = 1e-3
    finite = (compiled(1.0 + delta) - compiled(1.0 - delta)) / (2.0 * delta)
    assert jnp.isfinite(value)
    assert derivative == pytest.approx(finite, rel=2e-5, abs=1e-8)


@pytest.mark.physics
def test_cold_hunt_steady_solve_certifies_the_discrete_affine_problem():
    """Issue #113: a cold steady Hunt report is certified without the solver's own norms."""

    import scipy.sparse
    import scipy.sparse.linalg

    from lmhdx.design import volumetric_flow_rate
    from lmhdx.mesh import apply_five_point_operator
    from lmhdx.solvers import (
        _compute_current_and_lorentz,
        _face_current_components,
        _velocity_system_coefficients,
    )

    case = make_hunt_case(
        ha=20,
        width=2,
        height=2,
        ny=24,
        nz=24,
        wall_cells=3,
        wall_thickness=0.1,
        insulator_cells=3,
        insulator_thickness=0.1,
        fluid_conductivity=1,
        wall_conductance_ratio=0.05,
        insulator_conductivity_ratio=1e-12,
        density=1,
        viscosity=1,
    )
    # The max-norm potential residual floors near 3e-12 at this 1e-12 wall contrast.
    case = replace(
        case,
        time_stepper=replace(
            case.time_stepper,
            potential_iterations=160,
            steady_tolerance=1e-12,
            steady_potential_tolerance=1e-10,
        ),
        solver=replace(case.solver, coupling_tolerance=1e-12),
    )
    unit_velocity, unit_potential, *_ = solve_fully_developed_fields(case, forcing=1.0)
    drive = 0.05 / float(volumetric_flow_rate(case, unit_velocity))
    solution = solve_steady(replace(case, forcing=drive))
    velocity, potential = solution.state.u, solution.state.phi

    def relative(left, right):
        left, right = np.asarray(left).ravel(), np.asarray(right).ravel()
        return float(np.linalg.norm(left - right) / np.linalg.norm(right))

    assert solution.status == "converged" and solution.residual <= 1e-12
    assert float(volumetric_flow_rate(case, velocity)) == pytest.approx(0.05, rel=1e-10)
    assert relative(velocity, drive * unit_velocity) <= 1e-10
    assert relative(potential, drive * unit_potential) <= 1e-10

    # Assemble the discrete momentum and charge equations from the operators
    # alone: viscous stencil = Lorentz force + drive, net face current = 0.
    mesh = solution.mesh
    materials = build_material_fields(case, mesh)
    _, by, bz = magnetic_field_components(case.magnetic_field, mesh)
    fluid, sigma = materials.fluid_mask, materials.conductivity
    metric = mesh.dy[:, None] * mesh.dz[None, :]
    viscous = [
        coefficient * metric
        for coefficient in _velocity_system_coefficients(
            mesh, materials.viscosity, jnp.zeros_like(velocity), fluid
        )
    ]
    shape, cells, anchor = mesh.yz_shape, velocity.size, case.reference_phi_cell

    def operator(state):
        u, phi = state[:cells].reshape(shape), state[cells:].reshape(shape)
        _, _, lorentz = _compute_current_and_lorentz(mesh, sigma, fluid, u, phi, by, bz)
        momentum = jnp.where(
            fluid, apply_five_point_operator(*viscous, u) - metric * lorentz / materials.density, u
        )
        face_jy, face_jz, _, _ = _face_current_components(mesh, sigma, fluid, u, phi, by, bz)
        net_current = (
            jnp.diff(jnp.pad(face_jy, ((1, 1), (0, 0))), axis=0) * mesh.dz[None, :]
            + jnp.diff(jnp.pad(face_jz, ((0, 0), (1, 1))), axis=1) * mesh.dy[:, None]
        )
        return jnp.concatenate((momentum.ravel(), net_current.at[anchor].set(phi[anchor]).ravel()))

    matrix = np.array(jax.jacfwd(operator)(jnp.zeros(2 * cells)))
    drive_source = np.where(
        np.asarray(fluid), drive * np.asarray(metric) / np.asarray(materials.density), 0.0
    )
    rhs = np.concatenate((drive_source.ravel(), np.zeros(cells)))
    state = np.concatenate((np.asarray(velocity).ravel(), np.asarray(potential).ravel()))
    residual = matrix @ state - rhs
    motional_source = matrix[cells:, :cells] @ state[:cells]
    assert np.linalg.norm(residual[:cells]) / np.linalg.norm(rhs[:cells]) <= 1e-10
    assert np.linalg.norm(residual[cells:]) / np.linalg.norm(motional_source) <= 1e-10

    # An independent sparse direct solve of the same equations.
    row_scale = 1.0 / np.max(np.abs(matrix), axis=1)
    direct = scipy.sparse.linalg.spsolve(
        scipy.sparse.csr_matrix(matrix * row_scale[:, None]), rhs * row_scale
    )
    assert relative(velocity, direct[:cells]) <= 1e-10
    assert relative(potential, direct[cells:]) <= 1e-10


def test_fully_developed_implicit_gradients_match_independent_checks(differentiable_hartmann_case):
    def objective(forcing, field_scale):
        velocity, *_ = solve_fully_developed_fields(
            differentiable_hartmann_case,
            forcing=forcing,
            magnetic_field_scale=field_scale,
        )
        return jnp.mean(velocity)

    compiled = jax.jit(objective)
    value, (forcing_gradient, field_gradient) = jax.jit(jax.value_and_grad(objective, argnums=(0, 1)))(
        1.1, 1.0
    )
    delta = 1e-3
    finite_field = (compiled(1.1, 1.0 + delta) - compiled(1.1, 1.0 - delta)) / (2.0 * delta)

    assert forcing_gradient == pytest.approx(value / 1.1, rel=2e-7)
    assert field_gradient == pytest.approx(finite_field, rel=2e-5, abs=1e-8)

    def velocity(parameters):
        return solve_fully_developed_fields(
            differentiable_hartmann_case,
            forcing=parameters[0],
            magnetic_field_scale=parameters[1],
        )[0]

    parameters = jnp.asarray([1.1, 1.0])
    direction = jnp.asarray([0.3, -0.2])
    cotangent = jnp.cos(jnp.arange(64, dtype=float)).reshape(8, 8)
    _, tangent = jax.jvp(velocity, (parameters,), (direction,))
    _, pullback = jax.vjp(velocity, parameters)
    assert jnp.vdot(cotangent, tangent) == pytest.approx(
        jnp.vdot(direction, pullback(cotangent)[0]), rel=2e-8, abs=1e-10
    )


def test_shercliff_field_scale_gradient_matches_finite_difference():
    case = make_shercliff_case(ha=5, ny=6, nz=6)

    def response(scale):
        return jnp.mean(solve_fully_developed_fields(case, magnetic_field_scale=scale)[0])

    compiled = jax.jit(response)
    value, derivative = jax.jit(jax.value_and_grad(response))(1.0)
    delta = 1e-3
    finite = (compiled(1.0 + delta) - compiled(1.0 - delta)) / (2.0 * delta)
    assert jnp.isfinite(value)
    assert derivative == pytest.approx(finite, rel=2e-5, abs=1e-8)


@pytest.mark.physics
@pytest.mark.parametrize("control", ["forcing", "length", "viscosity", "hartmann_friction", "dt"])
def test_q2d_mixed_precision_matches_analytic_fields_and_derivatives(control):
    coordinate = 2.0 * jnp.pi * jnp.arange(8) / 8
    initial = (jnp.sin(coordinate[:, None]) * jnp.sin(coordinate[None, :])).astype(jnp.float32)

    def inputs(value):
        parameters = dict(length=(2.0 * np.pi,) * 2, viscosity=0.02, hartmann_friction=0.1, dt=0.01)
        if control == "forcing":
            parameters[control] = 0.03 * value * initial
        elif control == "length":
            parameters[control] = (2.0 * np.pi * value, 2.0 * np.pi)
        else:
            parameters[control] *= value
        return parameters

    def objective(value):
        return jnp.mean(lmhdx.evolve_q2d(initial, **inputs(value), steps=4)[0] ** 2)

    def analytic(value):
        parameters = inputs(value)
        rate = parameters["viscosity"] * sum((2.0 * np.pi / side) ** 2 for side in parameters["length"])
        rate += parameters["hartmann_friction"]
        time = 4 * parameters["dt"]
        decay = jnp.exp(-rate * time)
        source = 0.03 * value if control == "forcing" else 0.0
        return initial * (decay + source * (-jnp.expm1(-rate * time)) / rate)

    value = jnp.asarray(1.1, dtype=jnp.float64)
    fields = jax.jit(lambda x: lmhdx.evolve_q2d(initial, **inputs(x), steps=4))(value)
    result = solve_q2d(Q2DProblem(initial, **inputs(value), steps=4, history_stride=2))
    assert result.problem.initial_vorticity.dtype == result.problem.forcing.dtype == jnp.float64
    assert all(field.dtype == jnp.float64 for field in fields)
    assert result.vorticity == pytest.approx(fields[0], abs=1.0e-12)
    assert fields[0] == pytest.approx(analytic(value), abs=2.0e-8)
    actual = jax.jit(jax.value_and_grad(objective))(value)
    expected = jax.value_and_grad(lambda x: jnp.mean(analytic(x) ** 2))(value)
    assert np.asarray(actual) == pytest.approx(np.asarray(expected), rel=2.0e-6, abs=1.0e-10)
    tangent = jax.jvp(objective, (value,), (jnp.ones_like(value),))[1]
    assert tangent == pytest.approx(actual[1], rel=1.0e-9, abs=1.0e-12)


@pytest.mark.physics
@pytest.mark.parametrize("devices", [2, 4])
def test_q2d_sharded_setup_preserves_fields_and_diagnostics(devices):
    code = """
from dataclasses import replace, asdict
import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from lmhdx.q2d import evolve_q2d, make_q2d_case, solve_q2d
assert len(jax.devices()) in (2, 4)
placement = NamedSharding(Mesh(np.array(jax.devices()), ('d',)), P('d', None))
for dtype in (np.float32, np.float64):
    case = make_q2d_case(shape=(16, 16), steps=4, history_stride=2)
    initial = np.asarray(case.initial_vorticity, dtype=dtype)
    reference = solve_q2d(replace(case, initial_vorticity=jax.device_put(initial, jax.devices()[0])))
    sharded = solve_q2d(replace(case, initial_vorticity=jax.device_put(initial, placement)))
    assert reference.status == sharded.status == 'completed'
    for name in ('vorticity', 'velocity_x', 'velocity_y', 'vorticity_history'):
        expected, actual = np.asarray(getattr(reference, name)), np.asarray(getattr(sharded, name))
        assert actual.dtype == dtype
        np.testing.assert_allclose(actual, expected, rtol=100*np.finfo(dtype).eps, atol=100*np.finfo(dtype).eps)
    np.testing.assert_allclose(list(asdict(sharded.diagnostics).values()),
                               list(asdict(reference.diagnostics).values()),
                               rtol=100*np.finfo(dtype).eps, atol=100*np.finfo(dtype).eps)
    def objective(field, friction):
        final = evolve_q2d(field, hartmann_friction=friction, steps=4, dt=.01)[0]
        return jnp.mean(final**2)
    derivative = jax.jit(jax.value_and_grad(objective, argnums=1))
    control = jnp.asarray(.1, dtype=dtype)
    expected = derivative(jax.device_put(initial, jax.devices()[0]), control)
    actual = derivative(jax.device_put(initial, placement), control)
    np.testing.assert_allclose(np.asarray(actual), np.asarray(expected),
                               rtol=100*np.finfo(dtype).eps, atol=100*np.finfo(dtype).eps)
    # A Taylor-Green mode has zero nonlinear advection: d(mean(w²))/d(friction)=-2*t*mean(w²).
    np.testing.assert_allclose(actual[1], -.08*actual[0], rtol=100*np.finfo(dtype).eps)
"""
    subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        timeout=90,
        env={
            **os.environ,
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_X64": "true",
            "XLA_FLAGS": f"--xla_force_host_platform_device_count={devices}",
            "JAX_NUM_CPU_DEVICES": str(devices),
        },
    )


@pytest.mark.physics
def test_q2d_real_dtype_floor_and_weak_scalar_defaults():
    for dtype in (jnp.int32, jnp.float32, jnp.float64):
        initial = jnp.zeros((4, 4), dtype=dtype)
        expected = jnp.result_type(dtype, jnp.float32)
        case = Q2DProblem(initial, steps=1)
        assert case.initial_vorticity.dtype == expected
        assert lmhdx.evolve_q2d(initial, steps=1)[0].dtype == expected
    initial = jnp.zeros((4, 4), dtype=jnp.complex64)
    with pytest.raises(ValueError, match="must be real"):
        Q2DProblem(initial)
    with pytest.raises(ValueError, match="must be real"):
        lmhdx.evolve_q2d(initial)


@pytest.mark.physics
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("stride", [0, 2])
def test_q2d_divergence_diagnostic_uses_reported_final_velocity(dtype, stride):
    initial = np.random.default_rng(71).normal(size=(16, 16)).astype(dtype)
    case = Q2DProblem(initial, length=(3.0, 5.0), steps=5, history_stride=stride, dt=0.001)
    result = solve_q2d(case)
    ux, uy = np.asarray(result.velocity_x), np.asarray(result.velocity_y)
    if stride:
        np.testing.assert_array_equal(result.vorticity_history[-1], result.vorticity)
        assert result.frame_times[-1] == pytest.approx(case.steps * case.dt)
    else:
        assert result.vorticity_history.shape == (0, *initial.shape)
    kx = 2 * np.pi * np.fft.fftfreq(16, d=3.0 / 16)[:, None]
    ky = 2 * np.pi * np.fft.fftfreq(16, d=5.0 / 16)[None, :]
    divergence = np.fft.ifftn(1j * kx * np.fft.fftn(ux) + 1j * ky * np.fft.fftn(uy)).real
    scale = max(np.max(np.abs(ux)), np.max(np.abs(uy))) * max(np.max(abs(kx)), np.max(abs(ky)))
    assert result.status == "completed"
    np.testing.assert_allclose(
        result.diagnostics.max_divergence, np.max(np.abs(divergence)), atol=20 * np.finfo(dtype).eps * scale
    )


def _complex_transform_q2d(problem):
    """NumPy IFRK4 on full-plane complex transforms, the layout the real-transform state replaced."""
    omega, forcing = np.asarray(problem.initial_vorticity), np.asarray(problem.forcing)
    nu, gamma, dt = problem.viscosity, problem.hartmann_friction, problem.dt
    spacing = [side / size for side, size in zip(problem.length, omega.shape, strict=True)]
    kx, ky = (2 * np.pi * np.fft.fftfreq(n, d=h) for n, h in zip(omega.shape, spacing, strict=True))
    kx, ky = kx[:, None], ky[None, :]
    k2 = kx**2 + ky**2
    kept = [np.abs(np.fft.fftfreq(n) * n) <= n / 3 for n in omega.shape]
    mask = kept[0][:, None] & kept[1][None, :]
    decay = np.exp(-dt * (nu * k2 + gamma))
    half = np.sqrt(decay)

    def physical(field_hat):
        return np.fft.ifftn(field_hat).real

    def spectral(field):
        field_hat = np.fft.fftn(field) * mask
        field_hat[0, 0] = 0.0
        return field_hat

    def flow(w):
        psi = np.divide(w, k2, out=np.zeros_like(w), where=k2 > 0)
        return psi, physical(1j * ky * psi), physical(-1j * kx * psi)

    forcing_hat = spectral(forcing)

    def rate(w):
        _, ux, uy = flow(w)
        return (forcing_hat - np.fft.fftn(ux * physical(1j * kx * w) + uy * physical(1j * ky * w))) * mask

    def measures(w):
        psi, ux, uy = flow(w)
        energy, enstrophy = 0.5 * np.mean(ux**2 + uy**2), 0.5 * np.mean(physical(w) ** 2)
        courant = dt * np.max(np.abs(ux) / spacing[0] + np.abs(uy) / spacing[1])
        return (
            energy,
            enstrophy,
            courant,
            -2 * (nu * enstrophy + gamma * energy) + np.mean(physical(psi) * forcing),
        )

    w = spectral(omega)
    before = initial = measures(w)
    budget, courant, frames = 0.0, initial[2], [physical(w)]
    for step in range(1, problem.steps + 1):
        first = rate(w)
        second = rate(half * (w + 0.5 * dt * first))
        third = rate(half * w + 0.5 * dt * second)
        fourth = rate(decay * w + dt * half * third)
        w = (decay * w + dt / 6 * (decay * first + 2 * half * (second + third) + fourth)) * mask
        w[0, 0] = 0.0
        after = measures(w)
        budget += 0.5 * dt * (before[3] + after[3])
        before, courant = after, max(courant, after[2])
        if problem.history_stride and (step % problem.history_stride == 0 or step == problem.steps):
            frames.append(physical(w))
    _, ux, uy = flow(w)
    residual = abs(before[0] - initial[0] - budget) / max(initial[0], abs(budget))
    divergence = np.max(np.abs(physical(1j * kx * np.fft.fftn(ux) + 1j * ky * np.fft.fftn(uy))))
    return (
        physical(w),
        ux,
        uy,
        np.stack(frames),
        (initial[0], before[0], before[1], residual, divergence, courant),
    )


@pytest.mark.physics
@pytest.mark.parametrize(
    "shape,length,viscosity,friction,dt,steps,stride,forced",
    [
        ((16, 16), (3.0, 5.0), 1e-2, 0.1, 1e-3, 5, 2, False),
        ((15, 20), (2.0, 3.0), 1e-2, 0.1, 5e-4, 6, 4, True),
        ((9, 9), (2 * np.pi, 2 * np.pi), 3e-3, 4e-2, 5e-3, 4, 0, False),
        ((64, 64), (2 * np.pi, 2 * np.pi), 0.0, 0.8, 4e-3, 100, 25, True),
    ],
)
def test_q2d_real_transforms_match_the_complex_transform_reference(
    shape, length, viscosity, friction, dt, steps, stride, forced
):
    # Odd and even axes, both Nyquist lines, unresolved initial spectra and a nonlinear 100-step run.
    rng = np.random.default_rng(sum(shape) + steps)
    forcing = jnp.asarray(0.5 * rng.normal(size=shape)) if forced else None
    parameters = dict(length=length, viscosity=viscosity, hartmann_friction=friction, dt=dt, steps=steps)
    problem = Q2DProblem(
        jnp.asarray(rng.normal(size=shape)), forcing=forcing, history_stride=stride, **parameters
    )
    result = solve_q2d(problem)
    evolved = lmhdx.evolve_q2d(problem.initial_vorticity, forcing=problem.forcing, **parameters)
    vorticity, ux, uy, frames, expected = _complex_transform_q2d(problem)

    assert result.vorticity.dtype == jnp.float64
    references = (vorticity, ux, uy)
    for actual, reference in zip(
        (result.vorticity, result.velocity_x, result.velocity_y, *evolved), 2 * references
    ):
        np.testing.assert_allclose(actual, reference, rtol=0, atol=1e-12 * np.max(np.abs(reference)))
    if stride:
        np.testing.assert_allclose(
            result.vorticity_history, frames, rtol=0, atol=1e-12 * np.max(np.abs(frames))
        )
    diagnostics = result.diagnostics
    np.testing.assert_allclose(
        [
            diagnostics.kinetic_energy_initial,
            diagnostics.kinetic_energy_final,
            diagnostics.enstrophy_final,
            diagnostics.max_courant,
        ],
        [expected[0], expected[1], expected[2], expected[5]],
        rtol=1e-12,
    )
    # The residual is a defect normalized by the energy: both sides carry its round-off, about 1e-16.
    assert diagnostics.energy_budget_residual == pytest.approx(expected[3], rel=1e-12, abs=1e-14)
    # The final divergence is round-off in both layouts, so only its size is comparable.
    scale = max(np.max(np.abs(ux)), np.max(np.abs(uy))) * np.pi * max(np.divide(shape, length))
    assert abs(diagnostics.max_divergence - expected[4]) <= 100 * np.finfo(np.float64).eps * scale


@pytest.mark.physics
def test_q2d_energy_acceptance_matches_modal_budget_and_time_refinement():
    residuals = []
    for dt, steps in ((0.25, 8), (0.125, 16), (0.02, 100)):
        case = make_q2d_case(
            shape=(8, 8), viscosity=jnp.float64(0.1), hartmann_friction=1.0, dt=dt, steps=steps
        )
        result = solve_q2d(case)
        # E/E0=exp(-2*(nu*k²+friction)*t); exact trap sum of the physical rate.
        energy = np.exp(-2.4 * dt * np.arange(steps + 1))
        budget = -1.2 * dt * np.sum(energy[:-1] + energy[1:])
        expected = abs(energy[-1] - 1 - budget) / max(1, abs(budget))
        assert result.diagnostics.energy_budget_residual == pytest.approx(expected, rel=2e-5, abs=2e-7)
        assert result.diagnostics.max_courant < 1
        assert result.converged == (expected <= case.energy_budget_tolerance)
        assert result.status == ("completed" if result.converged else "energy_budget_exceeded")
        residuals.append(result.diagnostics.energy_budget_residual)
    assert 3.8 < residuals[0] / residuals[1] < 4.1
    assert residuals[-1] < 1e-3 < residuals[1]
    for amplitude in (1e-4, 1.0):
        scaled = solve_q2d(
            make_q2d_case(
                shape=(8, 8),
                amplitude=amplitude,
                viscosity=jnp.float32(0.1),
                hartmann_friction=1.0,
                dt=0.25,
                steps=8,
            )
        )
        assert not scaled.converged
        assert scaled.diagnostics.energy_budget_residual == pytest.approx(residuals[0], rel=3e-5)
    boundary = solve_q2d(replace(case, energy_budget_tolerance=result.diagnostics.energy_budget_residual))
    rejected = solve_q2d(
        replace(case, energy_budget_tolerance=0.5 * result.diagnostics.energy_budget_residual)
    )
    assert boundary.converged and rejected.status == "energy_budget_exceeded"
    np.testing.assert_array_equal(rejected.vorticity, result.vorticity)
    relaxed = solve_q2d(
        make_q2d_case(
            shape=(8, 8), viscosity=0.1, hartmann_friction=1.0, dt=0.25, steps=8, energy_budget_tolerance=0.1
        )
    )
    assert relaxed.converged
    for tolerance in (0.0, -1.0, np.inf, np.nan):
        with pytest.raises(ValueError, match="energy_budget_tolerance"):
            make_q2d_case(shape=(8, 8), energy_budget_tolerance=tolerance)


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
    result = lmhdx.solve(case)
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
    forcing = (0.03 * jnp.cos(3.0 * x[:, None] - x[None, :])).astype(jnp.float32)
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
        vorticity, _, _ = lmhdx.evolve_q2d(
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
