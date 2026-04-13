from dataclasses import replace
from pathlib import Path

import jax.numpy as jnp
import pytest

from lmx.cases import make_hartmann_case, make_hunt_case, make_shercliff_case
from lmx.core import Diagnostics, MHDState, Solution
from lmx.mesh import generate_layered_duct_mesh, generate_rect_duct_mesh
from lmx.physics import _boundary_sides, build_material_fields, magnetic_field_components
from lmx.specs import BoundaryCondition, CaseSpec, GeometrySpec, MagneticFieldSpec, RegionSpec, TimeStepperConfig
from lmx.solvers import _build_mesh, solve_steady, solve_transient
import lmx.solvers as solvers
from lmx.validation import hartmann_acceptance, hartmann_analytic_profile, write_acceptance_report


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
def test_transient_solver_can_start_from_nonzero_initial_velocity(monkeypatch: pytest.MonkeyPatch):
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

    monkeypatch.setattr(solvers, "_fully_developed_case_step", fake_fully_developed_case_step)

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


@pytest.mark.unit
def test_magnetic_field_components_support_analytic_and_reject_tabulated_without_loader():
    mesh = generate_rect_duct_mesh(width=2.0, height=2.0, ny=4, nz=4)
    analytic = MagneticFieldSpec(
        kind="analytic",
        fn=lambda y, z: jnp.stack((jnp.zeros_like(y), y + z, y - z), axis=-1),
    )
    bx, by, bz = magnetic_field_components(analytic, mesh, time=0.0)

    assert jnp.allclose(bx, 0.0)
    assert by.shape == mesh.yz_shape
    assert bz.shape == mesh.yz_shape

    with pytest.raises(NotImplementedError, match="Tabulated magnetic fields"):
        magnetic_field_components(MagneticFieldSpec(kind="tabulated", table_path="field.csv"), mesh, time=0.0)

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
