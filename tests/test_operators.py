import jax.numpy as jnp
import pytest

from lmx.mesh import generate_layered_duct_mesh, generate_rect_duct_mesh
from lmx.operators import (
    _broadcast_spacing_y,
    _broadcast_spacing_z,
    center_coordinates,
    center_spacing_y,
    center_spacing_z,
    divergence_flux,
    face_average_x,
    face_average_z,
    face_divergence,
    gradient_scalar,
    laplacian_scalar,
)


pytestmark = pytest.mark.unit


def test_gradient_of_linear_field():
    mesh = generate_rect_duct_mesh(width=2.0, height=2.0, ny=32, nz=32)
    y, z = jnp.meshgrid(mesh.y_centers, mesh.z_centers, indexing="ij")
    field = 2.0 * y - 3.0 * z
    gy, gz = gradient_scalar(field, mesh)
    assert jnp.allclose(gy[2:-2, 2:-2], 2.0, atol=5e-2)
    assert jnp.allclose(gz[2:-2, 2:-2], -3.0, atol=5e-2)


def test_gradient_of_linear_field_on_clustered_mesh_is_exact_near_boundaries():
    mesh = generate_layered_duct_mesh(
        width=2.0,
        height=2.0,
        ny=48,
        nz=48,
        wall_thickness=(0.0, 0.0, 0.1, 0.1),
        wall_cells=(0, 0, 2, 2),
        target_ha=100.0,
    )
    y, z = jnp.meshgrid(mesh.y_centers, mesh.z_centers, indexing="ij")
    field = 2.0 * y - 3.0 * z
    gy, gz = gradient_scalar(field, mesh)
    assert jnp.allclose(gy[:, 2:-2], 2.0, atol=5e-2)
    assert jnp.allclose(gz[2:-2, :], -3.0, atol=5e-2)


def test_laplacian_of_quadratic_field():
    mesh = generate_rect_duct_mesh(width=2.0, height=2.0, ny=40, nz=40)
    y, z = jnp.meshgrid(mesh.y_centers, mesh.z_centers, indexing="ij")
    field = y**2 + z**2
    lap = laplacian_scalar(field, mesh)
    assert jnp.allclose(lap[2:-2, 2:-2], 4.0, atol=2e-1)


def test_laplacian_of_quadratic_field_on_clustered_mesh():
    mesh = generate_layered_duct_mesh(
        width=2.0,
        height=2.0,
        ny=48,
        nz=48,
        wall_thickness=(0.0, 0.0, 0.1, 0.1),
        wall_cells=(0, 0, 2, 2),
        target_ha=100.0,
    )
    y, z = jnp.meshgrid(mesh.y_centers, mesh.z_centers, indexing="ij")
    field = y**2 + z**2
    lap = laplacian_scalar(field, mesh, mask=mesh.fluid_mask)
    assert mesh.fluid_mask is not None
    interior = mesh.fluid_mask.copy()
    interior = interior & jnp.pad(mesh.fluid_mask[:-1, :], ((1, 0), (0, 0)))
    interior = interior & jnp.pad(mesh.fluid_mask[1:, :], ((0, 1), (0, 0)))
    interior = interior & jnp.pad(mesh.fluid_mask[:, :-1], ((0, 0), (1, 0)))
    interior = interior & jnp.pad(mesh.fluid_mask[:, 1:], ((0, 0), (0, 1)))
    interior = interior.at[:2, :].set(False)
    interior = interior.at[-2:, :].set(False)
    interior = interior.at[:, :2].set(False)
    interior = interior.at[:, -2:].set(False)
    assert jnp.allclose(lap[interior], 4.0, atol=3e-1)


def test_operator_helpers_cover_spacings_face_averages_and_divergence():
    mesh = generate_rect_duct_mesh(width=2.0, height=3.0, ny=3, nz=4)
    field = jnp.arange(mesh.ny * mesh.nz, dtype=float).reshape(mesh.yz_shape)

    yy, zz = center_coordinates(mesh)
    assert yy.shape == mesh.yz_shape
    assert zz.shape == mesh.yz_shape
    assert _broadcast_spacing_y(mesh).shape == (mesh.ny, 1)
    assert _broadcast_spacing_z(mesh).shape == (1, mesh.nz)
    assert center_spacing_y(mesh).shape == (mesh.ny - 1,)
    assert center_spacing_z(mesh).shape == (mesh.nz - 1,)

    face_y = face_average_x(field)
    face_z = face_average_z(field)
    assert face_y.shape == (mesh.ny - 1, mesh.nz)
    assert face_z.shape == (mesh.ny, mesh.nz - 1)

    div_flux = divergence_flux(jnp.ones(mesh.yz_shape), 2.0 * jnp.ones(mesh.yz_shape), mesh)
    assert div_flux.shape == mesh.yz_shape

    face_flux_y = jnp.zeros((mesh.ny + 1, mesh.nz))
    face_flux_z = jnp.zeros((mesh.ny, mesh.nz + 1))
    face_flux_y = face_flux_y.at[1:-1, :].set(1.0)
    face_flux_z = face_flux_z.at[:, 1:-1].set(-0.5)
    div_faces = face_divergence(face_flux_y, face_flux_z, mesh)
    assert div_faces.shape == mesh.yz_shape


def test_center_spacing_returns_empty_for_single_cell_axes():
    mesh = generate_rect_duct_mesh(width=1.0, height=1.0, ny=1, nz=1)
    assert center_spacing_y(mesh).shape == (0,)
    assert center_spacing_z(mesh).shape == (0,)
