import jax.numpy as jnp
import pytest

from lmx.mesh import generate_layered_duct_mesh, generate_rect_duct_mesh
from lmx.operators import gradient_scalar, laplacian_scalar


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
