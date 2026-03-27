import jax.numpy as jnp
import pytest

from lmx.mesh import generate_rect_duct_mesh
from lmx.operators import gradient_scalar, laplacian_scalar


pytestmark = pytest.mark.unit


def test_gradient_of_linear_field():
    mesh = generate_rect_duct_mesh(width=2.0, height=2.0, ny=32, nz=32)
    y, z = jnp.meshgrid(mesh.y_centers, mesh.z_centers, indexing="ij")
    field = 2.0 * y - 3.0 * z
    gy, gz = gradient_scalar(field, mesh)
    assert jnp.allclose(gy[2:-2, 2:-2], 2.0, atol=5e-2)
    assert jnp.allclose(gz[2:-2, 2:-2], -3.0, atol=5e-2)


def test_laplacian_of_quadratic_field():
    mesh = generate_rect_duct_mesh(width=2.0, height=2.0, ny=40, nz=40)
    y, z = jnp.meshgrid(mesh.y_centers, mesh.z_centers, indexing="ij")
    field = y**2 + z**2
    lap = laplacian_scalar(field, mesh)
    assert jnp.allclose(lap[2:-2, 2:-2], 4.0, atol=2e-1)
