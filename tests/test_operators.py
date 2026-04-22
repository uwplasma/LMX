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
    mesh = generate_rect_duct_mesh(width=2.0, height=2.0, ny=48, nz=48, target_ha=100.0, magnetic_axis="z")
    y, z = jnp.meshgrid(mesh.y_centers, mesh.z_centers, indexing="ij")
    field = y**2 + z**2
    lap = laplacian_scalar(field, mesh)
    interior = jnp.ones(mesh.yz_shape, dtype=bool)
    interior = interior.at[:6, :].set(False)
    interior = interior.at[-6:, :].set(False)
    interior = interior.at[:, :6].set(False)
    interior = interior.at[:, -6:].set(False)
    interior_values = lap[interior]
    assert jnp.isfinite(interior_values).all()
    assert float(jnp.mean(interior_values)) == pytest.approx(4.0, abs=0.15)


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


def test_gradient_observed_order_for_smooth_manufactured_solution():
    errors_y = []
    errors_z = []
    spacings = []

    for n in (16, 32, 64):
        mesh = generate_rect_duct_mesh(width=2.0, height=2.0, ny=n, nz=n)
        y, z = jnp.meshgrid(mesh.y_centers, mesh.z_centers, indexing="ij")
        field = jnp.sin(jnp.pi * y) * jnp.cos(0.5 * jnp.pi * z)
        exact_y = jnp.pi * jnp.cos(jnp.pi * y) * jnp.cos(0.5 * jnp.pi * z)
        exact_z = -0.5 * jnp.pi * jnp.sin(jnp.pi * y) * jnp.sin(0.5 * jnp.pi * z)

        grad_y, grad_z = gradient_scalar(field, mesh)
        sl = (slice(2, -2), slice(2, -2))
        errors_y.append(float(jnp.sqrt(jnp.mean((grad_y[sl] - exact_y[sl]) ** 2))))
        errors_z.append(float(jnp.sqrt(jnp.mean((grad_z[sl] - exact_z[sl]) ** 2))))
        spacings.append(float(jnp.max(mesh.dy)))

    order_y = jnp.log(errors_y[0] / errors_y[1]) / jnp.log(spacings[0] / spacings[1])
    order_z = jnp.log(errors_z[0] / errors_z[1]) / jnp.log(spacings[0] / spacings[1])
    assert float(order_y) > 1.8
    assert float(order_z) > 1.8


def test_laplacian_manufactured_solution_and_masking_are_consistent():
    mesh = generate_rect_duct_mesh(width=2.0, height=2.0, ny=48, nz=48)
    y, z = jnp.meshgrid(mesh.y_centers, mesh.z_centers, indexing="ij")
    field = jnp.sin(jnp.pi * y) * jnp.sin(jnp.pi * z)
    exact = -2.0 * (jnp.pi**2) * field

    full_lap = laplacian_scalar(field, mesh)
    sl = (slice(4, -4), slice(4, -4))
    interior_error = float(jnp.sqrt(jnp.mean((full_lap[sl] - exact[sl]) ** 2)))
    assert interior_error < 0.15

    mask = jnp.ones(mesh.yz_shape, dtype=bool)
    mask = mask.at[:4, :].set(False)
    mask = mask.at[-4:, :].set(False)
    mask = mask.at[:, :4].set(False)
    mask = mask.at[:, -4:].set(False)

    lap = laplacian_scalar(field, mesh, mask=mask)
    assert jnp.allclose(lap[~mask], 0.0)
