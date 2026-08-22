from __future__ import annotations

import jax.numpy as jnp

from .mesh import StructuredMesh


def apply_five_point_operator(
    diagonal: jnp.ndarray,
    west: jnp.ndarray,
    east: jnp.ndarray,
    south: jnp.ndarray,
    north: jnp.ndarray,
    field: jnp.ndarray,
) -> jnp.ndarray:
    west_field = jnp.pad(field[:-1, :], ((1, 0), (0, 0)))
    east_field = jnp.pad(field[1:, :], ((0, 1), (0, 0)))
    south_field = jnp.pad(field[:, :-1], ((0, 0), (1, 0)))
    north_field = jnp.pad(field[:, 1:], ((0, 0), (0, 1)))
    return (
        diagonal * field
        - west * west_field
        - east * east_field
        - south * south_field
        - north * north_field
    )


def five_point_residual_norm(
    diagonal: jnp.ndarray,
    west: jnp.ndarray,
    east: jnp.ndarray,
    south: jnp.ndarray,
    north: jnp.ndarray,
    rhs: jnp.ndarray,
    field: jnp.ndarray,
) -> jnp.ndarray:
    applied = apply_five_point_operator(diagonal, west, east, south, north, field)
    numerator = jnp.max(jnp.abs(applied - rhs))
    scale = jnp.maximum(jnp.max(jnp.abs(applied)), jnp.max(jnp.abs(rhs)))
    return numerator / jnp.maximum(scale, 1e-12)


def apply_poisson_operator(
    diagonal: jnp.ndarray,
    west: jnp.ndarray,
    east: jnp.ndarray,
    south: jnp.ndarray,
    north: jnp.ndarray,
    phi: jnp.ndarray,
    anchor: tuple[int, int],
) -> jnp.ndarray:
    projected = phi.at[anchor].set(0.0)
    west_phi = jnp.pad(projected[:-1, :], ((1, 0), (0, 0)))
    east_phi = jnp.pad(projected[1:, :], ((0, 1), (0, 0)))
    south_phi = jnp.pad(projected[:, :-1], ((0, 0), (1, 0)))
    north_phi = jnp.pad(projected[:, 1:], ((0, 0), (0, 1)))
    matrix_phi = (
        diagonal * projected
        - west * west_phi
        - east * east_phi
        - south * south_phi
        - north * north_phi
    )
    return matrix_phi.at[anchor].set(phi[anchor])


def poisson_residual_norm(
    diagonal: jnp.ndarray,
    west: jnp.ndarray,
    east: jnp.ndarray,
    south: jnp.ndarray,
    north: jnp.ndarray,
    rhs: jnp.ndarray,
    phi: jnp.ndarray,
    anchor: tuple[int, int],
) -> jnp.ndarray:
    rhs_masked = rhs.at[anchor].set(0.0)
    matrix_phi = apply_poisson_operator(diagonal, west, east, south, north, phi, anchor)
    numerator = jnp.max(jnp.abs(matrix_phi - rhs_masked))
    scale = jnp.maximum(jnp.max(jnp.abs(matrix_phi)), jnp.max(jnp.abs(rhs_masked)))
    return numerator / jnp.maximum(scale, 1e-12)


def _broadcast_spacing_y(mesh: StructuredMesh) -> jnp.ndarray:
    return mesh.dy[:, None]


def _broadcast_spacing_z(mesh: StructuredMesh) -> jnp.ndarray:
    return mesh.dz[None, :]


def center_coordinates(mesh: StructuredMesh) -> tuple[jnp.ndarray, jnp.ndarray]:
    return jnp.meshgrid(mesh.y_centers, mesh.z_centers, indexing="ij")


def face_average_x(field: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * (field[:-1, :] + field[1:, :])


def face_average_z(field: jnp.ndarray) -> jnp.ndarray:
    return 0.5 * (field[:, :-1] + field[:, 1:])


def center_spacing_y(mesh: StructuredMesh) -> jnp.ndarray:
    centers = mesh.y_centers
    if centers.size <= 1:
        return jnp.ones((0,), dtype=centers.dtype)
    return centers[1:] - centers[:-1]


def center_spacing_z(mesh: StructuredMesh) -> jnp.ndarray:
    centers = mesh.z_centers
    if centers.size <= 1:
        return jnp.ones((0,), dtype=centers.dtype)
    return centers[1:] - centers[:-1]


def gradient_scalar(field: jnp.ndarray, mesh: StructuredMesh) -> tuple[jnp.ndarray, jnp.ndarray]:
    y_centers = mesh.y_centers.astype(field.dtype)
    z_centers = mesh.z_centers.astype(field.dtype)
    delta_y = center_spacing_y(mesh).astype(field.dtype)
    delta_z = center_spacing_z(mesh).astype(field.dtype)
    fy = jnp.zeros_like(field)
    fz = jnp.zeros_like(field)
    fy = fy.at[1:-1, :].set((field[2:, :] - field[:-2, :]) / (y_centers[2:, None] - y_centers[:-2, None]))
    fz = fz.at[:, 1:-1].set((field[:, 2:] - field[:, :-2]) / (z_centers[None, 2:] - z_centers[None, :-2]))
    if field.shape[0] > 1:
        fy = fy.at[0, :].set((field[1, :] - field[0, :]) / delta_y[0])
        fy = fy.at[-1, :].set((field[-1, :] - field[-2, :]) / delta_y[-1])
    if field.shape[1] > 1:
        fz = fz.at[:, 0].set((field[:, 1] - field[:, 0]) / delta_z[0])
        fz = fz.at[:, -1].set((field[:, -1] - field[:, -2]) / delta_z[-1])
    return fy, fz


def laplacian_scalar(field: jnp.ndarray, mesh: StructuredMesh, mask: jnp.ndarray | None = None) -> jnp.ndarray:
    fluid_mask = jnp.ones_like(field, dtype=bool) if mask is None else mask
    dy = mesh.dy[:, None]
    dz = mesh.dz[None, :]
    delta_y = center_spacing_y(mesh)
    delta_z = center_spacing_z(mesh)

    west_connected = fluid_mask & jnp.pad(fluid_mask[:-1, :], ((1, 0), (0, 0)))
    east_connected = fluid_mask & jnp.pad(fluid_mask[1:, :], ((0, 1), (0, 0)))
    south_connected = fluid_mask & jnp.pad(fluid_mask[:, :-1], ((0, 0), (1, 0)))
    north_connected = fluid_mask & jnp.pad(fluid_mask[:, 1:], ((0, 0), (0, 1)))

    west_value = jnp.where(west_connected, jnp.pad(field[:-1, :], ((1, 0), (0, 0))), 0.0)
    east_value = jnp.where(east_connected, jnp.pad(field[1:, :], ((0, 1), (0, 0))), 0.0)
    south_value = jnp.where(south_connected, jnp.pad(field[:, :-1], ((0, 0), (1, 0))), 0.0)
    north_value = jnp.where(north_connected, jnp.pad(field[:, 1:], ((0, 0), (0, 1))), 0.0)

    west_distance = jnp.where(west_connected, jnp.pad(delta_y[:, None], ((1, 0), (0, 0))), 0.5 * dy)
    east_distance = jnp.where(east_connected, jnp.pad(delta_y[:, None], ((0, 1), (0, 0))), 0.5 * dy)
    south_distance = jnp.where(south_connected, jnp.pad(delta_z[None, :], ((0, 0), (1, 0))), 0.5 * dz)
    north_distance = jnp.where(north_connected, jnp.pad(delta_z[None, :], ((0, 0), (0, 1))), 0.5 * dz)

    flux_y = ((east_value - field) / jnp.maximum(east_distance, 1e-12) - (field - west_value) / jnp.maximum(west_distance, 1e-12)) / jnp.maximum(dy, 1e-12)
    flux_z = ((north_value - field) / jnp.maximum(north_distance, 1e-12) - (field - south_value) / jnp.maximum(south_distance, 1e-12)) / jnp.maximum(dz, 1e-12)
    lap = flux_y + flux_z
    if mask is not None:
        lap = jnp.where(mask, lap, 0.0)
    return lap


def divergence_flux(flux_y: jnp.ndarray, flux_z: jnp.ndarray, mesh: StructuredMesh) -> jnp.ndarray:
    dy = _broadcast_spacing_y(mesh)
    dz = _broadcast_spacing_z(mesh)
    return flux_y / dy + flux_z / dz


def face_divergence(
    face_flux_y: jnp.ndarray,
    face_flux_z: jnp.ndarray,
    mesh: StructuredMesh,
) -> jnp.ndarray:
    dy = _broadcast_spacing_y(mesh)
    dz = _broadcast_spacing_z(mesh)
    diff_y = (face_flux_y[1:, :] - face_flux_y[:-1, :]) / dy
    diff_z = (face_flux_z[:, 1:] - face_flux_z[:, :-1]) / dz
    return diff_y + diff_z
