from __future__ import annotations

import jax.numpy as jnp

from .mesh import StructuredMesh


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
    dy = _broadcast_spacing_y(mesh)
    dz = _broadcast_spacing_z(mesh)
    fy = jnp.zeros_like(field)
    fz = jnp.zeros_like(field)
    fy = fy.at[1:-1, :].set((field[2:, :] - field[:-2, :]) / (mesh.y_centers[2:, None] - mesh.y_centers[:-2, None]))
    fz = fz.at[:, 1:-1].set((field[:, 2:] - field[:, :-2]) / (mesh.z_centers[None, 2:] - mesh.z_centers[None, :-2]))
    fy = fy.at[0, :].set((field[1, :] - field[0, :]) / dy[0, :])
    fy = fy.at[-1, :].set((field[-1, :] - field[-2, :]) / dy[-1, :])
    fz = fz.at[:, 0].set((field[:, 1] - field[:, 0]) / dz[:, 0])
    fz = fz.at[:, -1].set((field[:, -1] - field[:, -2]) / dz[:, -1])
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
