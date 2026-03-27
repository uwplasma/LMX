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
    west = jnp.pad(field[:-1, :], ((1, 0), (0, 0)))
    east = jnp.pad(field[1:, :], ((0, 1), (0, 0)))
    south = jnp.pad(field[:, :-1], ((0, 0), (1, 0)))
    north = jnp.pad(field[:, 1:], ((0, 0), (0, 1)))
    dy = _broadcast_spacing_y(mesh)
    dz = _broadcast_spacing_z(mesh)
    lap = (east - 2.0 * field + west) / (dy**2) + (north - 2.0 * field + south) / (dz**2)
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
