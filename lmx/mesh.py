from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin

import jax.numpy as jnp


@dataclass(frozen=True)
class StructuredMesh:
    x_faces: jnp.ndarray
    y_faces: jnp.ndarray
    z_faces: jnp.ndarray
    geometry: str = "rect_duct"
    point_coordinates: jnp.ndarray | None = None
    fluid_mask: jnp.ndarray | None = None
    sigma: jnp.ndarray | None = None

    @property
    def nx(self) -> int:
        return int(self.x_faces.size - 1)

    @property
    def ny(self) -> int:
        return int(self.y_faces.size - 1)

    @property
    def nz(self) -> int:
        return int(self.z_faces.size - 1)

    @property
    def x_centers(self) -> jnp.ndarray:
        return 0.5 * (self.x_faces[:-1] + self.x_faces[1:])

    @property
    def y_centers(self) -> jnp.ndarray:
        return 0.5 * (self.y_faces[:-1] + self.y_faces[1:])

    @property
    def z_centers(self) -> jnp.ndarray:
        return 0.5 * (self.z_faces[:-1] + self.z_faces[1:])

    @property
    def dx(self) -> jnp.ndarray:
        return jnp.diff(self.x_faces)

    @property
    def dy(self) -> jnp.ndarray:
        return jnp.diff(self.y_faces)

    @property
    def dz(self) -> jnp.ndarray:
        return jnp.diff(self.z_faces)

    @property
    def yz_shape(self) -> tuple[int, int]:
        return (self.ny, self.nz)


def _clustered_segment(start: float, stop: float, count: int, beta: float = 2.5) -> jnp.ndarray:
    if count <= 1:
        return jnp.asarray([start, stop], dtype=float)
    s = jnp.linspace(-1.0, 1.0, count + 1)
    mapped = jnp.tanh(beta * s) / jnp.tanh(beta)
    scaled = 0.5 * (mapped + 1.0)
    return start + (stop - start) * scaled


def generate_rect_duct_mesh(
    width: float,
    height: float,
    length: float = 1.0,
    nx: int = 1,
    ny: int = 64,
    nz: int = 64,
) -> StructuredMesh:
    x_faces = jnp.linspace(0.0, length, nx + 1)
    y_faces = jnp.linspace(-0.5 * width, 0.5 * width, ny + 1)
    z_faces = jnp.linspace(-0.5 * height, 0.5 * height, nz + 1)
    return StructuredMesh(x_faces=x_faces, y_faces=y_faces, z_faces=z_faces, geometry="rect_duct")


def generate_layered_duct_mesh(
    width: float,
    height: float,
    length: float = 1.0,
    nx: int = 1,
    ny: int = 64,
    nz: int = 64,
    wall_thickness: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    wall_cells: tuple[int, int, int, int] = (0, 0, 0, 0),
    target_ha: float | None = None,
) -> StructuredMesh:
    left_t, right_t, bottom_t, top_t = wall_thickness
    left_c, right_c, bottom_c, top_c = wall_cells

    fluid_y = _clustered_segment(-0.5 * width, 0.5 * width, ny, beta=3.0 if target_ha else 2.0)
    fluid_z = _clustered_segment(-0.5 * height, 0.5 * height, nz, beta=3.0 if target_ha else 2.0)

    if left_c:
        left_faces = jnp.linspace(-0.5 * width - left_t, -0.5 * width, left_c + 1)
        fluid_y = jnp.concatenate([left_faces[:-1], fluid_y])
    if right_c:
        right_faces = jnp.linspace(0.5 * width, 0.5 * width + right_t, right_c + 1)
        fluid_y = jnp.concatenate([fluid_y, right_faces[1:]])
    if bottom_c:
        bottom_faces = jnp.linspace(-0.5 * height - bottom_t, -0.5 * height, bottom_c + 1)
        fluid_z = jnp.concatenate([bottom_faces[:-1], fluid_z])
    if top_c:
        top_faces = jnp.linspace(0.5 * height, 0.5 * height + top_t, top_c + 1)
        fluid_z = jnp.concatenate([fluid_z, top_faces[1:]])

    x_faces = jnp.linspace(0.0, length, nx + 1)
    y_faces = jnp.asarray(fluid_y)
    z_faces = jnp.asarray(fluid_z)

    yc, zc = jnp.meshgrid(0.5 * (y_faces[:-1] + y_faces[1:]), 0.5 * (z_faces[:-1] + z_faces[1:]), indexing="ij")
    fluid_mask = (jnp.abs(yc) <= 0.5 * width) & (jnp.abs(zc) <= 0.5 * height)
    return StructuredMesh(
        x_faces=x_faces,
        y_faces=y_faces,
        z_faces=z_faces,
        geometry="layered_duct",
        fluid_mask=fluid_mask,
    )


def generate_pipe_ogrid_mesh(
    radius: float,
    length: float = 1.0,
    nx: int = 8,
    nr: int = 24,
    ntheta: int = 64,
) -> StructuredMesh:
    x_faces = jnp.linspace(0.0, length, nx + 1)
    r_faces = _clustered_segment(0.0, radius, nr, beta=2.0)
    theta_faces = jnp.linspace(0.0, 2.0 * pi, ntheta + 1)
    y_faces = r_faces
    z_faces = theta_faces

    points = []
    for x in x_faces:
        for r in r_faces:
            for theta in theta_faces:
                points.append((float(x), float(r * cos(theta)), float(r * sin(theta))))
    point_coordinates = jnp.asarray(points).reshape((nx + 1, nr + 1, ntheta + 1, 3))
    return StructuredMesh(
        x_faces=x_faces,
        y_faces=y_faces,
        z_faces=z_faces,
        geometry="pipe_ogrid",
        point_coordinates=point_coordinates,
    )
