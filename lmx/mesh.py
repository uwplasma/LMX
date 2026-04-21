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


def _geometric_widths(total_width: float, cells: int, ratio: float) -> jnp.ndarray:
    if cells <= 0:
        return jnp.asarray([], dtype=float)
    if cells == 1 or abs(ratio - 1.0) < 1e-12:
        return jnp.full((cells,), total_width / max(cells, 1), dtype=float)
    scale = total_width * (ratio - 1.0) / (ratio**cells - 1.0)
    indices = jnp.arange(cells, dtype=float)
    return scale * ratio**indices


def _symmetric_boundary_layer_segment(
    start: float,
    stop: float,
    count: int,
    *,
    layer_thickness: float,
    layer_cells: int,
    growth_ratio: float = 1.35,
) -> jnp.ndarray:
    length = stop - start
    if count <= 1 or layer_cells <= 0 or layer_thickness <= 0.0:
        return jnp.linspace(start, stop, count + 1)
    capped_layer_thickness = min(layer_thickness, 0.24 * length)
    capped_layer_cells = min(layer_cells, max(1, (count - 2) // 2))
    if capped_layer_cells <= 0:
        return jnp.linspace(start, stop, count + 1)
    core_cells = count - 2 * capped_layer_cells
    if core_cells <= 0:
        return _clustered_segment(start, stop, count, beta=1.5)
    left_widths = _geometric_widths(capped_layer_thickness, capped_layer_cells, growth_ratio)
    right_widths = left_widths[::-1]
    core_width = (length - 2.0 * capped_layer_thickness) / core_cells
    core_widths = jnp.full((core_cells,), core_width, dtype=float)
    widths = jnp.concatenate([left_widths, core_widths, right_widths])
    faces = start + jnp.concatenate([jnp.asarray([0.0], dtype=float), jnp.cumsum(widths)])
    return faces


def _segmented_boundary_layer_segment(
    start: float,
    stop: float,
    count: int,
    *,
    wall_layer_width: float,
    wall_fraction: float = 0.10,
    expansion_fraction: float = 0.25,
    core_fraction: float = 0.30,
    min_wall_cells: int = 5,
    min_expansion_cells: int = 4,
    growth_ratio: float = 10.0,
) -> jnp.ndarray:
    length = stop - start
    if count <= 1 or wall_layer_width <= 0.0:
        return jnp.linspace(start, stop, count + 1)

    wall_cells = max(min_wall_cells, int(round(wall_fraction * count)))
    expansion_cells = max(min_expansion_cells, int(round(expansion_fraction * count)))
    core_cells = int(round(core_fraction * count))

    total_assigned = 2 * wall_cells + 2 * expansion_cells + core_cells
    if total_assigned > count:
        overflow = total_assigned - count
        reducible_expansion = max(0, expansion_cells - min_expansion_cells)
        reduce_expansion = min(overflow // 2 + overflow % 2, reducible_expansion)
        expansion_cells -= reduce_expansion
        overflow -= 2 * reduce_expansion
        reducible_wall = max(0, wall_cells - max(3, min_wall_cells - 1))
        reduce_wall = min(overflow // 2 + overflow % 2, reducible_wall)
        wall_cells -= reduce_wall
        overflow -= 2 * reduce_wall
        core_cells = max(2, count - 2 * wall_cells - 2 * expansion_cells)
    else:
        core_cells = count - 2 * wall_cells - 2 * expansion_cells

    if core_cells <= 0 or wall_cells <= 0:
        return _clustered_segment(start, stop, count, beta=1.8)

    max_wall_width = 0.18 * length
    wall_width = min(wall_layer_width, max_wall_width)
    target_core_width = max(core_fraction * length, 0.20 * length)
    expansion_width = max(0.0, 0.5 * (length - target_core_width) - wall_width)
    core_width = length - 2.0 * wall_width - 2.0 * expansion_width
    if core_width <= 0.0:
        target_core_width = 0.12 * length
        expansion_width = max(0.0, 0.5 * (length - target_core_width) - wall_width)
        core_width = length - 2.0 * wall_width - 2.0 * expansion_width
    if core_width <= 0.0:
        return _symmetric_boundary_layer_segment(
            start,
            stop,
            count,
            layer_thickness=wall_width,
            layer_cells=wall_cells,
            growth_ratio=min(growth_ratio, 1.8),
        )

    wall_widths = jnp.full((wall_cells,), wall_width / wall_cells, dtype=float)
    if expansion_cells <= 1 or expansion_width <= 0.0:
        left_expansion = jnp.full((expansion_cells,), expansion_width / max(expansion_cells, 1), dtype=float)
    else:
        # Treat growth_ratio as the total expansion across the segment, not the
        # per-cell ratio. A large per-cell ratio on fine meshes creates
        # numerically zero-width cells near the wall.
        per_cell_ratio = max(growth_ratio, 1.0) ** (1.0 / max(expansion_cells - 1, 1))
        left_expansion = _geometric_widths(expansion_width, expansion_cells, per_cell_ratio)
    right_expansion = left_expansion[::-1]
    core_widths = jnp.full((core_cells,), core_width / core_cells, dtype=float)
    widths = jnp.concatenate([wall_widths, left_expansion, core_widths, right_expansion, wall_widths])
    faces = start + jnp.concatenate([jnp.asarray([0.0], dtype=float), jnp.cumsum(widths)])
    return faces


def generate_rect_duct_mesh(
    width: float,
    height: float,
    length: float = 1.0,
    nx: int = 1,
    ny: int = 64,
    nz: int = 64,
    target_ha: float | None = None,
    magnetic_axis: str | None = None,
) -> StructuredMesh:
    x_faces = jnp.linspace(0.0, length, nx + 1)
    if target_ha and target_ha > 0.0:
        side_y = 0.5 * width / jnp.sqrt(target_ha)
        side_z = 0.5 * height / jnp.sqrt(target_ha)
        hartmann_y = 0.5 * width / target_ha
        hartmann_z = 0.5 * height / target_ha
        if magnetic_axis == "y":
            y_layer_thickness = hartmann_y
            z_layer_thickness = side_z
        elif magnetic_axis == "z":
            y_layer_thickness = side_y
            z_layer_thickness = hartmann_z
        else:
            y_layer_thickness = hartmann_y
            z_layer_thickness = hartmann_z
        if target_ha >= 100.0:
            y_faces = _segmented_boundary_layer_segment(
                -0.5 * width,
                0.5 * width,
                ny,
                wall_layer_width=float(y_layer_thickness),
                wall_fraction=0.10,
                expansion_fraction=0.25,
                core_fraction=0.30,
                min_wall_cells=max(5, int(round(0.10 * ny))),
                min_expansion_cells=max(4, int(round(0.20 * ny))),
                growth_ratio=10.0,
            )
            z_faces = _segmented_boundary_layer_segment(
                -0.5 * height,
                0.5 * height,
                nz,
                wall_layer_width=float(z_layer_thickness),
                wall_fraction=0.10,
                expansion_fraction=0.25,
                core_fraction=0.30,
                min_wall_cells=max(5, int(round(0.10 * nz))),
                min_expansion_cells=max(4, int(round(0.20 * nz))),
                growth_ratio=10.0,
            )
        else:
            y_faces = _symmetric_boundary_layer_segment(
                -0.5 * width,
                0.5 * width,
                ny,
                layer_thickness=float(y_layer_thickness),
                layer_cells=max(4, int(round((0.16 if magnetic_axis == "z" else 0.12) * ny))),
                growth_ratio=1.35,
            )
            z_faces = _symmetric_boundary_layer_segment(
                -0.5 * height,
                0.5 * height,
                nz,
                layer_thickness=float(z_layer_thickness),
                layer_cells=max(4, int(round((0.16 if magnetic_axis == "y" else 0.12) * nz))),
                growth_ratio=1.35,
            )
    else:
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
    magnetic_axis: str | None = None,
) -> StructuredMesh:
    left_t, right_t, bottom_t, top_t = wall_thickness
    left_c, right_c, bottom_c, top_c = wall_cells

    if target_ha and target_ha > 0.0:
        fluid_mesh = generate_rect_duct_mesh(
            width=width,
            height=height,
            length=length,
            nx=max(nx, 1),
            ny=ny,
            nz=nz,
            target_ha=target_ha,
            magnetic_axis=magnetic_axis,
        )
        fluid_y = fluid_mesh.y_faces
        fluid_z = fluid_mesh.z_faces
    else:
        fluid_y = _clustered_segment(-0.5 * width, 0.5 * width, ny, beta=2.0)
        fluid_z = _clustered_segment(-0.5 * height, 0.5 * height, nz, beta=2.0)

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
