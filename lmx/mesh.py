from __future__ import annotations

from dataclasses import dataclass
from math import cos, pi, sin
from typing import Sequence

import jax.numpy as jnp
import numpy as np

from .wall_models import WallLayer


@dataclass(frozen=True)
class StructuredMesh:
    x_faces: jnp.ndarray
    y_faces: jnp.ndarray
    z_faces: jnp.ndarray
    geometry: str = "rect_duct"
    point_coordinates: jnp.ndarray | None = None
    fluid_mask: jnp.ndarray | None = None
    sigma: jnp.ndarray | None = None
    region_ids: jnp.ndarray | None = None
    region_names: tuple[str, ...] = ()

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


def _validated_faces(faces: jnp.ndarray, *, name: str) -> jnp.ndarray:
    values = jnp.asarray(faces, dtype=float)
    if values.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if values.size < 2:
        raise ValueError(f"{name} must contain at least two faces")
    if bool(jnp.any(jnp.diff(values) <= 0.0)):
        raise ValueError(f"{name} must be strictly increasing")
    return values


def generate_rect_duct_mesh_from_faces(
    *,
    y_faces: jnp.ndarray,
    z_faces: jnp.ndarray,
    length: float = 1.0,
    nx: int = 1,
) -> StructuredMesh:
    """Build a rectangular duct mesh from explicit cross-section faces."""

    y = _validated_faces(y_faces, name="y_faces")
    z = _validated_faces(z_faces, name="z_faces")
    x_faces = jnp.linspace(0.0, length, nx + 1)
    return StructuredMesh(x_faces=x_faces, y_faces=y, z_faces=z, geometry="rect_duct")


def generate_layered_duct_mesh_from_fluid_faces(
    *,
    fluid_y_faces: jnp.ndarray,
    fluid_z_faces: jnp.ndarray,
    width: float,
    height: float,
    length: float = 1.0,
    nx: int = 1,
    wall_thickness: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
    wall_cells: tuple[int, int, int, int] = (0, 0, 0, 0),
) -> StructuredMesh:
    """Build a layered duct around explicit fluid-region faces."""

    fluid_y = _validated_faces(fluid_y_faces, name="fluid_y_faces")
    fluid_z = _validated_faces(fluid_z_faces, name="fluid_z_faces")
    left_t, right_t, bottom_t, top_t = wall_thickness
    left_c, right_c, bottom_c, top_c = wall_cells
    if left_c:
        left_faces = jnp.linspace(
            float(fluid_y[0]) - left_t, float(fluid_y[0]), left_c + 1
        )
        fluid_y = jnp.concatenate([left_faces[:-1], fluid_y])
    if right_c:
        right_faces = jnp.linspace(
            float(fluid_y[-1]), float(fluid_y[-1]) + right_t, right_c + 1
        )
        fluid_y = jnp.concatenate([fluid_y, right_faces[1:]])
    if bottom_c:
        bottom_faces = jnp.linspace(
            float(fluid_z[0]) - bottom_t, float(fluid_z[0]), bottom_c + 1
        )
        fluid_z = jnp.concatenate([bottom_faces[:-1], fluid_z])
    if top_c:
        top_faces = jnp.linspace(
            float(fluid_z[-1]), float(fluid_z[-1]) + top_t, top_c + 1
        )
        fluid_z = jnp.concatenate([fluid_z, top_faces[1:]])

    x_faces = jnp.linspace(0.0, length, nx + 1)
    yc, zc = jnp.meshgrid(
        0.5 * (fluid_y[:-1] + fluid_y[1:]),
        0.5 * (fluid_z[:-1] + fluid_z[1:]),
        indexing="ij",
    )
    half_width = 0.5 * width
    half_height = 0.5 * height
    fluid_mask = (jnp.abs(yc) <= half_width) & (jnp.abs(zc) <= half_height)
    return StructuredMesh(
        x_faces=x_faces,
        y_faces=fluid_y,
        z_faces=fluid_z,
        geometry="layered_duct",
        fluid_mask=fluid_mask,
    )


def _clustered_segment(
    start: float, stop: float, count: int, beta: float = 2.5
) -> jnp.ndarray:
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
    left_widths = _geometric_widths(
        capped_layer_thickness, capped_layer_cells, growth_ratio
    )
    right_widths = left_widths[::-1]
    core_width = (length - 2.0 * capped_layer_thickness) / core_cells
    core_widths = jnp.full((core_cells,), core_width, dtype=float)
    widths = jnp.concatenate([left_widths, core_widths, right_widths])
    faces = start + jnp.concatenate(
        [jnp.asarray([0.0], dtype=float), jnp.cumsum(widths)]
    )
    return faces


def _smooth_boundary_layer_segment(
    start: float,
    stop: float,
    count: int,
    *,
    layer_thickness: float,
    layer_cells: int,
) -> jnp.ndarray:
    """Build a symmetric geometric mesh with an exact wall-layer cell count.

    Unlike a piecewise wall/expansion/core construction, every adjacent cell
    width changes by one geometric ratio. This avoids the large spacing jump
    immediately outside thin high-Ha layers while preserving the requested
    number of cells inside the physical layer thickness.
    """

    length = float(stop - start)
    if count <= 1 or layer_thickness <= 0.0 or layer_cells <= 0:
        return jnp.linspace(start, stop, count + 1)
    half_cells = count // 2
    cells_in_layer = min(int(layer_cells), half_cells)
    half_length = 0.5 * length
    if cells_in_layer <= 0 or layer_thickness >= half_length:
        return jnp.linspace(start, stop, count + 1)
    has_center = count % 2

    def geometric_sum(ratio: float, cells: int) -> float:
        if abs(ratio - 1.0) < 1.0e-12:
            return float(cells)
        return (ratio**cells - 1.0) / (ratio - 1.0)

    target_ratio = half_length / float(layer_thickness)

    def span_ratio(ratio: float) -> float:
        half_span = geometric_sum(ratio, half_cells)
        if has_center:
            half_span += 0.5 * ratio**half_cells
        return half_span / geometric_sum(ratio, cells_in_layer)

    if target_ratio <= span_ratio(1.0):
        return jnp.linspace(start, stop, count + 1)
    lower = 1.0
    upper = 1.1
    while span_ratio(upper) < target_ratio:
        upper *= 1.25
        if upper > 10.0:
            return jnp.linspace(start, stop, count + 1)
    for _ in range(80):
        middle = 0.5 * (lower + upper)
        if span_ratio(middle) < target_ratio:
            lower = middle
        else:
            upper = middle
    ratio = 0.5 * (lower + upper)
    first_width = float(layer_thickness) / geometric_sum(ratio, cells_in_layer)
    left = first_width * ratio ** jnp.arange(half_cells, dtype=float)
    pieces = [left]
    if has_center:
        pieces.append(jnp.asarray([first_width * ratio**half_cells], dtype=float))
    pieces.append(left[::-1])
    widths = jnp.concatenate(pieces)
    # Remove only roundoff in the nonlinear ratio solve; relative spacing and
    # the exact layer allocation are otherwise unchanged.
    widths = widths * (length / jnp.sum(widths))
    return start + jnp.concatenate([jnp.asarray([0.0]), jnp.cumsum(widths)])


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
    """Generate a rectangular duct mesh with optional MHD layer clustering."""

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
            y_faces = _smooth_boundary_layer_segment(
                -0.5 * width,
                0.5 * width,
                ny,
                layer_thickness=float(y_layer_thickness),
                layer_cells=max(5, int(round(0.10 * ny))),
            )
            z_faces = _smooth_boundary_layer_segment(
                -0.5 * height,
                0.5 * height,
                nz,
                layer_thickness=float(z_layer_thickness),
                layer_cells=max(5, int(round(0.10 * nz))),
            )
        else:
            y_layer_cells = max(
                5, int(round((0.20 if magnetic_axis == "y" else 0.16) * ny))
            )
            z_layer_cells = max(
                5, int(round((0.20 if magnetic_axis == "z" else 0.16) * nz))
            )
            y_faces = _symmetric_boundary_layer_segment(
                -0.5 * width,
                0.5 * width,
                ny,
                layer_thickness=float(y_layer_thickness),
                layer_cells=y_layer_cells,
                growth_ratio=1.35,
            )
            z_faces = _symmetric_boundary_layer_segment(
                -0.5 * height,
                0.5 * height,
                nz,
                layer_thickness=float(z_layer_thickness),
                layer_cells=z_layer_cells,
                growth_ratio=1.35,
            )
    else:
        y_faces = jnp.linspace(-0.5 * width, 0.5 * width, ny + 1)
        z_faces = jnp.linspace(-0.5 * height, 0.5 * height, nz + 1)
    return StructuredMesh(
        x_faces=x_faces, y_faces=y_faces, z_faces=z_faces, geometry="rect_duct"
    )


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
    """Generate a rectangular fluid mesh plus explicit surrounding wall cells."""

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
        bottom_faces = jnp.linspace(
            -0.5 * height - bottom_t, -0.5 * height, bottom_c + 1
        )
        fluid_z = jnp.concatenate([bottom_faces[:-1], fluid_z])
    if top_c:
        top_faces = jnp.linspace(0.5 * height, 0.5 * height + top_t, top_c + 1)
        fluid_z = jnp.concatenate([fluid_z, top_faces[1:]])

    x_faces = jnp.linspace(0.0, length, nx + 1)
    y_faces = jnp.asarray(fluid_y)
    z_faces = jnp.asarray(fluid_z)

    yc, zc = jnp.meshgrid(
        0.5 * (y_faces[:-1] + y_faces[1:]),
        0.5 * (z_faces[:-1] + z_faces[1:]),
        indexing="ij",
    )
    fluid_mask = (jnp.abs(yc) <= 0.5 * width) & (jnp.abs(zc) <= 0.5 * height)
    return StructuredMesh(
        x_faces=x_faces,
        y_faces=y_faces,
        z_faces=z_faces,
        geometry="layered_duct",
        fluid_mask=fluid_mask,
    )


_WALL_SIDES = ("left", "right", "bottom", "top")


def generate_multilayer_duct_mesh(
    *,
    width: float,
    height: float,
    length: float = 1.0,
    nx: int = 1,
    ny: int = 64,
    nz: int = 64,
    wall_layers: dict[str, Sequence[WallLayer]] | None = None,
    fluid_conductivity: float = 1.0,
    target_ha: float | None = None,
    magnetic_axis: str | None = None,
) -> StructuredMesh:
    """Build a rectangular duct with explicit nested wall-layer cells.

    ``wall_layers`` maps ``left``, ``right``, ``bottom``, and/or ``top`` to
    ``WallLayer`` sequences ordered from the fluid outward. Faces are inserted
    at every layer interface, and ``mesh.sigma``/``mesh.region_ids`` are filled
    so downstream QA and material-field construction can use the explicit
    multilayer geometry directly.
    """

    if width <= 0.0 or height <= 0.0 or length <= 0.0:
        raise ValueError("width, height, and length must be positive")
    if ny <= 0 or nz <= 0 or nx <= 0:
        raise ValueError("nx, ny, and nz must be positive")
    if fluid_conductivity <= 0.0:
        raise ValueError("fluid_conductivity must be positive")
    layers_by_side = _normalized_wall_layers(wall_layers or {})
    if target_ha and target_ha > 0.0:
        fluid_mesh = generate_rect_duct_mesh(
            width=width,
            height=height,
            length=length,
            nx=nx,
            ny=ny,
            nz=nz,
            target_ha=target_ha,
            magnetic_axis=magnetic_axis,
        )
        y_faces = fluid_mesh.y_faces
        z_faces = fluid_mesh.z_faces
    else:
        y_faces = _clustered_segment(-0.5 * width, 0.5 * width, ny, beta=2.0)
        z_faces = _clustered_segment(-0.5 * height, 0.5 * height, nz, beta=2.0)

    left_faces = _negative_wall_faces(-0.5 * width, layers_by_side["left"])
    right_faces = _positive_wall_faces(0.5 * width, layers_by_side["right"])
    bottom_faces = _negative_wall_faces(-0.5 * height, layers_by_side["bottom"])
    top_faces = _positive_wall_faces(0.5 * height, layers_by_side["top"])
    if left_faces.size:
        y_faces = jnp.concatenate([left_faces[:-1], y_faces])
    if right_faces.size:
        y_faces = jnp.concatenate([y_faces, right_faces[1:]])
    if bottom_faces.size:
        z_faces = jnp.concatenate([bottom_faces[:-1], z_faces])
    if top_faces.size:
        z_faces = jnp.concatenate([z_faces, top_faces[1:]])

    x_faces = jnp.linspace(0.0, length, nx + 1)
    yc, zc = np.meshgrid(
        np.asarray(0.5 * (y_faces[:-1] + y_faces[1:]), dtype=float),
        np.asarray(0.5 * (z_faces[:-1] + z_faces[1:]), dtype=float),
        indexing="ij",
    )
    fluid_mask = (np.abs(yc) <= 0.5 * width) & (np.abs(zc) <= 0.5 * height)
    region_names, region_sigmas, region_ids = _multilayer_region_assignment(
        yc,
        zc,
        width=width,
        height=height,
        fluid_mask=fluid_mask,
        fluid_conductivity=fluid_conductivity,
        wall_layers=layers_by_side,
    )
    sigma = np.asarray(region_sigmas, dtype=float)[region_ids]
    return StructuredMesh(
        x_faces=x_faces,
        y_faces=jnp.asarray(y_faces),
        z_faces=jnp.asarray(z_faces),
        geometry="layered_duct",
        fluid_mask=jnp.asarray(fluid_mask),
        sigma=jnp.asarray(sigma, dtype=float),
        region_ids=jnp.asarray(region_ids, dtype=int),
        region_names=tuple(region_names),
    )


def _normalized_wall_layers(
    wall_layers: dict[str, Sequence[WallLayer]],
) -> dict[str, tuple[WallLayer, ...]]:
    normalized: dict[str, tuple[WallLayer, ...]] = {}
    for side in _WALL_SIDES:
        layers = tuple(wall_layers.get(side, ()))
        for layer in layers:
            if layer.thickness <= 0.0:
                raise ValueError(
                    f"{side} wall layer {layer.name!r} has non-positive thickness"
                )
            if layer.cells <= 0:
                raise ValueError(
                    f"{side} wall layer {layer.name!r} must have at least one cell"
                )
            if layer.conductivity < 0.0:
                raise ValueError(
                    f"{side} wall layer {layer.name!r} has negative conductivity"
                )
        normalized[side] = layers
    unknown = set(wall_layers) - set(_WALL_SIDES)
    if unknown:
        raise ValueError(f"unsupported wall side(s): {sorted(unknown)}")
    return normalized


def _negative_wall_faces(
    inner_boundary: float, layers: Sequence[WallLayer]
) -> jnp.ndarray:
    if not layers:
        return jnp.asarray([], dtype=float)
    total = sum(float(layer.thickness) for layer in layers)
    cursor = float(inner_boundary) - total
    segments = []
    for layer in reversed(layers):
        stop = cursor + float(layer.thickness)
        segment = jnp.linspace(cursor, stop, int(layer.cells) + 1)
        segments.append(segment if not segments else segment[1:])
        cursor = stop
    return jnp.concatenate(segments)


def _positive_wall_faces(
    inner_boundary: float, layers: Sequence[WallLayer]
) -> jnp.ndarray:
    if not layers:
        return jnp.asarray([], dtype=float)
    cursor = float(inner_boundary)
    segments = []
    for layer in layers:
        stop = cursor + float(layer.thickness)
        segment = jnp.linspace(cursor, stop, int(layer.cells) + 1)
        segments.append(segment if not segments else segment[1:])
        cursor = stop
    return jnp.concatenate(segments)


def _multilayer_region_assignment(
    yc: np.ndarray,
    zc: np.ndarray,
    *,
    width: float,
    height: float,
    fluid_mask: np.ndarray,
    fluid_conductivity: float,
    wall_layers: dict[str, tuple[WallLayer, ...]],
) -> tuple[list[str], list[float], np.ndarray]:
    region_names = ["fluid"]
    region_sigmas = [float(fluid_conductivity)]
    region_for_side_layer: dict[tuple[str, int], int] = {}
    for side in _WALL_SIDES:
        for index, layer in enumerate(wall_layers[side]):
            region_for_side_layer[(side, index)] = len(region_names)
            region_names.append(f"{side}:{layer.name}")
            region_sigmas.append(float(layer.conductivity))

    region_ids = np.zeros(yc.shape, dtype=int)
    distances = {
        "left": np.where(yc < -0.5 * width, -0.5 * width - yc, np.inf),
        "right": np.where(yc > 0.5 * width, yc - 0.5 * width, np.inf),
        "bottom": np.where(zc < -0.5 * height, -0.5 * height - zc, np.inf),
        "top": np.where(zc > 0.5 * height, zc - 0.5 * height, np.inf),
    }
    distance_stack = np.stack([distances[side] for side in _WALL_SIDES], axis=0)
    side_index = np.argmin(distance_stack, axis=0)
    for side_number, side in enumerate(_WALL_SIDES):
        side_mask = (~fluid_mask) & (side_index == side_number)
        if not np.any(side_mask) or not wall_layers[side]:
            continue
        distance = distances[side]
        lower = 0.0
        for index, layer in enumerate(wall_layers[side]):
            upper = lower + float(layer.thickness)
            layer_mask = (
                side_mask & (distance > lower - 1.0e-12) & (distance <= upper + 1.0e-12)
            )
            region_ids[layer_mask] = region_for_side_layer[(side, index)]
            lower = upper
        region_ids[side_mask & (distance > lower)] = region_for_side_layer[
            (side, len(wall_layers[side]) - 1)
        ]
    return region_names, region_sigmas, region_ids


def generate_pipe_ogrid_mesh(
    radius: float,
    length: float = 1.0,
    nx: int = 8,
    nr: int = 24,
    ntheta: int = 64,
    wall_thickness: float = 0.0,
    wall_cells: int = 0,
    target_ha: float | None = None,
    hartmann_layer_cells: int | None = None,
) -> StructuredMesh:
    """Build a straight pipe O-grid, optionally including an annular wall.

    ``radius`` is always the fluid radius.  When an explicit wall is requested,
    ``nr`` continues to count fluid cells and ``wall_cells`` counts additional
    solid cells.  This convention prevents a wall refinement from silently
    coarsening the fluid domain.
    """

    if radius <= 0.0 or length <= 0.0:
        raise ValueError("radius and length must be positive")
    if nx <= 0 or nr <= 0 or ntheta <= 0:
        raise ValueError("nx, nr, and ntheta must be positive")
    if wall_thickness < 0.0 or wall_cells < 0:
        raise ValueError("wall_thickness and wall_cells cannot be negative")
    if (wall_thickness > 0.0) != (wall_cells > 0):
        raise ValueError("wall_thickness and wall_cells must be enabled together")

    x_faces = jnp.linspace(0.0, length, nx + 1)
    if target_ha is not None and target_ha > 0.0:
        layer_cells = (
            max(5, int(round(0.10 * nr)))
            if hartmann_layer_cells is None
            else hartmann_layer_cells
        )
        if layer_cells <= 0 or 2 * layer_cells >= nr:
            raise ValueError("hartmann_layer_cells must fit twice within nr")
        r_faces = _smooth_boundary_layer_segment(
            0.0,
            radius,
            nr,
            layer_thickness=radius / target_ha,
            layer_cells=layer_cells,
        )
    else:
        r_faces = _clustered_segment(0.0, radius, nr, beta=2.0)
    if wall_cells:
        wall_faces = jnp.linspace(radius, radius + wall_thickness, wall_cells + 1)
        r_faces = jnp.concatenate([r_faces, wall_faces[1:]])
    theta_faces = jnp.linspace(0.0, 2.0 * pi, ntheta + 1)
    y_faces = r_faces
    z_faces = theta_faces

    xx, rr, tt = jnp.meshgrid(x_faces, r_faces, theta_faces, indexing="ij")
    point_coordinates = jnp.stack([xx, rr * jnp.cos(tt), rr * jnp.sin(tt)], axis=-1)
    radial_centers = 0.5 * (r_faces[:-1] + r_faces[1:])
    fluid_mask = jnp.broadcast_to(
        (radial_centers <= radius + 1.0e-12)[:, None],
        (radial_centers.size, ntheta),
    )
    return StructuredMesh(
        x_faces=x_faces,
        y_faces=y_faces,
        z_faces=z_faces,
        geometry="pipe_ogrid",
        point_coordinates=point_coordinates,
        fluid_mask=fluid_mask,
    )


def generate_bent_pipe_mesh(
    *,
    tube_radius: float,
    bend_radius: float,
    bend_angle: float = 0.5 * pi,
    nx: int = 24,
    nr: int = 24,
    ntheta: int = 64,
) -> StructuredMesh:
    arc_length = bend_radius * bend_angle
    s_faces = jnp.linspace(0.0, arc_length, nx + 1)
    r_faces = _clustered_segment(0.0, tube_radius, nr, beta=2.0)
    theta_faces = jnp.linspace(0.0, 2.0 * pi, ntheta + 1)

    points = []
    for s in s_faces:
        phi = float(bend_angle * s / max(arc_length, 1.0e-12))
        center = jnp.asarray(
            [
                bend_radius * sin(phi),
                bend_radius * (1.0 - cos(phi)),
                0.0,
            ],
            dtype=float,
        )
        tangent = jnp.asarray([cos(phi), sin(phi), 0.0], dtype=float)
        normal = jnp.asarray([-sin(phi), cos(phi), 0.0], dtype=float)
        binormal = jnp.asarray([0.0, 0.0, 1.0], dtype=float)
        _ = tangent  # documents local frame construction for future solver work
        for r in r_faces:
            for theta in theta_faces:
                offset = (
                    float(r) * cos(theta) * normal + float(r) * sin(theta) * binormal
                )
                xyz = center + offset
                points.append((float(xyz[0]), float(xyz[1]), float(xyz[2])))

    point_coordinates = jnp.asarray(points).reshape((nx + 1, nr + 1, ntheta + 1, 3))
    return StructuredMesh(
        x_faces=s_faces,
        y_faces=r_faces,
        z_faces=theta_faces,
        geometry="bent_pipe",
        point_coordinates=point_coordinates,
    )


def generate_centerline_pipe_mesh(
    centerline: dict[str, np.ndarray | jnp.ndarray],
    *,
    tube_radius: float,
    nx: int | None = None,
    nr: int = 24,
    ntheta: int = 64,
    geometry: str = "centerline_pipe",
) -> StructuredMesh:
    """Build a mapped circular-pipe O-grid along an arbitrary 3D centerline.

    This mesh is intended for geometry QA, VTK export, and future mapped-pipe
    solver work. Existing extruded solver paths still use `pipe_ogrid` and the
    single-radius `bent_pipe` branch until the generalized centerline metrics
    are promoted into the operator stack.
    """

    if tube_radius <= 0.0:
        raise ValueError("tube_radius must be positive")
    station, coordinates = _validated_centerline_arrays(centerline)
    station_count = int(nx) + 1 if nx is not None else station.size
    if station_count < 3:
        raise ValueError("centerline pipe mesh requires at least three stations")
    s_faces = np.linspace(float(station[0]), float(station[-1]), station_count)
    center = np.column_stack(
        [np.interp(s_faces, station, coordinates[:, axis]) for axis in range(3)]
    )
    tangent = _centerline_tangent(center, s_faces)
    normal, binormal = _centerline_frames(tangent)

    r_faces = np.asarray(
        _clustered_segment(0.0, tube_radius, nr, beta=2.0), dtype=float
    )
    theta_faces = np.linspace(0.0, 2.0 * pi, ntheta + 1)
    points = np.empty((station_count, r_faces.size, theta_faces.size, 3), dtype=float)
    for station_index in range(station_count):
        for radial_index, radius in enumerate(r_faces):
            points[station_index, radial_index, :, :] = (
                center[station_index]
                + radius * np.cos(theta_faces)[:, None] * normal[station_index]
                + radius * np.sin(theta_faces)[:, None] * binormal[station_index]
            )

    return StructuredMesh(
        x_faces=jnp.asarray(s_faces),
        y_faces=jnp.asarray(r_faces),
        z_faces=jnp.asarray(theta_faces),
        geometry=geometry,
        point_coordinates=jnp.asarray(points),
    )


def centerline_pipe_mesh_quality_metrics(
    mesh: StructuredMesh,
) -> dict[str, float | int | bool | str]:
    """Return geometric quality metrics for a mapped centerline-pipe mesh."""

    if mesh.point_coordinates is None:
        raise ValueError("centerline pipe mesh quality requires point_coordinates")
    points = np.asarray(mesh.point_coordinates, dtype=float)
    if points.ndim != 4 or points.shape[-1] != 3:
        raise ValueError("point_coordinates must have shape (nx+1, nr+1, ntheta+1, 3)")
    center = points[:, 0, 0, :]
    station_edges = np.linalg.norm(np.diff(center, axis=0), axis=1)
    target_radius = float(mesh.y_faces[-1])
    outer = points[:, -1, :, :]
    outer_radius = np.linalg.norm(outer - center[:, None, :], axis=-1)
    radius_error = np.abs(outer_radius - target_radius)
    theta_closure = np.linalg.norm(points[:, :, 0, :] - points[:, :, -1, :], axis=-1)
    radial_edges = np.linalg.norm(np.diff(points[:, :, :-1, :], axis=1), axis=-1)
    theta_edges = np.linalg.norm(np.diff(points[:, -1, :, :], axis=1), axis=-1)
    finite_fraction = float(np.mean(np.isfinite(points)))
    station_positive = bool(np.all(station_edges > 0.0))
    max_radius_error = float(np.max(radius_error))
    max_theta_closure_error = float(np.max(theta_closure))
    validation_pass = bool(
        finite_fraction == 1.0
        and station_positive
        and max_radius_error <= max(1.0e-7, 1.0e-6 * target_radius)
        and max_theta_closure_error <= max(1.0e-7, 1.0e-6 * target_radius)
    )
    return {
        "geometry": mesh.geometry,
        "station_count": int(points.shape[0]),
        "radial_face_count": int(points.shape[1]),
        "theta_face_count": int(points.shape[2]),
        "cell_count": int(
            (points.shape[0] - 1) * (points.shape[1] - 1) * (points.shape[2] - 1)
        ),
        "finite_fraction": finite_fraction,
        "station_positive": station_positive,
        "path_length": float(np.sum(station_edges)),
        "min_station_spacing": float(np.min(station_edges))
        if station_edges.size
        else 0.0,
        "max_station_spacing": float(np.max(station_edges))
        if station_edges.size
        else 0.0,
        "target_radius": target_radius,
        "max_radius_error": max_radius_error,
        "rms_radius_error": float(np.sqrt(np.mean(radius_error**2))),
        "max_theta_closure_error": max_theta_closure_error,
        "min_radial_edge": float(np.min(radial_edges)) if radial_edges.size else 0.0,
        "min_outer_theta_edge": float(np.min(theta_edges)) if theta_edges.size else 0.0,
        "validation_pass": validation_pass,
    }


def _validated_centerline_arrays(
    centerline: dict[str, np.ndarray | jnp.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(centerline["x"], dtype=float)
    y = np.asarray(centerline["y"], dtype=float)
    z = np.asarray(centerline["z"], dtype=float)
    if not (x.shape == y.shape == z.shape):
        raise ValueError("centerline x, y, and z arrays must have the same shape")
    if x.ndim != 1 or x.size < 3:
        raise ValueError(
            "centerline arrays must be one-dimensional with at least three points"
        )
    coordinates = np.column_stack([x, y, z])
    if "station" in centerline:
        station = np.asarray(centerline["station"], dtype=float)
        if station.shape != x.shape:
            raise ValueError("centerline station array must match x/y/z shape")
    else:
        station = np.concatenate(
            [[0.0], np.cumsum(np.linalg.norm(np.diff(coordinates, axis=0), axis=1))]
        )
    if np.any(np.diff(station) <= 0.0):
        raise ValueError("centerline station must be strictly increasing")
    if not np.all(np.isfinite(coordinates)) or not np.all(np.isfinite(station)):
        raise ValueError("centerline coordinates and station must be finite")
    return station, coordinates


def _centerline_tangent(center: np.ndarray, station: np.ndarray) -> np.ndarray:
    tangent = np.gradient(center, station, axis=0)
    return tangent / np.maximum(np.linalg.norm(tangent, axis=1, keepdims=True), 1.0e-14)


def _fallback_normal(tangent: np.ndarray) -> np.ndarray:
    reference = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(reference, tangent))) > 0.94:
        reference = np.array([0.0, 1.0, 0.0])
    normal = np.cross(reference, tangent)
    return normal / max(float(np.linalg.norm(normal)), 1.0e-14)


def _centerline_frames(tangent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = np.empty_like(tangent)
    binormal = np.empty_like(tangent)
    normal[0] = _fallback_normal(tangent[0])
    binormal[0] = np.cross(tangent[0], normal[0])
    binormal[0] /= max(float(np.linalg.norm(binormal[0])), 1.0e-14)
    for index in range(1, tangent.shape[0]):
        projected = (
            normal[index - 1]
            - np.dot(normal[index - 1], tangent[index]) * tangent[index]
        )
        if np.linalg.norm(projected) <= 1.0e-12:
            projected = _fallback_normal(tangent[index])
        normal[index] = projected / max(float(np.linalg.norm(projected)), 1.0e-14)
        binormal[index] = np.cross(tangent[index], normal[index])
        binormal[index] /= max(float(np.linalg.norm(binormal[index])), 1.0e-14)
    return normal, binormal
