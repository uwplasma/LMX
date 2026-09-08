"""Tensor-product grid, staggered field container and wall-resolving coordinates.

The grid holds host-side face coordinates and is hashable static metadata: it is
never a traced value, so eigendecompositions and stencil bookkeeping happen once
at trace time. A :class:`Field` carries the traced array plus its staggered
position, following the marker-and-cell convention where an offset of ``0.5``
along an axis means cell-centred and ``0.0`` means the lower face.

Coordinate families cluster cells near walls, which high-Hartmann flow requires:
the Hartmann layer scales as ``a/Ha`` and the side layer as ``a/sqrt(Ha)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax
import numpy as np

__all__ = [
    "CENTER",
    "FACE",
    "Field",
    "Grid",
    "geometric_faces",
    "tanh_faces",
    "uniform_faces",
    "wall_resolving_faces",
]

CENTER = 0.5
FACE = 0.0
_AXES = ("x", "y", "z")


def _validated(faces: Sequence[float] | np.ndarray, *, name: str) -> np.ndarray:
    values = np.asarray(faces, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError(f"{name} must be a one-dimensional array of at least two faces")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be finite")
    if not np.all(np.diff(values) > 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return values


@dataclass(frozen=True)
class Grid:
    """Immutable tensor-product grid defined by strictly increasing face coordinates."""

    x_faces: np.ndarray
    y_faces: np.ndarray
    z_faces: np.ndarray

    def __post_init__(self) -> None:
        for axis in _AXES:
            object.__setattr__(self, f"{axis}_faces", _validated(getattr(self, f"{axis}_faces"), name=axis))

    def __hash__(self) -> int:
        return hash(tuple(faces.tobytes() for faces in self.faces))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Grid):
            return NotImplemented
        return all(np.array_equal(a, b) for a, b in zip(self.faces, other.faces, strict=True))

    @property
    def faces(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (self.x_faces, self.y_faces, self.z_faces)

    @property
    def shape(self) -> tuple[int, int, int]:
        """Number of cells along each axis."""
        return tuple(int(faces.size - 1) for faces in self.faces)

    @property
    def widths(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cell widths along each axis."""
        return tuple(np.diff(faces) for faces in self.faces)

    @property
    def centers(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Cell-centre coordinates along each axis."""
        return tuple(0.5 * (faces[:-1] + faces[1:]) for faces in self.faces)

    @property
    def extent(self) -> tuple[float, float, float]:
        """Domain length along each axis."""
        return tuple(float(faces[-1] - faces[0]) for faces in self.faces)

    def axis_index(self, axis: str | int) -> int:
        """Resolve ``"x"``/``"y"``/``"z"`` or an integer to an axis index."""
        if isinstance(axis, int):
            if axis not in (0, 1, 2):
                raise ValueError(f"axis index {axis} is out of range")
            return axis
        try:
            return _AXES.index(axis)
        except ValueError as error:
            raise ValueError(f"unknown axis {axis!r}") from error

    def cell_volumes(self) -> np.ndarray:
        """Volume of every cell, shaped like the cell-centred field."""
        dx, dy, dz = self.widths
        return dx[:, None, None] * dy[None, :, None] * dz[None, None, :]

    def face_areas(self, axis: str | int) -> np.ndarray:
        """Area of the faces normal to ``axis``, shaped like that face field."""
        index = self.axis_index(axis)
        area = np.ones(self.face_shape(index))
        for other, width in enumerate(self.widths):
            if other == index:
                continue
            area = area * width.reshape([-1 if axes == other else 1 for axes in range(3)])
        return area

    def face_shape(self, axis: str | int) -> tuple[int, int, int]:
        """Shape of a field living on the faces normal to ``axis`` (walls included)."""
        index = self.axis_index(axis)
        shape = list(self.shape)
        shape[index] += 1
        return tuple(shape)

    def offset_shape(self, offset: tuple[float, float, float]) -> tuple[int, int, int]:
        """Shape implied by a staggered ``offset``; ``FACE`` adds one entry on that axis."""
        return tuple(n + (1 if o == FACE else 0) for n, o in zip(self.shape, offset, strict=True))


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class Field:
    """A traced array together with its staggered position on a :class:`Grid`."""

    data: jax.Array
    offset: tuple[float, float, float]
    grid: Grid

    def tree_flatten(self):
        return (self.data,), (self.offset, self.grid)

    @classmethod
    def tree_unflatten(cls, static, children):
        offset, grid = static
        return cls(children[0], offset, grid)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.data.shape)

    @property
    def dtype(self):
        return self.data.dtype

    def replace_data(self, data: jax.Array) -> "Field":
        """Return the same staggered position carrying ``data``."""
        return Field(data, self.offset, self.grid)


def uniform_faces(count: int, lower: float, upper: float) -> np.ndarray:
    """Return ``count`` equal cells spanning ``[lower, upper]``."""
    if count < 1:
        raise ValueError("count must be positive")
    if not upper > lower:
        raise ValueError("upper must exceed lower")
    return np.linspace(lower, upper, count + 1)


def geometric_faces(
    count: int, lower: float, upper: float, ratio: float, *, both_ends: bool = True
) -> np.ndarray:
    """Return faces whose successive cell widths grow by ``ratio`` away from the wall.

    With ``both_ends`` the distribution is mirrored about the midpoint, which
    clusters cells at both walls of a duct.
    """
    if count < 1:
        raise ValueError("count must be positive")
    if ratio <= 0.0:
        raise ValueError("ratio must be positive")
    if not upper > lower:
        raise ValueError("upper must exceed lower")
    if both_ends:
        if count % 2:
            raise ValueError("count must be even when clustering at both ends")
        half = geometric_faces(count // 2, 0.0, 0.5, ratio, both_ends=False)
        normalized = np.concatenate([half[:-1], 1.0 - half[::-1]])
    else:
        steps = np.ones(count) if np.isclose(ratio, 1.0) else ratio ** np.arange(count)
        edges = np.concatenate([[0.0], np.cumsum(steps)])
        normalized = edges / edges[-1]
    return lower + (upper - lower) * normalized


def tanh_faces(count: int, lower: float, upper: float, beta: float) -> np.ndarray:
    """Return symmetric hyperbolic-tangent faces; larger ``beta`` clusters harder at both walls."""
    if count < 1:
        raise ValueError("count must be positive")
    if beta <= 0.0:
        raise ValueError("beta must be positive")
    if not upper > lower:
        raise ValueError("upper must exceed lower")
    uniform = np.linspace(-1.0, 1.0, count + 1)
    stretched = np.tanh(beta * uniform) / np.tanh(beta)
    normalized = 0.5 * (stretched + 1.0)
    normalized = (normalized - normalized[0]) / (normalized[-1] - normalized[0])
    return lower + (upper - lower) * normalized


def wall_resolving_faces(
    count: int,
    lower: float,
    upper: float,
    *,
    layer_thickness: float,
    cells_in_layer: int = 8,
    max_ratio: float | None = 1.15,
) -> np.ndarray:
    """Return faces resolving a wall layer of ``layer_thickness`` at both ends.

    The smallest ``cells_in_layer`` cells fall inside the layer while successive
    widths grow by at most ``max_ratio``, the stretching limit reported for
    high-Hartmann duct meshes. Raise when the request cannot be met with
    ``count`` cells so a caller never silently runs an unresolved layer.

    ``max_ratio=None`` fits the gentlest ratio that still spans the domain. The
    widths are rescaled to fill the half-width either way, so a ratio larger
    than the cell count needs buys no resolution -- it buys an operator whose
    entries span orders of magnitude for nothing.
    """
    if layer_thickness <= 0.0:
        raise ValueError("layer_thickness must be positive")
    if cells_in_layer < 1:
        raise ValueError("cells_in_layer must be positive")
    if max_ratio is None:
        return wall_resolving_faces(
            count,
            lower,
            upper,
            layer_thickness=layer_thickness,
            cells_in_layer=cells_in_layer,
            max_ratio=_fitted_ratio(count, 0.5 * (upper - lower), layer_thickness, cells_in_layer),
        )
    if max_ratio < 1.0:
        raise ValueError("max_ratio must be at least one")
    if count % 2:
        raise ValueError("count must be even when clustering at both ends")
    half_cells, half_width = count // 2, 0.5 * (upper - lower)
    if layer_thickness > half_width:
        raise ValueError("layer_thickness must not exceed the half-width")
    first = layer_thickness / _geometric_sum(cells_in_layer, max_ratio)
    widths = first * max_ratio ** np.arange(half_cells)
    reach = float(np.sum(widths))
    if reach < half_width:
        raise ValueError(
            f"{count} cells reach {reach:.4g} of the required {half_width:.4g}; "
            "increase count, cells_in_layer or max_ratio"
        )
    # Scaling down to fill the half-width only thins the inner cells further,
    # so the layer request is still satisfied after this rescaling.
    widths *= half_width / reach
    half = np.concatenate([[0.0], np.cumsum(widths)])
    return np.concatenate([lower + half[:-1], upper - half[::-1]])


def _fitted_ratio(count: int, half_width: float, layer_thickness: float, cells_in_layer: int) -> float:
    """Return the smallest growth ratio whose widths still reach ``half_width``."""
    if count % 2:
        raise ValueError("count must be even when clustering at both ends")

    def reaches(ratio: float) -> bool:
        first = layer_thickness / _geometric_sum(cells_in_layer, ratio)
        return float(np.sum(first * ratio ** np.arange(count // 2))) >= half_width

    low, high = 1.0, 4.0
    if not reaches(high):
        raise ValueError(
            f"{count} cells cannot span {half_width:.4g} while resolving a layer of "
            f"{layer_thickness:.4g}; increase count or cells_in_layer"
        )
    for _ in range(60):
        middle = 0.5 * (low + high)
        low, high = (low, middle) if reaches(middle) else (middle, high)
    return high


def _geometric_sum(terms: int, ratio: float) -> float:
    """Return ``sum(ratio**k)`` for ``k`` below ``terms``."""
    if np.isclose(ratio, 1.0):
        return float(terms)
    return float((ratio**terms - 1.0) / (ratio - 1.0))
