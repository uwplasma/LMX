"""Finite-volume stencils on the staggered grid.

Gradients map a cell-centred field to the faces normal to one axis; the
divergence maps face-normal fields back to cells as the net flux through the
cell's faces divided by its volume. Written this way the two are exact discrete
adjoints under the volume and face weights of :func:`cell_inner_product` and
:func:`face_inner_product`, which is the identity a compatible pressure or
potential solve depends on: it is what makes the projection idempotent and keeps
the Lorentz force from doing spurious work in the core of a high-Hartmann duct.

Ghost cells carry the wall condition (see :mod:`lmx.bc`), so a wall face is
differenced with the same expression as an interior face. The ghost cell mirrors
the width of the wall cell, which places its centre one wall-cell width outside
and makes that single expression reproduce both the prescribed-value and the
prescribed-gradient condition.

Accuracy at a prescribed-value wall is deliberate. The wall flux is the two-point
difference ``(p_0 - g) / (dx_0 / 2)``, which is centred a quarter cell inside the
wall and so is first order there, while every interior face is second order. This
is the stencil that keeps the assembled Laplacian symmetric and the fluxes
conservative, which the pressure and potential solves depend on; a three-point
wall stencil would raise the wall order at the cost of that symmetry. The choice
is pinned by test, not left implicit.

On a stretched mesh the interior two-point gradient is centred between the two
cell centres rather than on the face, so its truncation error is first order in
the spacing change. Solution order there is a manufactured-solution question,
verified in the step that owns that study rather than asserted here.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .bc import BoundaryCondition, pad
from .grid import CENTER, FACE, Field, Grid

__all__ = [
    "axis_divergence",
    "cell_inner_product",
    "divergence",
    "face_distances",
    "face_gradient",
    "face_inner_product",
    "face_interpolate",
    "laplacian",
    "staggered_laplacian",
]


def face_distances(grid: Grid, axis: int, condition: BoundaryCondition) -> np.ndarray:
    """Return the centre-to-centre distance across every face normal to ``axis``."""
    widths = np.asarray(grid.widths[axis])
    interior = 0.5 * (widths[:-1] + widths[1:])
    if condition.is_periodic:
        wrap = 0.5 * (widths[0] + widths[-1])
        return np.concatenate(([wrap], interior, [wrap]))
    return np.concatenate(([widths[0]], interior, [widths[-1]]))


def face_gradient(field: Field, axis: int | str, condition: BoundaryCondition) -> Field:
    """Differentiate a cell-centred field onto the faces normal to ``axis``."""
    grid = field.grid
    index = grid.axis_index(axis)
    _require_cell_centred(field)
    padded = pad(field.data, index, condition, grid=grid)
    difference = _take(padded, index, slice(1, None)) - _take(padded, index, slice(None, -1))
    distances = face_distances(grid, index, condition)
    offset = tuple(FACE if position == index else CENTER for position in range(3))
    return Field(difference / _broadcast(distances, index, field.dtype), offset, grid)


def axis_divergence(face: Field, axis: int | str) -> Field:
    """Return what one face-normal field contributes to the divergence."""
    grid = face.grid
    index = grid.axis_index(axis)
    _require_face(face, index)
    area = _as_array(grid.face_areas(index), face.dtype)
    flux = area * face.data
    contribution = _take(flux, index, slice(1, None)) - _take(flux, index, slice(None, -1))
    volumes = _as_array(grid.cell_volumes(), face.dtype)
    return Field(contribution / volumes, (CENTER, CENTER, CENTER), grid)


def divergence(faces: tuple[Field, Field, Field]) -> Field:
    """Return the net outward flux per unit volume of three face-normal fields."""
    grid = faces[0].grid
    total = None
    for index, face in enumerate(faces):
        if face.grid != grid:
            raise ValueError("face fields must share one grid")
        contribution = axis_divergence(face, index).data
        total = contribution if total is None else total + contribution
    return Field(total, (CENTER, CENTER, CENTER), grid)


def laplacian(field: Field, conditions: tuple[BoundaryCondition, ...]) -> Field:
    """Return the divergence of the gradient of a cell-centred field."""
    if len(conditions) != 3:
        raise ValueError("laplacian needs one boundary condition per axis")
    gradients = tuple(face_gradient(field, index, conditions[index]) for index in range(3))
    return divergence(gradients)


def face_interpolate(field: Field, axis: int | str, condition: BoundaryCondition) -> Field:
    """Interpolate a cell-centred field onto faces, weighted by the distance to each centre.

    Distance weighting keeps the interpolation second order on a stretched mesh,
    where the arithmetic mean is only first order.
    """
    grid = field.grid
    index = grid.axis_index(axis)
    _require_cell_centred(field)
    widths = np.asarray(grid.widths[index])
    padded = pad(field.data, index, condition, grid=grid)
    ghosted = np.concatenate(([widths[0]], widths, [widths[-1]]))
    lower_weight = ghosted[1:] / (ghosted[:-1] + ghosted[1:])
    weight = _broadcast(lower_weight, index, field.dtype)
    lower = _take(padded, index, slice(None, -1))
    upper = _take(padded, index, slice(1, None))
    offset = tuple(FACE if position == index else CENTER for position in range(3))
    return Field(weight * lower + (1.0 - weight) * upper, offset, grid)


def cell_inner_product(left: Field, right: Field) -> jnp.ndarray:
    """Return the volume-weighted inner product of two cell-centred fields."""
    _require_cell_centred(left)
    _require_cell_centred(right)
    if left.grid != right.grid:
        raise ValueError("fields must share one grid")
    volumes = _as_array(left.grid.cell_volumes(), left.dtype)
    return jnp.sum(volumes * left.data * right.data)


def face_inner_product(
    left: Field, right: Field, axis: int | str, condition: BoundaryCondition
) -> jnp.ndarray:
    """Return the face-weighted inner product of two face-normal fields.

    The weight is the face area times the centre-to-centre distance, the volume
    of the control cell straddling that face, which is the weight under which
    :func:`divergence` and :func:`face_gradient` are exact adjoints.
    """
    grid = left.grid
    index = grid.axis_index(axis)
    _require_face(left, index)
    _require_face(right, index)
    if left.grid != right.grid:
        raise ValueError("fields must share one grid")
    weights = grid.face_areas(index) * _shaped(face_distances(grid, index, condition), index)
    return jnp.sum(_as_array(weights, left.dtype) * left.data * right.data)


def _require_cell_centred(field: Field) -> None:
    if field.offset != (CENTER, CENTER, CENTER):
        raise ValueError(f"expected a cell-centred field, got offset {field.offset}")
    if field.shape != field.grid.shape:
        raise ValueError(f"cell field shape {field.shape} does not match grid {field.grid.shape}")


def _require_face(field: Field, index: int) -> None:
    expected = tuple(FACE if position == index else CENTER for position in range(3))
    if field.offset != expected:
        raise ValueError(f"expected a field on faces normal to axis {index}, got offset {field.offset}")
    if field.shape != field.grid.face_shape(index):
        raise ValueError(f"face field shape {field.shape} does not match grid {field.grid.face_shape(index)}")


def _take(data: jnp.ndarray, axis: int, selection: slice) -> jnp.ndarray:
    return data[(slice(None),) * axis + (selection,)]


def _shaped(values: np.ndarray, axis: int) -> np.ndarray:
    return values.reshape([-1 if position == axis else 1 for position in range(3)])


def _broadcast(values: np.ndarray, axis: int, dtype) -> jnp.ndarray:
    return _as_array(_shaped(values, axis), dtype)


def _as_array(values: np.ndarray, dtype) -> jnp.ndarray:
    return jnp.asarray(values, dtype=dtype)


def staggered_laplacian(
    field: Field, conditions: tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition]
) -> Field:
    """Return the Laplacian of a field at any staggered position.

    A velocity component in the marker-and-cell layout is cell-centred along two
    axes and face-centred along the third, so its Laplacian needs both stencils.
    Along a cell-centred axis the wall condition supplies a ghost value and the
    scalar path applies unchanged. Along a face-centred axis the field already
    sits on the wall, so no ghost exists and none is invented: the value is
    differenced to the cell centres with the cell widths and back to the faces
    with the centre-to-centre distances.

    On a face-centred axis with a wall the two boundary faces are returned as
    zero. Their value is prescribed by the boundary condition, not evolved, and
    returning zero keeps a caller that updates them from silently using a
    one-sided stencil that does not exist. A periodic axis wraps and has no such
    face.
    """
    if len(conditions) != 3:
        raise ValueError("a staggered Laplacian needs one boundary condition per axis")
    grid = field.grid
    if field.shape != grid.offset_shape(field.offset):
        raise ValueError(f"field shape {field.shape} does not match its offset {field.offset}")
    total = None
    for axis, condition in enumerate(conditions):
        if field.offset[axis] == CENTER:
            contribution = _centred_axis_laplacian(field, axis, condition)
        else:
            contribution = _face_axis_laplacian(field, axis, condition)
        total = contribution if total is None else total + contribution
    return field.replace_data(total)


def _centred_axis_laplacian(field: Field, axis: int, condition: BoundaryCondition) -> jnp.ndarray:
    """Second difference along an axis on which the field is cell centred."""
    grid = field.grid
    padded = pad(field.data, axis, condition, grid=grid)
    distances = face_distances(grid, axis, condition)
    gradient = (_take(padded, axis, slice(1, None)) - _take(padded, axis, slice(None, -1))) / _broadcast(
        distances, axis, field.dtype
    )
    widths = np.asarray(grid.widths[axis])
    difference = _take(gradient, axis, slice(1, None)) - _take(gradient, axis, slice(None, -1))
    return difference / _broadcast(widths, axis, field.dtype)


def _face_axis_laplacian(field: Field, axis: int, condition: BoundaryCondition) -> jnp.ndarray:
    """Second difference along an axis on which the field sits on the faces."""
    grid = field.grid
    widths = np.asarray(grid.widths[axis])
    data = field.data
    if condition.is_periodic:
        # The first and last faces coincide; drop the duplicate before wrapping.
        interior = _take(data, axis, slice(None, -1))
        gradient = (_roll(interior, axis, -1) - interior) / _broadcast(widths, axis, field.dtype)
        distances = 0.5 * (widths + np.roll(widths, 1))
        difference = gradient - _roll(gradient, axis, 1)
        result = difference / _broadcast(distances, axis, field.dtype)
        return jnp.concatenate((result, _take(result, axis, slice(None, 1))), axis=axis)
    gradient = (_take(data, axis, slice(1, None)) - _take(data, axis, slice(None, -1))) / _broadcast(
        widths, axis, field.dtype
    )
    distances = 0.5 * (widths[:-1] + widths[1:])
    inner = (_take(gradient, axis, slice(1, None)) - _take(gradient, axis, slice(None, -1))) / _broadcast(
        distances, axis, field.dtype
    )
    zeros = jnp.zeros_like(_take(data, axis, slice(None, 1)))
    return jnp.concatenate((zeros, inner, zeros), axis=axis)


def _roll(data: jnp.ndarray, axis: int, shift: int) -> jnp.ndarray:
    return jnp.roll(data, shift, axis=axis)
