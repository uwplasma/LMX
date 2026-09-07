"""Charge-conservative electric coupling on the staggered grid.

For inductionless flow the current density is

.. math:: \\mathbf J = \\sigma(-\\nabla\\phi + \\mathbf u\\times\\mathbf B),
   \\qquad \\nabla\\cdot\\mathbf J = 0,

and the momentum equation is driven by :math:`\\mathbf J\\times\\mathbf B`.

The discretization follows Ni et al., *J. Comput. Phys.* **227** (2007) 174 and
205, which is what the codes that reproduce the ALEX ducts use. Its rule is that
one face-normal current

.. math:: J_{n,f} = \\sigma_f\\left[-\\frac{\\phi_N-\\phi_P}{d_{PN}}
   + (\\mathbf u_f\\times\\mathbf B_f)\\cdot\\mathbf n_f\\right]

is the single source of truth: the potential equation is the divergence of
exactly that flux, and the Lorentz force is rebuilt from the same numbers. The
reason is quantitative. In the core of a duct the momentum balance is
:math:`-\\nabla p + \\mathbf J\\times\\mathbf B = 0` to :math:`O(Ha^{-2})`, so an
:math:`O(\\Delta)` inconsistency between the two parts of :math:`\\mathbf J` is
amplified by :math:`Ha^2` and appears as a spurious core current. Computing the
potential from one stencil and the force from another is what that rule forbids.

Face conductivity is the harmonic mean, which is the series resistance of the two
half-cells and therefore the right average across a fluid-wall jump; the
arithmetic mean would overstate the current entering a poorly conducting wall.

Velocities live on their own faces, so a transverse component is averaged to the
cell centre and then interpolated to the face where the electromotive force is
needed. Both steps are second order on a stretched mesh.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .bc import NEUMANN, BoundaryCondition, pad
from .grid import CENTER, FACE, Field
from .ops import divergence, face_gradient, face_interpolate

__all__ = [
    "cell_average",
    "charge_residual",
    "face_conductivity",
    "face_current",
    "face_electromotive_force",
    "lorentz_force",
    "wall_insulated",
]

# (normal axis, first transverse axis, second transverse axis) in cyclic order.
_CYCLIC = ((0, 1, 2), (1, 2, 0), (2, 0, 1))


def cell_average(field: Field, axis: int | str) -> Field:
    """Average a face-normal field onto cell centres along ``axis``."""
    grid = field.grid
    index = grid.axis_index(axis)
    expected = tuple(FACE if position == index else CENTER for position in range(3))
    if field.offset != expected:
        raise ValueError(f"expected a field on faces normal to axis {index}, got offset {field.offset}")
    lower = field.data[(slice(None),) * index + (slice(None, -1),)]
    upper = field.data[(slice(None),) * index + (slice(1, None),)]
    return Field(0.5 * (lower + upper), (CENTER, CENTER, CENTER), grid)


def face_conductivity(sigma: Field, axis: int | str, condition: BoundaryCondition) -> Field:
    """Return the harmonic-mean conductivity on the faces normal to ``axis``.

    The harmonic mean is the series resistance of the two half-cells, so it is
    the average that reproduces the current through a fluid-wall interface. Its
    arithmetic counterpart lets a poorly conducting wall draw too much current.
    """
    grid = sigma.grid
    index = grid.axis_index(axis)
    if sigma.offset != (CENTER, CENTER, CENTER):
        raise ValueError("conductivity must be cell centred")
    if jnp.ndim(sigma.data) != 3 or sigma.shape != grid.shape:
        raise ValueError(f"conductivity shape {sigma.shape} does not match grid {grid.shape}")
    padded = pad(sigma.data, index, condition, grid=grid)
    lower = padded[(slice(None),) * index + (slice(None, -1),)]
    upper = padded[(slice(None),) * index + (slice(1, None),)]
    widths = np.asarray(grid.widths[index])
    ghosted = np.concatenate(([widths[0]], widths, [widths[-1]]))
    lower_half = _broadcast(0.5 * ghosted[:-1], index, sigma.dtype)
    upper_half = _broadcast(0.5 * ghosted[1:], index, sigma.dtype)
    total = lower_half + upper_half
    resistance = lower_half / lower + upper_half / upper
    offset = tuple(FACE if position == index else CENTER for position in range(3))
    return Field(total / resistance, offset, grid)


def face_electromotive_force(
    velocity: tuple[Field, Field, Field],
    magnetic_field: tuple[Field, Field, Field],
    axis: int | str,
    conditions: tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition],
) -> Field:
    """Return ``(u x B).n`` on the faces normal to ``axis``.

    ``velocity`` holds the three face-normal components in the marker-and-cell
    layout and ``magnetic_field`` the three cell-centred components. Both
    transverse velocity components are averaged to cell centres and interpolated
    to the target face, so the electromotive force is evaluated where the current
    flux needs it.
    """
    grid = velocity[0].grid
    index = grid.axis_index(axis)
    _, first, second = _CYCLIC[index]
    velocity_first = face_interpolate(cell_average(velocity[first], first), index, conditions[index])
    velocity_second = face_interpolate(cell_average(velocity[second], second), index, conditions[index])
    field_first = face_interpolate(magnetic_field[first], index, conditions[index])
    field_second = face_interpolate(magnetic_field[second], index, conditions[index])
    emf = velocity_first.data * field_second.data - velocity_second.data * field_first.data
    offset = tuple(FACE if position == index else CENTER for position in range(3))
    return Field(emf, offset, grid)


def wall_insulated(flux: Field, axis: int | str, condition: BoundaryCondition) -> Field:
    """Zero a face-normal flux on the two boundary faces of an insulating wall.

    An insulating wall carries no current, :math:`\\mathbf J\\cdot\\mathbf n = 0`,
    and both sides of the potential equation have to say so. The homogeneous
    Neumann Laplacian already drops the wall faces; unless the motional term is
    dropped there too, the equation asks the potential to absorb a boundary flux
    that the operator cannot produce, and the resulting current -- and the
    Lorentz force built from it -- is wrong wherever a side wall cuts across
    :math:`\\mathbf u\\times\\mathbf B`.

    Only a Neumann condition names an insulating wall. A prescribed potential
    is a perfectly conducting one and does carry current, so it is returned
    untouched, as is a periodic axis, which has no wall at all.
    """
    index = flux.grid.axis_index(axis)
    if condition.kind != NEUMANN:
        return flux
    selection = (slice(None),) * index
    data = flux.data.at[selection + (0,)].set(0.0).at[selection + (-1,)].set(0.0)
    return flux.replace_data(data)


def face_current(
    potential: Field,
    conductivity: Field,
    electromotive_force: Field,
    axis: int | str,
    condition: BoundaryCondition,
) -> Field:
    """Return the face-normal current density of Ohm's law on one face set.

    This is the only place a current is formed. The potential equation and the
    Lorentz force are both built from its result, which is what keeps them
    consistent at high Hartmann number.
    """
    gradient = face_gradient(potential, axis, condition)
    if conductivity.offset != gradient.offset or conductivity.shape != gradient.shape:
        raise ValueError("conductivity must live on the same faces as the potential gradient")
    if electromotive_force.offset != gradient.offset or electromotive_force.shape != gradient.shape:
        raise ValueError("electromotive force must live on the same faces as the potential gradient")
    return gradient.replace_data(conductivity.data * (electromotive_force.data - gradient.data))


def charge_residual(currents: tuple[Field, Field, Field]) -> Field:
    """Return the net current leaving each cell per unit volume."""
    return divergence(currents)


def lorentz_force(
    currents: tuple[Field, Field, Field],
    magnetic_field: tuple[Field, Field, Field],
    conditions: tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition],
) -> tuple[Field, Field, Field]:
    """Return the cell-centred Lorentz force built from the face currents.

    Ni's face form is used,

    .. math:: (\\mathbf J\\times\\mathbf B)_c
       = \\frac{1}{\\Omega_c}\\sum_f J_{n,f}\\,s_f\\,(\\mathbf r_f-\\mathbf r_c)
         \\times\\mathbf B_f,

    which never forms a cell-centred current vector and evaluates the magnetic
    field on the faces, so it stays correct where the field varies.
    """
    grid = currents[0].grid
    volumes = grid.cell_volumes()
    components = [jnp.zeros(grid.shape, dtype=currents[0].dtype) for _ in range(3)]
    for index, current in enumerate(currents):
        _, first, second = _CYCLIC[index]
        area = np.asarray(grid.face_areas(index))
        widths = np.asarray(grid.widths[index])
        # The face-to-centre arm is half a cell along the face normal, negative on
        # the lower face and positive on the upper one.
        arm = 0.5 * widths
        field_first = face_interpolate(magnetic_field[first], index, conditions[index])
        field_second = face_interpolate(magnetic_field[second], index, conditions[index])
        weighted = jnp.asarray(area, dtype=current.dtype) * current.data
        lower = (slice(None),) * index + (slice(None, -1),)
        upper = (slice(None),) * index + (slice(1, None),)
        arm_shaped = _broadcast(arm, index, current.dtype)
        # With the current stored along the positive axis, the outward flux and the
        # face-to-centre arm change sign together on the lower face, so the two
        # faces add rather than cancel. The cross product e_k x B contributes
        # +B_first to the second transverse component and -B_second to the first.
        for target, source, sign in ((second, field_first, 1.0), (first, field_second, -1.0)):
            contribution = (
                sign
                * arm_shaped
                * (weighted[lower] * source.data[lower] + weighted[upper] * source.data[upper])
            )
            components[target] = components[target] + contribution
    volume_array = jnp.asarray(volumes, dtype=currents[0].dtype)
    return tuple(Field(component / volume_array, (CENTER, CENTER, CENTER), grid) for component in components)


def _broadcast(values: np.ndarray, axis: int, dtype) -> jnp.ndarray:
    return jnp.asarray(values.reshape([-1 if position == axis else 1 for position in range(3)]), dtype=dtype)
