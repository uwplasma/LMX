"""Fully developed flow along a circular pipe in a transverse magnetic field.

A pipe is the other half of the duct validation, and the geometry the ALEX B1
experiment used. Nothing about the physics changes -- the same inductionless
system, the same insulating or thin conducting wall -- but the coordinates do,
and with them what separates and what does not.

Fully developed means the velocity is axial and depends only on the cross
section, :math:`\\mathbf u = u(r,\\theta)\\hat z`. With :math:`\\mathbf B = B\\hat x`
the Lorentz force is then purely axial as well, so the transverse momentum
equations are satisfied by rest and the whole problem is one scalar equation
coupled to the potential:

.. math::
   \\nabla^2 u + B\\,\\partial_y\\varphi - B^2 u + f = 0,
   \\qquad \\nabla\\cdot\\mathbf J = 0,

with :math:`\\mathbf J = -\\nabla\\varphi + \\mathbf u\\times\\mathbf B`. Both
unknowns are cell centred, which is why this needs none of the staggered vector
machinery of :mod:`lmx.core3d`: there is no cross flow to project.

The electromotive force is where the coordinates show. A uniform Cartesian field
is not uniform in polar components, :math:`B_r = B\\cos\\theta` and
:math:`B_\\theta = -B\\sin\\theta`, so :math:`\\mathbf u\\times\\mathbf B` has
:math:`B u\\sin\\theta` through a radial face and :math:`B u\\cos\\theta` through
an azimuthal one. The same face currents then carry the potential equation and
the axial force, which is the consistency the whole package is built on.

The system is linear in the velocity, so it is one preconditioned Krylov solve,
not a Newton iteration. The preconditioner is the damped operator
:math:`(B^2 - \\nabla^2)^{-1}`, factorized exactly by
:func:`lmx.poisson.fast_diagonal_polar_poisson`, which is what keeps the
iteration count from growing with the Hartmann number.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import solvax

from .bc import DIRICHLET, NEUMANN, PERIODIC, BoundaryCondition
from .em import cell_average, thin_wall_flux, wall_insulated
from .grid import CENTER, POLAR, Field, Grid, uniform_faces, wall_resolving_faces
from .ops import divergence, face_gradient, face_interpolate
from .poisson import fast_diagonal_polar_poisson

__all__ = ["PipeProblem", "pipe_grid", "pipe_problem", "solve_pipe"]

_WALL = BoundaryCondition(DIRICHLET)
_INSULATING = BoundaryCondition(NEUMANN)
_WRAP = BoundaryCondition(PERIODIC)


@dataclass(frozen=True)
class PipeProblem:
    """A circular pipe of unit radius in a transverse field of magnitude ``hartmann``."""

    grid: Grid
    hartmann: float
    wall_conductance: float = 0.0
    forcing: float = 1.0

    def __post_init__(self) -> None:
        if not self.grid.is_polar:
            raise ValueError("a pipe needs a polar grid")
        if float(self.hartmann) < 0.0:
            raise ValueError("hartmann must not be negative")
        if float(self.wall_conductance) < 0.0:
            raise ValueError("wall conductance must not be negative")

    @property
    def conditions(self) -> tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition]:
        """Conditions on the potential: insulating at the wall, periodic elsewhere."""
        return (_INSULATING, _WRAP, _WRAP)

    def factorization(self):
        """Factorize the potential Laplacian."""
        return fast_diagonal_polar_poisson(self.grid, self.conditions)

    def preconditioner(self):
        """Factorize the damped operator that preconditions the velocity solve."""
        return fast_diagonal_polar_poisson(
            self.grid,
            (_WALL, _WRAP, _WRAP),
            shift=max(float(self.hartmann) ** 2, 1.0),
            coefficient=1.0,
        )


def pipe_grid(radial: int, azimuthal: int, hartmann: float, *, cells_in_layer: int = 6) -> Grid:
    """Return a polar grid whose radial cells resolve the ``1/Ha`` wall layer."""
    if hartmann > 1.0:
        faces = wall_resolving_faces(
            2 * radial,
            -1.0,
            1.0,
            layer_thickness=1.0 / hartmann,
            cells_in_layer=cells_in_layer,
            max_ratio=None,
        )
        radial_faces = np.concatenate(([0.0], faces[radial + 1 :]))
    else:
        radial_faces = uniform_faces(radial, 0.0, 1.0)
    return Grid(
        radial_faces,
        uniform_faces(azimuthal, 0.0, 2.0 * np.pi),
        uniform_faces(1, 0.0, 1.0),
        geometry=POLAR,
    )


def pipe_problem(
    *, hartmann: float, radial: int = 48, azimuthal: int = 64, wall_conductance: float = 0.0
) -> PipeProblem:
    """Build a pipe whose mesh resolves the layer its Hartmann number implies."""
    return PipeProblem(pipe_grid(radial, azimuthal, hartmann), float(hartmann), float(wall_conductance))


def _angles(grid: Grid, axis: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``sin`` and ``cos`` of the azimuth where the faces of ``axis`` live."""
    angle = np.asarray(grid.faces[1] if axis == 1 else grid.centers[1])
    return np.sin(angle)[None, :, None], np.cos(angle)[None, :, None]


def _face_emf(velocity: Field, problem: PipeProblem) -> tuple[Field, Field, Field]:
    """Return ``(u x B).n`` on the radial, azimuthal and axial faces.

    A uniform Cartesian field has ``B_r = B cos(theta)`` and
    ``B_theta = -B sin(theta)``, so an axial velocity drives ``B u sin(theta)``
    through a radial face and ``B u cos(theta)`` through an azimuthal one.
    """
    grid = problem.grid
    field = float(problem.hartmann)
    radial = face_interpolate(velocity, 0, _WALL)
    azimuthal = face_interpolate(velocity, 1, _WRAP)
    radial_sin, _ = _angles(grid, 0)
    _, azimuthal_cos = _angles(grid, 1)
    axial = Field(jnp.zeros(grid.face_shape(2), dtype=velocity.dtype), (CENTER, CENTER, 0.0), grid)
    return (
        radial.replace_data(field * radial.data * jnp.asarray(radial_sin, dtype=velocity.dtype)),
        azimuthal.replace_data(field * azimuthal.data * jnp.asarray(azimuthal_cos, dtype=velocity.dtype)),
        axial,
    )


def _face_currents(velocity: Field, potential: Field, problem: PipeProblem) -> tuple[Field, Field, Field]:
    """Return the face-normal currents, closed at the wall by its own model."""
    conditions = problem.conditions
    emf = _face_emf(velocity, problem)
    currents = []
    for axis in range(3):
        gradient = face_gradient(potential, axis, conditions[axis])
        ohmic = wall_insulated(gradient.replace_data(emf[axis].data - gradient.data), axis, conditions[axis])
        if axis == 0 and problem.wall_conductance:
            wall = thin_wall_flux(potential, 0, conditions[0], problem.wall_conductance, conditions)
            ohmic = ohmic.replace_data(ohmic.data + wall.data)
        currents.append(ohmic)
    return tuple(currents)


def _axial_force(currents: tuple[Field, Field, Field], problem: PipeProblem) -> jnp.ndarray:
    """Return ``(J x B)_z`` at cell centres, from the same currents the charge balance uses."""
    grid = problem.grid
    sine, _ = _angles(grid, 0)
    _, cosine = _angles(grid, 0)
    radial = cell_average(currents[0], 0).data
    azimuthal = cell_average(currents[1], 1).data
    dtype = radial.dtype
    return -float(problem.hartmann) * (
        radial * jnp.asarray(sine, dtype=dtype) + azimuthal * jnp.asarray(cosine, dtype=dtype)
    )


def _potential(velocity: Field, problem: PipeProblem, factorization) -> Field:
    """Solve the charge balance for the potential the velocity induces."""
    conditions = problem.conditions
    motional = tuple(
        wall_insulated(component, axis, conditions[axis])
        for axis, component in enumerate(_face_emf(velocity, problem))
    )
    source = divergence(motional)
    if not problem.wall_conductance:
        return factorization.solve(source)

    def operator(potential: Field) -> Field:
        zero = potential.replace_data(jnp.zeros_like(potential.data))
        currents = _face_currents(zero, potential, problem)
        balance = divergence(currents)
        return balance.replace_data(-balance.data)

    def solve(matvec, target):
        return solvax.gmres(matvec, target, precond=factorization.solve, rtol=1.0e-12, max_restarts=20).x

    return jax.lax.custom_linear_solve(operator, source, solve, solve)


def pipe_residual(velocity: Field, problem: PipeProblem, factorization) -> Field:
    """Return the steady axial momentum residual of a candidate velocity."""
    from .ops import laplacian

    potential = _potential(velocity, problem, factorization)
    currents = _face_currents(velocity, potential, problem)
    viscous = laplacian(velocity, (_WALL, _WRAP, _WRAP))
    return velocity.replace_data(viscous.data + _axial_force(currents, problem) + float(problem.forcing))


def solve_pipe(problem: PipeProblem, *, tolerance: float = 1.0e-11, max_restarts: int = 40):
    """Return the axial velocity and the potential of a fully developed pipe.

    The system is linear in the velocity, so this is one preconditioned Krylov
    solve. The preconditioner inverts the damped operator exactly, which is the
    stiff part of the problem and the reason the iteration does not lengthen
    with the Hartmann number.
    """
    factorization = problem.factorization()
    damped = problem.preconditioner()
    start = Field(jnp.zeros(problem.grid.shape), (CENTER, CENTER, CENTER), problem.grid)
    drive = pipe_residual(start, problem, factorization)

    def operator(velocity: Field) -> Field:
        residual = pipe_residual(velocity, problem, factorization)
        return residual.replace_data(drive.data - residual.data)

    def solve(matvec, target):
        return solvax.gmres(
            matvec, target, precond=damped.solve, rtol=tolerance, restart=200, max_restarts=max_restarts
        ).x

    velocity = jax.lax.custom_linear_solve(operator, drive, solve, solve)
    residual = pipe_residual(velocity, problem, factorization)
    scale = float(jnp.max(jnp.abs(drive.data)))
    remaining = float(jnp.max(jnp.abs(residual.data)))
    if not np.isfinite(remaining) or remaining > 1.0e-6 * max(scale, 1.0):
        raise RuntimeError(
            f"the pipe solve did not converge: residual {remaining:.3e} against a drive of {scale:.3e}; "
            "raise max_restarts or resolve the wall layer"
        )
    return velocity, _potential(velocity, problem, factorization)


def flow_rate(velocity: Field) -> float:
    """Return the mean axial velocity over the cross-section."""
    volumes = np.asarray(velocity.grid.cell_volumes())
    return float((np.asarray(velocity.data) * volumes).sum() / volumes.sum())
