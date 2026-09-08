"""Solve the steady state directly instead of marching to it.

Marching a duct to its fully developed state costs a number of steps set by the
slowest physical time in the problem, and at blanket conditions that is the
viscous one, :math:`a^2/\\nu`. The state itself is the root of a residual, so it
can be found in a handful of Newton steps instead -- and, more importantly for
what this package is for, a root is differentiable through the implicit function
theorem, so a design derivative costs one adjoint solve rather than a tape of
every step.

The residual is written in the velocity alone. Both the potential and the
pressure are determined by the velocity through linear solves that
:mod:`lmx.poisson` performs exactly, so carrying them as unknowns would only add
a saddle-point structure that the projection already removes:

.. math:: \\mathbf R(\\mathbf u)
   = \\mathbb P\\left[\\nu\\nabla^2\\mathbf u
     + (\\mathbf F(\\mathbf u) + \\mathbf f)/\\rho
     - \\nabla\\cdot(\\mathbf u\\mathbf u)\\right],

with :math:`\\mathbb P` the discrete projection onto the divergence-free face
fields and :math:`\\mathbf F` the Ni face-form Lorentz force of the potential
that :math:`\\mathbf u` induces. A root of :math:`\\mathbf R` is a steady state
of :func:`lmx.core3d.step`, and the projection keeps the iteration inside the
subspace the time stepper never leaves.

The preconditioner is one projection step. Writing the implicit viscous operator
as :math:`M = (\\text{shift}\\,I - \\Delta t\\,\\nu\\nabla^2)/\\Delta t`, its
inverse tends to :math:`(\\lambda - \\nu\\nabla^2)^{-1}` as the pseudo-step
grows, which is the exact Jacobian inverse of the Stokes and damping part of the
residual. That is the part which is stiff -- the viscous term as the mesh is
refined and the damping as :math:`Ha^2` -- so the Krylov iteration is left with
the well-conditioned remainder.

The Krylov subspace is deliberately large. Restarted GMRES stagnates on this
operator once the field is strong: at :math:`Ha=300`, GMRES(60) with sixty
restarts -- 3600 iterations -- reduces the residual by a factor of forty and
stops improving, while GMRES(400) converges in 1077. The subspace is what the
method needs here, not more restarts of a short one.

An anisotropic preconditioner was tried and rejected. Eliminating the potential
from a fully developed duct leaves :math:`-\\lambda\\,\\partial_b\\nabla^{-2}
\\partial_b`, whose symbol in the velocity eigenbasis is
:math:`\\ell_b/\\sum_k\\ell_k`, and folding that into the factorization is nearly
free. It makes the iteration *worse* -- residual 1.6 against 0.12 at
:math:`Ha=300` -- because the potential obeys Neumann conditions while the
velocity obeys Dirichlet ones, so the two do not share an eigenbasis exactly
where it matters, in the Hartmann layer. The uniform rate is kept.

Failure is raised, never returned. A Newton iteration that stops on its step
limit, or an adjoint solve that does not converge, would otherwise hand back a
plausible-looking field and a gradient computed at a point that is not a root.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import solvax

from .advect import momentum_advection
from .core3d import (
    ChannelProblem,
    electric_state,
    enforce_face_constraints,
    project,
    velocity_condition,
    zero_velocity,
)
from .grid import Field
from .ops import face_interpolate, staggered_laplacian
from .poisson import FastDiagonalHelmholtz, FastDiagonalPoisson

__all__ = ["SteadySolution", "solve_steady_state", "steady_residual"]


@dataclass(frozen=True)
class SteadySolution:
    """A converged steady state and the evidence that it converged."""

    velocity: tuple[Field, Field, Field]
    pressure: Field
    potential: Field
    residual_norm: jnp.ndarray
    steps: int


def steady_residual(
    velocity: tuple[Field, Field, Field],
    problem: ChannelProblem,
    factorization: FastDiagonalPoisson | None = None,
    *,
    forcing: tuple[float, float, float] | None = None,
    field_scale: float | jnp.ndarray = 1.0,
) -> tuple[Field, Field, Field]:
    """Return the projected steady momentum residual of a velocity field.

    Zero exactly when ``velocity`` is a fixed point of :func:`lmx.core3d.step`.
    ``forcing`` and ``field_scale`` are the continuous design inputs and may be
    traced; everything ``problem`` carries is static, because the factorizations
    read concrete arrays.
    """
    factorization = problem.factorization() if factorization is None else factorization
    velocity = enforce_face_constraints(velocity, problem)
    scalar = problem.scalar_conditions
    drive = problem.forcing if forcing is None else forcing
    _, force = electric_state(velocity, problem, factorization, field_scale)
    conditions = tuple(velocity_condition(problem.conditions, axis) for axis in range(3))
    transport = (
        None
        if problem.advection == "off"
        else momentum_advection(velocity, conditions, limited=problem.advection == "limited")
    )
    terms = []
    for component, field in enumerate(velocity):
        # The damping rate is a preconditioning device, not a term: the conservative
        # face force already carries the whole Lorentz contribution, so subtracting
        # it here as well would count the same physics twice.
        body = face_interpolate(force[component], component, scalar[component])
        value = (
            problem.viscosity * staggered_laplacian(field, conditions).data
            + (body.data + drive[component]) / problem.density
        )
        if transport is not None:
            value = value - transport[component].data
        terms.append(field.replace_data(value))
    corrected, _ = project(tuple(terms), problem, factorization)
    return corrected


def _preconditioner(
    problem: ChannelProblem,
    factorization: FastDiagonalPoisson,
    viscous: tuple[FastDiagonalHelmholtz, ...],
    pseudo_step: float,
):
    """One projection step, used as the right preconditioner of the Newton system.

    ``viscous`` has to be factorized at ``pseudo_step`` and not at the problem's
    own step: a preconditioner built at the wrong step is a different operator,
    which is what leaves the Krylov iteration stalling as soon as the damping is
    stiff.
    """

    def apply(direction: tuple[Field, Field, Field]) -> tuple[Field, Field, Field]:
        scaled = tuple(
            viscous[component].solve(field.replace_data(pseudo_step * field.data))
            for component, field in enumerate(direction)
        )
        corrected, _ = project(scaled, problem, factorization)
        return corrected

    return apply


def solve_steady_state(
    problem: ChannelProblem,
    velocity: tuple[Field, Field, Field] | None = None,
    *,
    tolerance: float = 1.0e-9,
    max_steps: int = 40,
    pseudo_step: float | None = None,
    forcing: tuple[float, float, float] | None = None,
    field_scale: float | jnp.ndarray = 1.0,
    linear_tolerance: float = 1.0e-6,
    linear_restart: int = 400,
    linear_max_restarts: int = 6,
) -> SteadySolution:
    """Find the steady state by matrix-free Newton-Krylov, differentiably.

    The solve is registered through :func:`solvax.root_solve`, so a derivative
    with respect to anything ``problem`` closes over -- the drive, the field, a
    material property -- comes from one linearised solve at the root rather than
    from differentiating the Newton iteration.
    """
    factorization = problem.factorization()
    step = float(problem.dt if pseudo_step is None else pseudo_step)
    if step <= 0.0:
        raise ValueError("pseudo_step must be positive")
    viscous = dataclasses.replace(problem, dt=step).viscous_factorizations()
    start = zero_velocity(problem) if velocity is None else enforce_face_constraints(velocity, problem)

    def residual(state):
        return steady_residual(state, problem, factorization, forcing=forcing, field_scale=field_scale)

    scale = _norm(residual(start))

    def solver(function, guess):
        solution = solvax.newton_krylov(
            function,
            guess,
            precond=_preconditioner(problem, factorization, viscous, step),
            rtol=tolerance,
            max_steps=max_steps,
            linear_rtol=linear_tolerance,
            linear_restart=linear_restart,
            linear_max_restarts=linear_max_restarts,
        )
        return solution.x

    root = solvax.root_solve(residual, start, solver, tangent_solve=_tangent_solve)
    final = _norm(residual(root))
    checked, reference = _concrete(final), _concrete(scale)
    if (
        checked is not None
        and reference is not None
        and (not np.isfinite(checked) or checked > 10.0 * tolerance * max(reference, 1.0))
    ):
        raise RuntimeError(
            f"the steady solve did not converge: residual {checked:.3e} against an initial "
            f"{reference:.3e}. "
            "Restarted GMRES stagnates on this operator once the field is strong, so the usual cure is "
            "a larger linear_restart rather than more restarts of a short one; a conducting wall, which "
            "makes the potential solve iterative too, needs more of both"
        )
    corrected, pressure = project(root, problem, factorization)
    potential, _ = electric_state(corrected, problem, factorization, field_scale)
    return SteadySolution(corrected, pressure, potential, final, max_steps)


def _krylov(matvec, target):
    return solvax.gmres(matvec, target, rtol=1.0e-10, restart=60, max_restarts=60).x


def _tangent_solve(operator, target):
    """Solve the linearised system at the root, matrix free, in both directions.

    :func:`jax.lax.custom_linear_solve` makes the Krylov iteration opaque to
    automatic differentiation and asks for the transposed solve explicitly, so
    the adjoint runs GMRES on the transposed operator rather than differentiating
    through the forward iteration -- which cannot be transposed, because a Krylov
    basis is not a linear function of its right-hand side.
    """
    return jax.lax.custom_linear_solve(operator, target, _krylov, _krylov)


def _norm(velocity: tuple[Field, Field, Field]):
    return jnp.sqrt(sum(jnp.sum(field.data**2) for field in velocity))


def _concrete(value) -> float | None:
    """Return ``value`` as a host float, or ``None`` while it is being traced."""
    try:
        return float(value)
    except (TypeError, jax.errors.TracerArrayConversionError, jax.errors.ConcretizationTypeError):
        return None
