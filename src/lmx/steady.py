"""Matrix-free Newton-Krylov steady flow with implicit derivatives.

The implicit function theorem differentiates converged roots with tangent and
transpose solves, without retaining nonlinear iterations.

Potential and pressure are eliminated through their discrete linear solves,
leaving a residual in velocity alone:

.. math:: \\mathbf R(\\mathbf u)
   = \\mathbb P\\left[\\nu\\nabla^2\\mathbf u
     + (\\mathbf F(\\mathbf u) + \\mathbf f)/\\rho
     - \\nabla\\cdot(\\mathbf u\\mathbf u)\\right],

with :math:`\\mathbb P` the discrete projection onto the divergence-free face
fields and :math:`\\mathbf F` the Ni face-form Lorentz force of the potential
that :math:`\\mathbf u` induces. A root of :math:`\\mathbf R` is a steady state
of :func:`lmx.core3d.step`, and the projection keeps the iteration inside the
subspace the time stepper never leaves.

The preconditioner projects a viscous/damping inverse, approaching
:math:`(\\lambda - \\nu\\nabla^2)^{-1}` as the pseudo-step grows.

Strong fields can require a large Krylov subspace rather than more restarts
of a short one; conducting walls also make the potential solve iterative.

Primal residuals and tangent/transpose convergence are certified. Rejection
raises eagerly; during tracing, it produces nonfinite fields and derivatives
without host callbacks. Optimizers must reject nonfinite values and gradients.
"""

from __future__ import annotations

import dataclasses
import functools
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
    face_lorentz_force,
    project,
    velocity_condition,
    zero_velocity,
)
from .grid import Field
from .ops import staggered_laplacian
from .poisson import FastDiagonalHelmholtz, FastDiagonalPoisson

__all__ = ["SteadySolution", "solve_steady_state", "steady_residual"]


@dataclass(frozen=True)
class SteadySolution:
    """Steady fields and residual evidence; traced rejection gives nonfinite fields."""

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
    drive = problem.forcing if forcing is None else forcing
    _, force = electric_state(velocity, problem, factorization, field_scale)
    body = face_lorentz_force(force, problem)
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
        value = (
            problem.viscosity * staggered_laplacian(field, conditions).data
            + (body[component].data + drive[component]) / problem.density
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

    The drive and ``field_scale`` are differentiable through :func:`solvax.root_solve`;
    close over static ``problem`` and solver controls when using :func:`jax.jit`.
    Factorizations are assembled at trace time. Rejected roots raise eagerly
    or yield nonfinite fields and derivatives during tracing.
    """
    step = float(problem.dt if pseudo_step is None else pseudo_step)
    for name, value in (
        ("pseudo_step", step),
        ("tolerance", tolerance),
        ("linear_tolerance", linear_tolerance),
    ):
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite")
    with jax.ensure_compile_time_eval():
        factorization = problem.factorization()
        viscous = dataclasses.replace(problem, dt=step).viscous_factorizations()
    start = zero_velocity(problem) if velocity is None else enforce_face_constraints(velocity, problem)

    def residual(state):
        return steady_residual(state, problem, factorization, forcing=forcing, field_scale=field_scale)

    scale = _norm(residual(start))

    precond = _preconditioner(problem, factorization, viscous, step)

    def solver(function, guess):
        solution = solvax.newton_krylov(
            function,
            guess,
            precond=precond,
            rtol=tolerance,
            max_steps=max_steps,
            linear_rtol=linear_tolerance,
            linear_restart=linear_restart,
            linear_max_restarts=linear_max_restarts,
        )
        return solution.x

    tangent_solve = functools.partial(_tangent_solve, precond=precond)
    root = solvax.root_solve(residual, start, solver, tangent_solve=tangent_solve)
    final = _norm(residual(root))
    accepted = (
        jnp.isfinite(final) & jnp.isfinite(scale) & (final <= 10.0 * tolerance * jnp.maximum(scale, 1.0))
    )
    root = _certified(root, accepted, "steady solve")
    corrected, pressure = project(root, problem, factorization)
    potential, _ = electric_state(corrected, problem, factorization, field_scale)
    return SteadySolution(corrected, pressure, potential, final, max_steps)


def _krylov(matvec, target, precond=None):
    result = solvax.gmres(matvec, target, precond=precond, rtol=1.0e-10, restart=60, max_restarts=60)
    return _certified(result.x, result.converged & jnp.isfinite(result.residual_norm), "steady linear solve")


def _certified(value, accepted, stage):
    """Reject eagerly; multiply by NaN under tracing so failed gradients fail too."""
    traced = any(isinstance(leaf, jax.core.Tracer) for leaf in jax.tree.leaves((value, accepted)))
    if not traced and not bool(accepted):
        raise RuntimeError(f"the {stage} did not converge")
    return jax.tree.map(lambda leaf: leaf * jnp.where(accepted, 1.0, jnp.nan), value)


def _tangent_solve(operator, target, precond=None):
    """Solve the linearised system at the root, matrix free, in both directions.

    :func:`jax.lax.custom_linear_solve` makes the Krylov iteration opaque to
    automatic differentiation and asks for the transposed solve explicitly, so
    the adjoint runs GMRES on the transposed operator rather than differentiating
    through the forward iteration -- which cannot be transposed, because a Krylov
    basis is not a linear function of its right-hand side.

    ``precond`` is the primal projection step. The transposed solve uses its
    exact transpose: the step is symmetric only in the face-volume inner product,
    and GMRES measures in the Euclidean one, where a stretched mesh makes the two
    differ by the width ratio.
    """

    def solve(matvec, rhs):
        return _krylov(matvec, rhs, precond)

    def transpose_solve(vecmat, rhs):
        if precond is None:
            return _krylov(vecmat, rhs)
        transposed = jax.linear_transpose(precond, rhs)
        return _krylov(vecmat, rhs, lambda direction: transposed(direction)[0])

    return jax.lax.custom_linear_solve(operator, target, solve, transpose_solve)


def _norm(velocity: tuple[Field, Field, Field]):
    return jnp.sqrt(sum(jnp.sum(field.data**2) for field in velocity))
