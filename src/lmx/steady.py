"""Steady flow with implicit derivatives, in memory independent of the iteration count.

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

Without advection and with insulating walls the residual is affine,
:math:`\\mathbf R(\\mathbf u) = A\\mathbf u + \\mathbf b`, and :math:`-A` is
symmetric positive definite on the constrained divergence-free fields in the
face-volume inner product, because the electromotive and force interpolations
are discrete adjoints. That case is one preconditioned conjugate-gradient
solve, differentiated by one more. Advection, or a conducting wall, whose
closure is not symmetric, takes matrix-free Newton-Krylov with restarted
GMRES, and the implicit function theorem differentiates its root with tangent
and transpose solves. Neither keeps more than a restart cycle of vectors.

The preconditioner projects a viscous inverse damped at :math:`\\sigma|B|^2/\\rho`
in every component, approaching :math:`(\\lambda - \\nu\\nabla^2)^{-1}` as the
pseudo-step grows. Its conditioning still degrades with the Hartmann number on
layer-resolving meshes, so the iteration count grows with it.

Primal residuals and tangent/transpose convergence are certified. Rejection
raises eagerly; during tracing, it produces nonfinite fields and derivatives
without host callbacks. Optimizers must reject nonfinite values and gradients.
"""

from __future__ import annotations

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
    velocity_offset,
    zero_velocity,
)
from .grid import Field
from .ops import face_inner_product, staggered_laplacian
from .poisson import FastDiagonalHelmholtz, FastDiagonalPoisson, fast_diagonal_helmholtz

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
    """One projection step: the conjugate-gradient and the Newton-GMRES preconditioner.

    It ends in :func:`lmx.core3d.project`, whose copy of the first periodic face
    completes the viscous solve, which returns that duplicate face as zero.

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


def _isotropic_viscous(problem: ChannelProblem, pseudo_step: float) -> tuple[FastDiagonalHelmholtz, ...]:
    """Viscous factorizations damping every component at ``sigma |B|^2 / rho``, for both routes.

    Damping only the components normal to the field, as the time step does,
    leaves a Schur-complement deficit once projected: on
    ``duct_problem(hartmann=300, cells=40)`` the preconditioned spectrum spans
    ``[2.2e-3, 2.0e3]`` and CG takes 3638 iterations; with one shift it spans
    ``[2.2e-3, 8.5]`` and CG takes 399.
    """
    rate = float(problem.conductivity) * float(np.dot(problem.magnetic_field, problem.magnetic_field))
    conditions = tuple(velocity_condition(problem.conditions, axis) for axis in range(3))
    return tuple(
        fast_diagonal_helmholtz(
            problem.grid,
            velocity_offset(component),
            conditions,
            shift=1.0 + pseudo_step * rate / float(problem.density),
            coefficient=pseudo_step * float(problem.viscosity),
            precision=problem.precision,
        )
        for component in range(3)
    )


def _face_weights(problem: ChannelProblem) -> tuple[Field, Field, Field]:
    """Return the diagonal of the face-volume inner product as a velocity."""
    conditions = tuple(velocity_condition(problem.conditions, axis) for axis in range(3))
    ones = jax.tree.map(jnp.ones_like, zero_velocity(problem))
    return jax.grad(
        lambda u: 0.5 * sum(face_inner_product(f, f, axis, conditions[axis]) for axis, f in enumerate(u))
    )(ones)


def _orthogonal_projection(
    velocity: tuple[Field, Field, Field], problem: ChannelProblem, factorization: FastDiagonalPoisson
) -> tuple[Field, Field, Field]:
    """Project onto the constrained divergence-free fields, orthogonally in the face volume.

    :func:`lmx.core3d.project` copies the first periodic face onto its duplicate,
    an oblique projection; both copies carry half the weight, so averaging them
    is the orthogonal one.
    """
    constrained = []
    for component, field in enumerate(velocity):
        data = field.data
        selection = (slice(None),) * component
        if problem.conditions[component].is_periodic:
            mean = 0.5 * (data[selection + (0,)] + data[selection + (-1,)])
            data = data.at[selection + (0,)].set(mean).at[selection + (-1,)].set(mean)
        else:
            data = data.at[selection + (0,)].set(0.0).at[selection + (-1,)].set(0.0)
        constrained.append(field.replace_data(data))
    return project(tuple(constrained), problem, factorization)[0]


def _stokes_limit_root(
    problem: ChannelProblem,
    start: tuple[Field, Field, Field],
    factorization: FastDiagonalPoisson,
    precond,
    *,
    forcing,
    field_scale,
    tolerance: float,
    max_iterations: int,
):
    """Solve the affine Stokes-limit problem with one preconditioned CG solve.

    ``R(u) = A u + b`` with ``-A`` symmetric positive definite on the constrained
    divergence-free fields ``V`` in the face-volume inner product ``W``. CG runs
    on ``y = W u`` with operator ``y -> -A W^-1 y`` and preconditioner
    ``r -> W P r``, so the residual it measures is ``R`` itself. The derivative
    is a symmetric :func:`jax.lax.custom_linear_solve`, whose operator must be
    symmetric on every vector because a cotangent is arbitrary:
    ``K y = -A Q W^-1 y + (I - Q) W^-1 y`` with ``Q`` the orthogonal projection
    onto ``V``. Its inverse projects once per solve, not per iteration.
    """
    weights = _face_weights(problem)
    inverse = jax.tree.map(jnp.reciprocal, weights)

    def operator(velocity):
        return steady_residual(
            velocity, problem, factorization, forcing=(0.0, 0.0, 0.0), field_scale=field_scale
        )

    def matvec(state):
        velocity = jax.tree.map(jnp.multiply, state, inverse)
        inside = _orthogonal_projection(velocity, problem, factorization)
        return jax.tree.map(lambda u, a, q: u - a - q, velocity, operator(inside), inside)

    def solve(_, target):
        inside = _orthogonal_projection(target, problem, factorization)
        result = solvax.pcg(
            lambda y: jax.tree.map(jnp.negative, operator(jax.tree.map(jnp.multiply, y, inverse))),
            inside,
            precond=lambda r: jax.tree.map(jnp.multiply, precond(r), weights),
            rtol=tolerance,
            max_steps=max_iterations,
        )
        kept = _certified(result.x, result.converged & jnp.isfinite(result.residual_norm), "steady CG solve")
        solution = jax.tree.map(lambda y, t, q, w: y + w * (t - q), kept, target, inside, weights)
        return solution, (result.iterations, result.residual_norm, result.converged)

    rhs = steady_residual(start, problem, factorization, forcing=forcing, field_scale=field_scale)
    step, diagnostics = jax.lax.custom_linear_solve(matvec, rhs, solve, symmetric=True, has_aux=True)
    return jax.tree.map(lambda u, y, w: u + y / w, start, step, weights), diagnostics


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
    linear_restart: int = 60,
    linear_max_restarts: int = 200,
) -> SteadySolution:
    """Find the steady state, differentiably, in memory independent of the iteration count.

    Without advection and with insulating walls the problem is affine and
    symmetric, and one preconditioned conjugate-gradient solve answers it; its
    budget is ``linear_restart * linear_max_restarts`` iterations, and
    ``linear_tolerance`` does not apply. Otherwise matrix-free Newton-Krylov
    runs restarted GMRES. The drive and ``field_scale`` are differentiable
    through implicit linear solves; close over static ``problem`` and solver
    controls when using :func:`jax.jit`. Factorizations are assembled at trace
    time. Rejected roots raise eagerly or yield nonfinite fields and
    derivatives during tracing.
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
        viscous = _isotropic_viscous(problem, step)
    start = zero_velocity(problem) if velocity is None else enforce_face_constraints(velocity, problem)

    def residual(state):
        return steady_residual(state, problem, factorization, forcing=forcing, field_scale=field_scale)

    scale = _norm(residual(start))

    precond = _preconditioner(problem, factorization, viscous, step)

    # The thin-wall closure is not symmetric yet (plan step 1.3b), so CG and its
    # symmetric adjoint are reserved for insulating walls.
    if problem.advection == "off" and not problem.conducting_walls:
        root, _ = _stokes_limit_root(
            problem,
            jax.lax.stop_gradient(_orthogonal_projection(start, problem, factorization)),
            factorization,
            precond,
            forcing=forcing,
            field_scale=field_scale,
            tolerance=tolerance,
            max_iterations=linear_restart * linear_max_restarts if max_steps > 0 else 0,
        )
        return _finish(root, residual, scale, tolerance, problem, factorization, field_scale, max_steps)

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
    return _finish(root, residual, scale, tolerance, problem, factorization, field_scale, max_steps)


def _finish(
    root, residual, scale, tolerance, problem, factorization, field_scale, max_steps
) -> SteadySolution:
    """Certify the root on the residual both routes share, then report its fields."""
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
