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

Without advection the residual is affine,
:math:`\\mathbf R(\\mathbf u) = A\\mathbf u + \\mathbf b`, and :math:`-A` is
symmetric positive definite on the constrained divergence-free fields in the
face-volume inner product, because the electromotive and force interpolations
are discrete adjoints and the thin-wall closure is a symmetric direct solve.
That case is one preconditioned conjugate-gradient solve, differentiated by
one more. Advection takes matrix-free Newton-Krylov with restarted GMRES, and
the implicit function theorem differentiates its root with tangent and
transpose solves. Neither keeps more than a restart cycle of vectors.

The preconditioner is a projected per-component inverse. A component along a
periodic axis, across an axis-aligned field, is solved exactly along each field
line in the discrete induction form of the insulating duct, and approximately
across the lines. The other components take a viscous inverse damped at
:math:`\\sigma|B|^2/\\rho`. On ``duct_problem`` meshes of 48 cells, CG needs
4 / 15 / 44 / 79 iterations at Ha 20 / 100 / 300 / 1000, against 33 / 148 /
403 / 804 with the damped inverse alone, and 68 against 714 on 64 cells at Ha 1000.

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
from .grid import CENTER, FACE, Field
from .ops import face_inner_product, staggered_laplacian
from .poisson import (
    FastDiagonalHelmholtz,
    FastDiagonalPoisson,
    assemble_staggered_axis_operator,
    fast_diagonal_helmholtz,
)

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
    viscous: tuple,
    pseudo_step: float,
):
    """One projection step: the conjugate-gradient and the Newton-GMRES preconditioner.

    ``viscous`` holds one solve per component, from :func:`_projection_solves`.
    It ends in :func:`lmx.core3d.project`, whose copy of the first periodic face
    completes a solve that returns that duplicate face as zero.

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


_factor_lines = jax.jit(jax.vmap(functools.partial(solvax.lu_factor_banded, lower_bw=2, upper_bw=2)))


class _FieldLine:
    """The projection-step solve of a velocity component across an axis-aligned field.

    With insulating walls and a flow invariant along the component, the
    divergence-free face currents are the discrete curl of a streamfunction on the
    cell edges, so the component's Lorentz operator is ``lambda K* L^-1 K``: ``K``
    is the compact gradient along the field after the average onto the
    electromotive faces, ``L`` the edge Laplacian. Its inverse is the velocity
    block of ``[[S, tau c K*], [-tau c K, S_e]]``, ``S = 1 - tau nu Lap`` at both
    positions and ``c^2 = lambda nu``: Shercliff's induction form. Along the field
    the block is pentadiagonal with velocity and streamfunction interleaved, and
    each line is factorized once. Across it the velocity eigenbasis carries the
    block, each mode's average represented by its norm and the edge Laplacian by
    its Rayleigh quotient, which is exact on uniform or periodic axes. A wide
    collocated difference in place of ``K`` misses the checkerboard modes, and its
    condition number grows with the Hartmann number instead.
    """

    def __init__(self, problem: ChannelProblem, component: int, axis: int, pseudo_step: float):
        grid, coefficient = problem.grid, pseudo_step * float(problem.viscosity)
        conditions = tuple(velocity_condition(problem.conditions, position) for position in range(3))
        offset = velocity_offset(component)
        edge = tuple(CENTER if position == component else FACE for position in range(3))
        velocity = fast_diagonal_helmholtz(grid, offset, conditions, coefficient=coefficient)
        stream = fast_diagonal_helmholtz(grid, edge, conditions, coefficient=coefficient)
        self.velocity, self.axis = velocity, axis
        self.across = [position for position in range(3) if position != axis]
        weights, laplacians = [], []
        for at in self.across:
            average = _average(np.asarray(grid.widths[at]), offset[at], conditions[at])
            unscaled = velocity.vectors[at] / velocity.scales[at][:, None]
            modes = stream.vectors[at].T @ (stream.scales[at][:, None] * (average @ unscaled))
            weights.append(np.sum(modes**2, axis=0))
            laplacians.append(stream.values[at] @ modes**2 / np.maximum(weights[-1], np.finfo(float).tiny))

        def pair(first, second):
            return (first[:, None] + second[None, :]).reshape(-1)

        rate = float(problem.conductivity) * float(np.dot(problem.magnetic_field, problem.magnetic_field))
        speed = pseudo_step * np.sqrt(rate * float(problem.viscosity) / float(problem.density))
        shift = 1.0 - coefficient * pair(*(velocity.values[at] for at in self.across))
        coefficients = [np.ones_like(shift), shift, 1.0 - coefficient * pair(*laplacians)]
        coefficients.append(speed * np.sqrt(np.outer(*weights).reshape(-1)))
        widths = np.asarray(grid.widths[axis])
        size, distances = 2 * widths.size - 1, 0.5 * (widths[:-1] + widths[1:])
        gradient = np.diff(np.eye(widths.size), axis=0) / distances[:, None]
        u, e = np.arange(0, size, 2), np.arange(1, size, 2)
        blocks = np.zeros((4, size, size))
        for index, at in ((u, offset), (e, edge)):
            operator = assemble_staggered_axis_operator(grid, axis, at, conditions[axis])
            blocks[0][np.ix_(index, index)] = -coefficient * operator
        blocks[1][u, u], blocks[2][e, e] = 1.0, 1.0
        blocks[3][np.ix_(u, e)] = gradient.T * distances[None, :] / widths[:, None]
        blocks[3][np.ix_(e, u)] = -gradient
        # solvax band storage: bands[r, j] = A[j + r - 2, j].
        rows = np.arange(5)[:, None] + np.arange(size)[None, :] - 2
        inside = (rows >= 0) & (rows < size)
        bands = np.where(inside, blocks[:, np.clip(rows, 0, size - 1), np.arange(size)], 0.0)
        self.factors = _factor_lines(jnp.asarray(np.einsum("km,krj->mrj", np.stack(coefficients), bands)))

    def solve(self, rhs: Field) -> Field:
        velocity = self.velocity
        data = rhs.data[velocity.slices]
        for at in self.across:
            data = _modal(data * _along(velocity.scales[at], at, data), velocity.vectors[at].T, at)
        lines = jnp.moveaxis(data, self.axis, -1)
        flat = lines.reshape(-1, lines.shape[-1])
        interleaved = jnp.zeros((flat.shape[0], 2 * flat.shape[1] - 1), flat.dtype).at[:, ::2].set(flat)
        flat = jax.vmap(solvax.lu_solve_banded)(self.factors, interleaved)[:, ::2]
        data = jnp.moveaxis(flat.reshape(lines.shape), -1, self.axis)
        for at in self.across:
            data = _modal(data, velocity.vectors[at], at) / _along(velocity.scales[at], at, data)
        return rhs.replace_data(jnp.zeros_like(rhs.data).at[velocity.slices].set(data))


def _along(values: np.ndarray, axis: int, like: jnp.ndarray) -> jnp.ndarray:
    shape = [-1 if position == axis else 1 for position in range(3)]
    return jnp.asarray(values.reshape(shape), dtype=like.dtype)


def _modal(data: jnp.ndarray, matrix: np.ndarray, axis: int) -> jnp.ndarray:
    matrix = jnp.asarray(matrix, dtype=data.dtype)
    return jnp.moveaxis(jnp.tensordot(matrix, data, axes=([1], [axis])), 0, axis)


def _average(widths: np.ndarray, position: float, condition) -> np.ndarray:
    """The electromotive average between free entries: centres to faces, or its half-weight transpose."""
    count = widths.size
    if condition.is_periodic:
        faces, left, right = np.arange(count), (np.arange(count) - 1) % count, np.arange(count)
    else:
        faces, left, right = np.arange(count - 1), np.arange(count - 1), np.arange(1, count)
    share = widths[left] / (widths[left] + widths[right])
    matrix = np.zeros((faces.size, count) if position == CENTER else (count, faces.size))
    for cells, weight in ((left, share), (right, 1.0 - share)):
        if position == CENTER:
            np.add.at(matrix, (faces, cells), weight)
        else:
            np.add.at(matrix, (cells, faces), 0.5)
    return matrix


def _projection_solves(problem: ChannelProblem, pseudo_step: float) -> tuple:
    """Field lines for a periodic-axis component across an axis-aligned walled field; damped otherwise.

    The induction form is exact for a component along an axis the flow is
    invariant on, which only a periodic axis can be. The other components keep one
    common damping, since the projection couples them and different shifts
    reopen the Schur-complement deficit. Field lines on those as well gave CG 85
    iterations instead of 44 on ``duct_problem(hartmann=300, cells=48)``, and
    1408 instead of 540 on the Ha 1000 test mesh.
    """
    solves = list(_isotropic_viscous(problem, pseudo_step))
    field = np.asarray(problem.magnetic_field, dtype=float) * float(problem.conductivity)
    axis = int(np.argmax(np.abs(field)))
    if np.count_nonzero(field) == 1 and not problem.conditions[axis].is_periodic:
        for component in range(3):
            if component != axis and problem.conditions[component].is_periodic:
                solves[component] = _FieldLine(problem, component, axis, pseudo_step)
    return tuple(solves)


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

    Without advection, with insulating or thin conducting walls, the problem is affine and
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
        viscous = _projection_solves(problem, step)
    start = zero_velocity(problem) if velocity is None else enforce_face_constraints(velocity, problem)

    def residual(state):
        return steady_residual(state, problem, factorization, forcing=forcing, field_scale=field_scale)

    scale = _norm(residual(start))

    precond = _preconditioner(problem, factorization, viscous, step)

    if problem.advection == "off":
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
