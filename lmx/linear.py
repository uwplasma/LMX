from __future__ import annotations

from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp

from solvax import pcg_linear_solve as _solvax_pcg_linear_solve


def apply_five_point_operator(
    diagonal: jnp.ndarray,
    west: jnp.ndarray,
    east: jnp.ndarray,
    south: jnp.ndarray,
    north: jnp.ndarray,
    field: jnp.ndarray,
) -> jnp.ndarray:
    west_field = jnp.pad(field[:-1, :], ((1, 0), (0, 0)))
    east_field = jnp.pad(field[1:, :], ((0, 1), (0, 0)))
    south_field = jnp.pad(field[:, :-1], ((0, 0), (1, 0)))
    north_field = jnp.pad(field[:, 1:], ((0, 0), (0, 1)))
    return (
        diagonal * field
        - west * west_field
        - east * east_field
        - south * south_field
        - north * north_field
    )


def five_point_residual_norm(
    diagonal: jnp.ndarray,
    west: jnp.ndarray,
    east: jnp.ndarray,
    south: jnp.ndarray,
    north: jnp.ndarray,
    rhs: jnp.ndarray,
    field: jnp.ndarray,
) -> jnp.ndarray:
    applied = apply_five_point_operator(diagonal, west, east, south, north, field)
    numerator = jnp.max(jnp.abs(applied - rhs))
    scale = jnp.maximum(jnp.max(jnp.abs(applied)), jnp.max(jnp.abs(rhs)))
    return numerator / jnp.maximum(scale, 1e-12)


def apply_poisson_operator(
    diagonal: jnp.ndarray,
    west: jnp.ndarray,
    east: jnp.ndarray,
    south: jnp.ndarray,
    north: jnp.ndarray,
    phi: jnp.ndarray,
    anchor: tuple[int, int],
) -> jnp.ndarray:
    projected = phi.at[anchor].set(0.0)
    west_phi = jnp.pad(projected[:-1, :], ((1, 0), (0, 0)))
    east_phi = jnp.pad(projected[1:, :], ((0, 1), (0, 0)))
    south_phi = jnp.pad(projected[:, :-1], ((0, 0), (1, 0)))
    north_phi = jnp.pad(projected[:, 1:], ((0, 0), (0, 1)))
    matrix_phi = (
        diagonal * projected
        - west * west_phi
        - east * east_phi
        - south * south_phi
        - north * north_phi
    )
    return matrix_phi.at[anchor].set(phi[anchor])


def poisson_residual_norm(
    diagonal: jnp.ndarray,
    west: jnp.ndarray,
    east: jnp.ndarray,
    south: jnp.ndarray,
    north: jnp.ndarray,
    rhs: jnp.ndarray,
    phi: jnp.ndarray,
    anchor: tuple[int, int],
) -> jnp.ndarray:
    rhs_masked = rhs.at[anchor].set(0.0)
    matrix_phi = apply_poisson_operator(diagonal, west, east, south, north, phi, anchor)
    numerator = jnp.max(jnp.abs(matrix_phi - rhs_masked))
    scale = jnp.maximum(jnp.max(jnp.abs(matrix_phi)), jnp.max(jnp.abs(rhs_masked)))
    return numerator / jnp.maximum(scale, 1e-12)


def solve_poisson_jacobi_state(
    diagonal: jnp.ndarray,
    west: jnp.ndarray,
    east: jnp.ndarray,
    south: jnp.ndarray,
    north: jnp.ndarray,
    rhs: jnp.ndarray,
    anchor: tuple[int, int],
    iterations: int,
    tolerance: float | None = None,
    relaxation: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    phi0 = jnp.zeros_like(rhs)

    omega = jnp.asarray(relaxation, dtype=rhs.dtype)

    def jacobi_update(phi: jnp.ndarray) -> jnp.ndarray:
        west_phi = jnp.pad(phi[:-1, :], ((1, 0), (0, 0)))
        east_phi = jnp.pad(phi[1:, :], ((0, 1), (0, 0)))
        south_phi = jnp.pad(phi[:, :-1], ((0, 0), (1, 0)))
        north_phi = jnp.pad(phi[:, 1:], ((0, 0), (0, 1)))
        updated = (
            rhs
            + west * west_phi
            + east * east_phi
            + south * south_phi
            + north * north_phi
        ) / diagonal
        blended = (1.0 - omega) * phi + omega * updated
        blended = blended.at[anchor].set(0.0)
        return blended

    def residual_norm(phi: jnp.ndarray) -> jnp.ndarray:
        return poisson_residual_norm(
            diagonal, west, east, south, north, rhs, phi, anchor
        )

    if tolerance is None or tolerance <= 0.0:
        phi = jax.lax.fori_loop(0, iterations, lambda i, p: jacobi_update(p), phi0)
        residual = residual_norm(phi)
        return phi, residual, jnp.asarray(iterations, dtype=jnp.int32)

    tolerance_value = jnp.asarray(tolerance, dtype=rhs.dtype)

    def cond_fun(state):
        count, _, residual = state
        return jnp.logical_and(count < iterations, residual > tolerance_value)

    def body_fun(state):
        count, phi, _ = state
        updated = jacobi_update(phi)
        residual = residual_norm(updated)
        return count + 1, updated, residual

    init_state = (
        jnp.asarray(0, dtype=jnp.int32),
        phi0,
        jnp.asarray(jnp.inf, dtype=rhs.dtype),
    )
    iteration_count, phi, residual = jax.lax.while_loop(cond_fun, body_fun, init_state)
    return phi, residual, iteration_count


@partial(
    jax.jit,
    static_argnames=("iterations", "tolerance", "preconditioner"),
)
def solve_five_point_solvax_pcg_state(
    diagonal: jnp.ndarray,
    west: jnp.ndarray,
    east: jnp.ndarray,
    south: jnp.ndarray,
    north: jnp.ndarray,
    rhs: jnp.ndarray,
    iterations: int,
    tolerance: float | None = None,
    preconditioner: str = "jacobi",
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve a five-point SPD system with the released SOLVAX PCG backend."""

    tiny = jnp.asarray(jnp.finfo(rhs.dtype).tiny, dtype=rhs.dtype)
    inverse_diagonal = 1.0 / jnp.maximum(diagonal, tiny)

    def matvec(field: jnp.ndarray) -> jnp.ndarray:
        return apply_five_point_operator(diagonal, west, east, south, north, field)

    if preconditioner in {"jacobi", "block_jacobi"}:

        def apply_preconditioner(residual: jnp.ndarray) -> jnp.ndarray:
            return inverse_diagonal * residual

    elif preconditioner == "none":

        def apply_preconditioner(residual: jnp.ndarray) -> jnp.ndarray:
            return residual

    else:
        raise ValueError(f"Unsupported preconditioner {preconditioner!r}")

    # SOLVAX uses an L2 relative residual while LMX certifies the returned
    # solution with its physical max-norm residual. The sqrt(N) factor makes
    # the inner criterion at least as strict for a same-order field scale.
    requested = 0.0 if tolerance is None else tolerance
    l2_tolerance = requested / (rhs.size**0.5)
    solution = _solvax_pcg_linear_solve(
        matvec,
        rhs,
        precond=apply_preconditioner,
        rtol=l2_tolerance,
        atol=0.0,
        max_steps=iterations,
        transpose_rtol=l2_tolerance,
        transpose_atol=0.0,
        transpose_max_steps=iterations,
    )
    residual = five_point_residual_norm(
        diagonal, west, east, south, north, rhs, solution.x
    )
    return solution.x, residual, solution.iterations


@partial(jax.jit, static_argnames=("anchor", "preconditioner"))
def solve_poisson_cg_state(
    diagonal: jnp.ndarray,
    west: jnp.ndarray,
    east: jnp.ndarray,
    south: jnp.ndarray,
    north: jnp.ndarray,
    rhs: jnp.ndarray,
    anchor: tuple[int, int],
    iterations: int,
    tolerance: float | None = None,
    initial: jnp.ndarray | None = None,
    residual_scale: jnp.ndarray | None = None,
    preconditioner: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    phi0 = (
        jnp.zeros_like(rhs)
        if initial is None
        else jnp.asarray(initial).at[anchor].set(0.0)
    )
    if phi0.shape != rhs.shape:
        raise ValueError(
            "Poisson CG initial guess must match the right-hand side shape"
        )
    if residual_scale is not None:
        residual_scale = jnp.asarray(residual_scale)
        if residual_scale.shape != rhs.shape:
            raise ValueError(
                "Poisson CG residual scale must match the right-hand side shape"
            )
    rhs_masked = rhs.at[anchor].set(0.0)
    inv_diagonal = 1.0 / jnp.maximum(diagonal, 1e-12)
    tolerance_value = (
        -jnp.ones((), dtype=rhs.dtype)
        if tolerance is None
        else jnp.asarray(tolerance, dtype=rhs.dtype)
    )
    tiny = jnp.asarray(jnp.finfo(rhs.dtype).tiny, dtype=rhs.dtype)

    residual0 = rhs_masked - apply_poisson_operator(
        diagonal, west, east, south, north, phi0, anchor
    )

    def apply_preconditioner(residual: jnp.ndarray) -> jnp.ndarray:
        return (
            inv_diagonal * residual
            if preconditioner is None
            else preconditioner(residual)
        )

    z0 = apply_preconditioner(residual0)
    p0 = z0
    rz0 = jnp.sum(residual0 * z0)
    if residual_scale is None:
        norm0 = poisson_residual_norm(
            diagonal, west, east, south, north, rhs, phi0, anchor
        )
    else:
        physical_residual0 = rhs - apply_five_point_operator(
            diagonal, west, east, south, north, phi0
        )
        norm0 = jnp.max(
            jnp.abs(physical_residual0) / jnp.maximum(residual_scale, 1.0e-30)
        )

    def cond_fun(state):
        count, _, residual, _, _, rz_old, active = state
        return jnp.logical_and(
            count < iterations, jnp.logical_and(active, residual > tolerance_value)
        )

    def body_fun(state):
        count, phi, residual, r, p, rz_old, _ = state
        ap = apply_poisson_operator(diagonal, west, east, south, north, p, anchor)
        denom = jnp.sum(p * ap)
        safe_denom = jnp.where(jnp.abs(denom) > tiny, denom, 1.0)
        alpha = rz_old / safe_denom
        phi_next = phi + alpha * p
        phi_next = phi_next.at[anchor].set(0.0)
        r_next = r - alpha * ap
        z_next = apply_preconditioner(r_next)
        rz_next = jnp.sum(r_next * z_next)
        safe_rz_old = jnp.where(jnp.abs(rz_old) > tiny, rz_old, 1.0)
        beta = rz_next / safe_rz_old
        p_next = z_next + beta * p
        if residual_scale is None:
            residual_next = poisson_residual_norm(
                diagonal, west, east, south, north, rhs, phi_next, anchor
            )
        else:
            physical_residual = rhs - apply_five_point_operator(
                diagonal, west, east, south, north, phi_next
            )
            residual_next = jnp.max(
                jnp.abs(physical_residual) / jnp.maximum(residual_scale, 1.0e-30)
            )
        active_next = jnp.logical_and(jnp.abs(denom) > tiny, rz_next > tiny)
        return count + 1, phi_next, residual_next, r_next, p_next, rz_next, active_next

    init_state = (
        jnp.asarray(0, dtype=jnp.int32),
        phi0,
        norm0,
        residual0,
        p0,
        rz0,
        jnp.asarray(rz0 > tiny),
    )
    iteration_count, phi, residual, _, _, _, _ = jax.lax.while_loop(
        cond_fun, body_fun, init_state
    )
    return phi, residual, iteration_count
