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
    static_argnames=("anchor", "iterations", "tolerance", "residual_scale_min", "preconditioner"),
)
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
    residual_scale_min: float | None = None,
    preconditioner: Callable[[jnp.ndarray], jnp.ndarray] | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve a symmetric anchored Poisson system with SOLVAX implicit PCG."""

    phi0 = jnp.zeros_like(rhs) if initial is None else jnp.asarray(initial).at[anchor].set(0.0)
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
    tiny = jnp.asarray(jnp.finfo(rhs.dtype).tiny, dtype=rhs.dtype)
    inverse_diagonal = 1.0 / jnp.maximum(diagonal, tiny)

    def apply_preconditioner(residual: jnp.ndarray) -> jnp.ndarray:
        # Extend the gauge-subspace preconditioner with an identity anchor.
        projected = residual.at[anchor].set(0.0)
        solved = inverse_diagonal * projected if preconditioner is None else preconditioner(projected)
        return solved.at[anchor].set(residual[anchor])

    def matvec(field: jnp.ndarray) -> jnp.ndarray:
        return apply_poisson_operator(diagonal, west, east, south, north, field, anchor)

    requested = 0.0 if tolerance is None else tolerance
    scaled_stopping = residual_scale is not None and residual_scale_min is not None
    rtol = 0.0 if scaled_stopping else requested / (rhs.size**0.5)
    atol = requested * residual_scale_min if scaled_stopping else 0.0
    solution = _solvax_pcg_linear_solve(
        matvec,
        rhs.at[anchor].set(0.0),
        x0=phi0,
        precond=apply_preconditioner,
        rtol=rtol,
        atol=atol,
        max_steps=iterations,
    )
    phi = solution.x.at[anchor].set(0.0)
    if residual_scale is None:
        residual = poisson_residual_norm(
            diagonal, west, east, south, north, rhs, phi, anchor
        )
    else:
        physical_residual = rhs - apply_five_point_operator(
            diagonal, west, east, south, north, phi
        )
        residual = jnp.max(
            jnp.abs(physical_residual) / jnp.maximum(residual_scale, tiny)
        )
    return phi, residual, solution.iterations
