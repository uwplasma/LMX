from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .mesh import StructuredMesh

try:
    import lineax as lx
except ModuleNotFoundError:  # pragma: no cover - exercised by environment-dependent paths
    lx = None


@dataclass(frozen=True)
class LinearSolveInfo:
    backend: str
    iterations: int
    residual: float


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
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    phi0 = jnp.zeros_like(rhs)
    rhs_masked = rhs.at[anchor].set(0.0)

    def jacobi_update(phi: jnp.ndarray) -> jnp.ndarray:
        west_phi = jnp.pad(phi[:-1, :], ((1, 0), (0, 0)))
        east_phi = jnp.pad(phi[1:, :], ((0, 1), (0, 0)))
        south_phi = jnp.pad(phi[:, :-1], ((0, 0), (1, 0)))
        north_phi = jnp.pad(phi[:, 1:], ((0, 0), (0, 1)))
        updated = (rhs + west * west_phi + east * east_phi + south * south_phi + north * north_phi) / diagonal
        updated = updated.at[anchor].set(0.0)
        return updated

    def residual_norm(phi: jnp.ndarray) -> jnp.ndarray:
        west_phi = jnp.pad(phi[:-1, :], ((1, 0), (0, 0)))
        east_phi = jnp.pad(phi[1:, :], ((0, 1), (0, 0)))
        south_phi = jnp.pad(phi[:, :-1], ((0, 0), (1, 0)))
        north_phi = jnp.pad(phi[:, 1:], ((0, 0), (0, 1)))
        matrix_phi = diagonal * phi - west * west_phi - east * east_phi - south * south_phi - north * north_phi
        matrix_phi = matrix_phi.at[anchor].set(0.0)
        numerator = jnp.max(jnp.abs(matrix_phi - rhs_masked))
        scale = jnp.maximum(jnp.max(jnp.abs(matrix_phi)), jnp.max(jnp.abs(rhs_masked)))
        return numerator / jnp.maximum(scale, 1e-12)

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

    init_state = (jnp.asarray(0, dtype=jnp.int32), phi0, jnp.asarray(jnp.inf, dtype=rhs.dtype))
    iteration_count, phi, residual = jax.lax.while_loop(cond_fun, body_fun, init_state)
    return phi, residual, iteration_count


def solve_poisson_jacobi(
    diagonal: jnp.ndarray,
    west: jnp.ndarray,
    east: jnp.ndarray,
    south: jnp.ndarray,
    north: jnp.ndarray,
    rhs: jnp.ndarray,
    anchor: tuple[int, int],
    iterations: int,
    tolerance: float | None = None,
) -> tuple[jnp.ndarray, LinearSolveInfo]:
    phi, residual, iteration_count = solve_poisson_jacobi_state(
        diagonal,
        west,
        east,
        south,
        north,
        rhs,
        anchor,
        iterations,
        tolerance=tolerance,
    )
    return phi, LinearSolveInfo(
        backend="jax-jacobi",
        iterations=int(iteration_count),
        residual=float(residual),
    )


def solve_poisson_lineax(
    diagonal: jnp.ndarray,
    west: jnp.ndarray,
    east: jnp.ndarray,
    south: jnp.ndarray,
    north: jnp.ndarray,
    rhs: jnp.ndarray,
    anchor: tuple[int, int],
    *,
    fallback_iterations: int = 400,
    max_steps: int = 500,
) -> tuple[jnp.ndarray, LinearSolveInfo]:
    if lx is None:
        return solve_poisson_jacobi(
            diagonal,
            west,
            east,
            south,
            north,
            rhs,
            anchor,
            iterations=fallback_iterations,
        )

    ny, nz = rhs.shape
    size = ny * nz

    def mv(vec: jnp.ndarray) -> jnp.ndarray:
        field = vec.reshape((ny, nz))
        west_phi = jnp.pad(field[:-1, :], ((1, 0), (0, 0)))
        east_phi = jnp.pad(field[1:, :], ((0, 1), (0, 0)))
        south_phi = jnp.pad(field[:, :-1], ((0, 0), (1, 0)))
        north_phi = jnp.pad(field[:, 1:], ((0, 0), (0, 1)))
        out = diagonal * field - west * west_phi - east * east_phi - south * south_phi - north * north_phi
        out = out.at[anchor].set(field[anchor])
        return out.reshape((size,))

    input_structure = jax.ShapeDtypeStruct((size,), rhs.dtype)
    op = lx.FunctionLinearOperator(mv, input_structure, tags=lx.positive_semidefinite_tag)
    rhs_vec = rhs.at[anchor].set(0.0).reshape((size,))
    solver = lx.CG(rtol=1e-10, atol=1e-10, max_steps=max_steps)
    sol = lx.linear_solve(op, rhs_vec, solver=solver)
    num_steps = sol.stats.get("num_steps", max_steps)
    iterations = int(num_steps) if isinstance(num_steps, int) else max_steps
    return sol.value.reshape((ny, nz)), LinearSolveInfo(backend="lineax-cg", iterations=iterations, residual=float("nan"))
