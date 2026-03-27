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


def solve_poisson_jacobi(
    diagonal: jnp.ndarray,
    west: jnp.ndarray,
    east: jnp.ndarray,
    south: jnp.ndarray,
    north: jnp.ndarray,
    rhs: jnp.ndarray,
    anchor: tuple[int, int],
    iterations: int,
) -> tuple[jnp.ndarray, LinearSolveInfo]:
    phi0 = jnp.zeros_like(rhs)

    def body(_, phi):
        west_phi = jnp.pad(phi[:-1, :], ((1, 0), (0, 0)))
        east_phi = jnp.pad(phi[1:, :], ((0, 1), (0, 0)))
        south_phi = jnp.pad(phi[:, :-1], ((0, 0), (1, 0)))
        north_phi = jnp.pad(phi[:, 1:], ((0, 0), (0, 1)))
        updated = (rhs + west * west_phi + east * east_phi + south * south_phi + north * north_phi) / diagonal
        updated = updated.at[anchor].set(0.0)
        return updated

    phi = jax.lax.fori_loop(0, iterations, lambda i, p: body(i, p), phi0)
    return phi, LinearSolveInfo(backend="jax-jacobi", iterations=iterations)


def solve_poisson_lineax(
    diagonal: jnp.ndarray,
    west: jnp.ndarray,
    east: jnp.ndarray,
    south: jnp.ndarray,
    north: jnp.ndarray,
    rhs: jnp.ndarray,
    anchor: tuple[int, int],
) -> tuple[jnp.ndarray, LinearSolveInfo]:
    if lx is None:
        return solve_poisson_jacobi(diagonal, west, east, south, north, rhs, anchor, iterations=400)

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

    op = lx.FunctionLinearOperator(mv, (size, size), tags=lx.positive_semidefinite_tag)
    rhs_vec = rhs.at[anchor].set(0.0).reshape((size,))
    solver = lx.CG(rtol=1e-10, atol=1e-10, max_steps=500)
    sol = lx.linear_solve(op, rhs_vec, solver=solver)
    return sol.value.reshape((ny, nz)), LinearSolveInfo(backend="lineax-cg", iterations=int(sol.stats["num_steps"]))
