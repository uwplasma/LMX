"""Run the projection step as one compiled trajectory.

A Python loop around :func:`lmx.core3d.step` dispatches every operation from the
host. On an accelerator that is the difference between a kernel queue the device
can run ahead on and a round trip per step, and it is the reason the audit that
opened this plan found one GPU slower than a laptop CPU on a small duct. The
whole trajectory is compiled here instead, with :func:`jax.lax.scan`.

Reverse-mode differentiation of a long trajectory is the other half of the
problem. Storing every intermediate state is what exhausts device memory, so the
step is wrapped in :func:`jax.checkpoint`: the forward pass keeps one state per
step and the backward pass recomputes the interior of each step. SOLVAX's
``checkpointed_fori_loop`` offers the square-root schedule for the deeper case;
per-step rematerialisation is the simpler contract and is what the momentum step
needs, since its expensive parts are the two Poisson solves rather than a long
chain of cheap operations.

Diagnostics come out as scan outputs rather than through a host callback, so a
run reports its history without ever synchronising mid-trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from .core3d import ChannelProblem, step, zero_velocity
from .grid import Field
from .ops import divergence
from .poisson import FastDiagonalPoisson

__all__ = ["Trajectory", "advance", "trajectory_diagnostics"]


@dataclass(frozen=True)
class Trajectory:
    """The end state of a run and the per-step history of its diagnostics."""

    velocity: tuple[Field, Field, Field]
    pressure: Field
    potential: Field
    divergence_residual: jnp.ndarray
    kinetic_energy: jnp.ndarray

    @property
    def steps(self) -> int:
        return int(self.divergence_residual.shape[0])


def trajectory_diagnostics(
    velocity: tuple[Field, Field, Field], problem: ChannelProblem
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the divergence residual and kinetic energy of one state."""
    residual = jnp.max(jnp.abs(divergence(velocity).data))
    volumes = jnp.asarray(problem.grid.cell_volumes(), dtype=velocity[0].dtype)
    energy = (
        0.5
        * float(problem.density)
        * sum(jnp.sum(volumes * _cell_average(component).data ** 2) for component in velocity)
    )
    return residual, energy


def _cell_average(field: Field) -> Field:
    """Average a face component onto cell centres for an energy that lives on cells."""
    from .em import cell_average

    axis = next(index for index, offset in enumerate(field.offset) if offset == 0.0)
    return cell_average(field, axis)


def advance(
    problem: ChannelProblem,
    steps: int,
    velocity: tuple[Field, Field, Field] | None = None,
    *,
    factorization: FastDiagonalPoisson | None = None,
    checkpoint: bool = True,
) -> Trajectory:
    """Run ``steps`` projection steps as one compiled scan.

    ``steps`` is static: it fixes the length of the compiled trajectory. The
    factorization is built once on the host, outside the trace, because its
    assembly reads concrete arrays.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    factorization = problem.factorization() if factorization is None else factorization
    velocity = zero_velocity(problem) if velocity is None else velocity

    def single(state, _):
        updated, pressure, potential = step(state, problem, factorization)
        return updated, (*trajectory_diagnostics(updated, problem), pressure, potential)

    body = jax.checkpoint(single) if checkpoint else single
    final, (residual, energy, pressures, potentials) = jax.lax.scan(body, velocity, xs=None, length=steps)
    last = jax.tree_util.tree_map(lambda leaf: leaf[-1], (pressures, potentials))
    return Trajectory(final, last[0], last[1], residual, energy)
