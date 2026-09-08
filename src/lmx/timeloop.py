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

from .core3d import ChannelProblem, face_currents, step, velocity_condition, zero_velocity
from .em import lorentz_force
from .grid import Field
from .ops import divergence, face_inner_product, face_interpolate, staggered_laplacian
from .poisson import FastDiagonalHelmholtz, FastDiagonalPoisson

__all__ = [
    "EnergyBudget",
    "Trajectory",
    "advance",
    "energy_budget",
    "kinetic_energy",
    "trajectory_diagnostics",
]


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


def kinetic_energy(velocity: tuple[Field, Field, Field], problem: ChannelProblem) -> jnp.ndarray:
    """Return the kinetic energy of a staggered velocity.

    Each component is weighted by the control volume of its own faces, the same
    measure under which the divergence and the gradient are exact adjoints. That
    is what makes the energy budget close: a cell-centred average would leave a
    defect of the order of the interpolation error rather than of round-off.
    """
    conditions = tuple(velocity_condition(problem.conditions, axis) for axis in range(3))
    return (
        0.5
        * float(problem.density)
        * sum(face_inner_product(field, field, axis, conditions[axis]) for axis, field in enumerate(velocity))
    )


def trajectory_diagnostics(
    velocity: tuple[Field, Field, Field], problem: ChannelProblem
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return the divergence residual and kinetic energy of one state."""
    return jnp.max(jnp.abs(divergence(velocity).data)), kinetic_energy(velocity, problem)


@dataclass(frozen=True)
class EnergyBudget:
    """The mechanical power crossing one control volume: the whole duct.

    The pressure does no work on a discretely divergence-free velocity, because
    the divergence and the gradient are exact adjoints under this measure, so
    the balance is between what the drive and the field put in and what
    viscosity takes out. Two independent statements come out of it, and they are
    worth keeping apart. ``defect`` is what the steady solve promises: the
    residual, projected onto the velocity, is zero. ``ohmic_defect`` is a
    property of the discretization instead -- continuously the Lorentz force
    does exactly minus the Joule dissipation, and how nearly the discrete face
    currents reproduce that is a measurement, not a guarantee.
    """

    drive: jnp.ndarray
    lorentz: jnp.ndarray
    viscous: jnp.ndarray
    joule: jnp.ndarray
    wall: jnp.ndarray

    @property
    def defect(self) -> jnp.ndarray:
        """Net power into the fluid, which is the rate of change of kinetic energy."""
        return self.drive + self.lorentz - self.viscous

    @property
    def ohmic_defect(self) -> jnp.ndarray:
        """How far the discrete Lorentz work is from the dissipation it should equal.

        A conducting wall takes current out of the fluid and dissipates it in the
        sheet, so the fluid's own Joule term is not the whole of it; ``wall`` is
        the electrical power crossing the boundary, and leaving it out of the
        comparison is a 26 % error at a wall conductance of 0.027.
        """
        return self.joule + self.wall + self.lorentz

    @property
    def scale(self) -> jnp.ndarray:
        """The largest term, for turning either defect into a relative number."""
        return jnp.max(jnp.abs(jnp.stack([self.drive, self.lorentz, self.viscous, self.joule, self.wall])))


def energy_budget(
    velocity: tuple[Field, Field, Field],
    problem: ChannelProblem,
    factorization: FastDiagonalPoisson | None = None,
) -> EnergyBudget:
    """Return the mechanical power balance of one state."""
    factorization = problem.factorization() if factorization is None else factorization
    scalar = problem.scalar_conditions
    conditions = tuple(velocity_condition(problem.conditions, axis) for axis in range(3))
    potential, currents, field = face_currents(velocity, problem, factorization)
    force = lorentz_force(currents, field, scalar)
    density = float(problem.density)
    drive = sum(
        float(problem.forcing[axis])
        * face_inner_product(
            component, component.replace_data(jnp.ones_like(component.data)), axis, conditions[axis]
        )
        for axis, component in enumerate(velocity)
    )
    lorentz = sum(
        face_inner_product(
            component, face_interpolate(force[axis], axis, scalar[axis]), axis, conditions[axis]
        )
        for axis, component in enumerate(velocity)
    )
    viscous = (
        -density
        * float(problem.viscosity)
        * sum(
            face_inner_product(component, staggered_laplacian(component, conditions), axis, conditions[axis])
            for axis, component in enumerate(velocity)
        )
    )
    conductivity = float(problem.conductivity)
    joule = (
        sum(face_inner_product(current, current, axis, scalar[axis]) for axis, current in enumerate(currents))
        / conductivity
        if conductivity
        else jnp.zeros(())
    )
    return EnergyBudget(
        jnp.asarray(drive),
        jnp.asarray(lorentz),
        jnp.asarray(viscous),
        jnp.asarray(joule),
        jnp.asarray(_wall_power(potential, currents, problem)),
    )


def _wall_power(potential: Field, currents: tuple[Field, Field, Field], problem: ChannelProblem):
    """Return the electrical power the fluid delivers to its conducting walls.

    The outward current at a wall face times the potential there, which is what
    the sheet dissipates. The stored face value points along the axis, so the
    outward direction is negative on the lower wall.
    """
    grid = problem.grid
    scalar = problem.scalar_conditions
    total = jnp.zeros((), dtype=potential.dtype)
    for axis in range(3):
        if scalar[axis].is_periodic or not float(problem.wall_conductance[axis]):
            continue
        area = jnp.asarray(grid.face_areas(axis), dtype=potential.dtype)
        for position, sign in ((0, -1.0), (-1, 1.0)):
            selection = (slice(None),) * axis + (position,)
            total = total + sign * jnp.sum(
                potential.data[selection] * currents[axis].data[selection] * area[selection]
            )
    return total


def advance(
    problem: ChannelProblem,
    steps: int,
    velocity: tuple[Field, Field, Field] | None = None,
    *,
    factorization: FastDiagonalPoisson | None = None,
    viscous: tuple[FastDiagonalHelmholtz, ...] | None = None,
    checkpoint: bool = True,
) -> Trajectory:
    """Run ``steps`` projection steps as one compiled scan.

    ``steps`` is static: it fixes the length of the compiled trajectory. The
    factorizations are built once on the host, outside the trace, because their
    assembly reads concrete arrays; passing ``viscous`` takes diffusion
    implicitly, which is what lets a long run choose its step for accuracy.
    """
    if steps < 1:
        raise ValueError("steps must be positive")
    factorization = problem.factorization() if factorization is None else factorization
    velocity = zero_velocity(problem) if velocity is None else velocity

    def single(state, _):
        updated, pressure, potential = step(state, problem, factorization, viscous)
        return updated, (*trajectory_diagnostics(updated, problem), pressure, potential)

    body = jax.checkpoint(single) if checkpoint else single
    final, (residual, energy, pressures, potentials) = jax.lax.scan(body, velocity, xs=None, length=steps)
    last = jax.tree_util.tree_map(lambda leaf: leaf[-1], (pressures, potentials))
    return Trajectory(final, last[0], last[1], residual, energy)
