"""One projection step for inductionless duct flow on the staggered grid.

The step is the classical fractional one, assembled from the pieces earlier plan
steps settled: the potential is solved first, the face currents of :mod:`lmx.em`
supply the Lorentz force, the momentum is advanced, and a pressure Poisson solve
projects the velocity back onto the discretely divergence-free space.

Two details are specific to magnetohydrodynamics and matter more than the rest of
the step put together.

*Stiffness from the field.* The velocity part of the Lorentz force is a damping,
:math:`-\\sigma(B^2 I - \\mathbf B\\mathbf B^{\\mathsf T})\\mathbf u`, with rate
:math:`\\sigma B^2/\\rho`. Treating it explicitly forces
:math:`\\Delta t \\propto Ha^{-2}`, which is why explicit codes stall at
blanket-scale Hartmann numbers. It is applied here as a correction to the
conservative face force, so the update reads
``u* = u + dt * rhs / (1 + dt * lambda)``. The correction vanishes when the
right-hand side does, so it changes the path to a steady state but not the steady
state itself, and the conservative force keeps its own discretization.

*Consistency.* The force comes from the same face currents the potential equation
is built on, never from a separately differenced potential. In the core the
balance is :math:`-\\nabla p + \\mathbf J\\times\\mathbf B = 0` to
:math:`O(Ha^{-2})`, so any inconsistency between the two is amplified by
:math:`Ha^2`. The insulating wall is part of that consistency: the motional term
is dropped on wall faces by :func:`lmx.em.wall_insulated` before its divergence
is taken, because the operator that receives it has no wall flux either. Leaving
it in makes the potential absorb a boundary current the wall cannot carry, and a
square duct at :math:`Ha=20` then runs at less than half its correct flow rate.

Two constraints must hold before the divergence of a face velocity means
anything, and both are imposed rather than assumed. A wall-normal component is
set to zero on its wall faces, and on a periodic axis the duplicated first and
last face are made equal. Without either the discrete divergence carries a net
flux through the boundary, and with every axis periodic or Neumann the pressure
has no way to remove that constant: the projection would return a field that is
still not divergence free.

*Stiffness from viscosity.* Diffusion may be taken either way. Left explicit it
bounds the step by :attr:`ChannelProblem.diffusive_step_limit`, reported rather
than enforced so a caller sweeping a parameter sees the constraint instead of a
silently clipped step. Passing the factorizations from
:meth:`ChannelProblem.viscous_factorizations` to :func:`step` solves

.. math:: \\left[(1+\\Delta t\\,\\lambda)I-\\Delta t\\,\\nu\\nabla^2\\right]\\mathbf u^{*}
   =(1+\\Delta t\\,\\lambda)\\mathbf u+\\Delta t\\,(\\mathbf F+\\mathbf f)/\\rho,

backward Euler on the viscous term with the damping correction folded into the
shift. That operator separates exactly as the pressure Laplacian does, so it
costs three contractions and a divide, and the step is then bounded by accuracy
rather than by the mesh. Both stiffnesses are gone at that point: the magnetic
one because it grows as :math:`Ha^2`, the viscous one because it grows as the
mesh is refined.

*Convective transport.* ``advection`` selects it: ``"off"`` is the Stokes limit,
appropriate at the large interaction parameters of a blanket channel and the
default so that no run acquires a convective step limit by accident;
``"central"`` is the conservative flux form of :mod:`lmx.advect`, second order on
a stretched mesh; ``"limited"`` adds the van Leer blend that keeps the thin side
layers bounded. Transport is explicit, so switching it on bounds the step by
:func:`lmx.advect.advective_step_limit`.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import solvax

from .advect import momentum_advection
from .bc import DIRICHLET, BoundaryCondition
from .em import (
    face_conductivity,
    face_current,
    face_electromotive_force,
    lorentz_force,
    thin_wall_flux,
    wall_insulated,
)
from .grid import CENTER, FACE, Field, Grid
from .ops import divergence, face_gradient, face_interpolate, staggered_laplacian
from .poisson import (
    FastDiagonalHelmholtz,
    FastDiagonalPoisson,
    fast_diagonal_helmholtz,
    fast_diagonal_poisson,
)

__all__ = [
    "ChannelProblem",
    "electric_state",
    "enforce_face_constraints",
    "project",
    "step",
    "velocity_condition",
    "velocity_offset",
    "zero_velocity",
]

_NO_SLIP = BoundaryCondition(DIRICHLET)
_INSULATING = BoundaryCondition("neumann")
_ADVECTION = ("off", "central", "limited")


def velocity_offset(component: int) -> tuple[float, float, float]:
    """Return the staggered position of one velocity component."""
    return tuple(FACE if axis == component else CENTER for axis in range(3))


def velocity_condition(
    conditions: tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition], axis: int
) -> BoundaryCondition:
    """Return the condition a velocity component sees along ``axis``.

    A periodic axis stays periodic. At a wall every component is homogeneous:
    the tangential ones by no slip, the normal one because it sits on the wall
    and is prescribed there by :func:`enforce_face_constraints`.
    """
    return conditions[axis] if conditions[axis].is_periodic else _NO_SLIP


@dataclass(frozen=True)
class ChannelProblem:
    """A duct segment with uniform material properties and a uniform imposed field."""

    grid: Grid
    conditions: tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition]
    density: float = 1.0
    viscosity: float = 1.0
    conductivity: float = 1.0
    magnetic_field: tuple[float, float, float] = (0.0, 0.0, 0.0)
    forcing: tuple[float, float, float] = (0.0, 0.0, 0.0)
    dt: float = 1.0e-3
    advection: str = "off"
    wall_conductance: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __post_init__(self) -> None:
        if len(self.conditions) != 3:
            raise ValueError("a channel needs one boundary condition per axis")
        for name in ("density", "viscosity", "dt"):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive")
        if float(self.conductivity) < 0.0:
            raise ValueError("conductivity must not be negative")
        if self.advection not in _ADVECTION:
            raise ValueError(f"advection must be one of {sorted(_ADVECTION)}, got {self.advection!r}")
        if len(self.wall_conductance) != 3:
            raise ValueError("a channel needs one wall conductance per axis")
        if any(float(value) < 0.0 for value in self.wall_conductance):
            raise ValueError("wall conductance must not be negative")

    @property
    def conducting_walls(self) -> bool:
        """Whether any wall carries current along itself."""
        return any(float(value) > 0.0 for value in self.wall_conductance)

    @property
    def scalar_conditions(self) -> tuple[BoundaryCondition, BoundaryCondition, BoundaryCondition]:
        """Conditions for pressure and potential: periodic, else a homogeneous Neumann wall."""
        return tuple(condition if condition.is_periodic else _INSULATING for condition in self.conditions)

    @property
    def damping_rates(self) -> tuple[float, float, float]:
        """Return the implicit damping rate of each velocity component.

        The diagonal of :math:`\\sigma(B^2 I - \\mathbf B\\mathbf B^{\\mathsf T})/\\rho`
        is used. Off-diagonal coupling stays in the explicit conservative force,
        which is exact for a field aligned with an axis and a documented
        approximation otherwise.
        """
        squared = float(np.dot(self.magnetic_field, self.magnetic_field))
        return tuple(
            float(self.conductivity) * (squared - float(component) ** 2) / float(self.density)
            for component in self.magnetic_field
        )

    @property
    def diffusive_step_limit(self) -> float:
        """Return the largest stable step for the explicit viscous term.

        The usual bound for an explicit centred Laplacian,
        ``dt <= 1 / (2 * nu * sum(1 / dx**2))``, evaluated on the finest cell of
        each axis. The magnetic damping imposes no such limit here because it is
        treated implicitly, so this is normally the binding constraint.
        """
        inverse = sum(1.0 / float(np.min(widths)) ** 2 for widths in self.grid.widths)
        return 1.0 / (2.0 * float(self.viscosity) * inverse)

    def viscous_factorizations(self) -> tuple[FastDiagonalHelmholtz, ...]:
        """Factorize the implicit viscous operator for each velocity component.

        Build these once on the host and pass them to :func:`step` to take
        diffusion implicitly. The shift carries the magnetic damping, so one
        solve removes both stiff terms.
        """
        conditions = tuple(velocity_condition(self.conditions, axis) for axis in range(3))
        return tuple(
            fast_diagonal_helmholtz(
                self.grid,
                velocity_offset(component),
                conditions,
                shift=1.0 + float(self.dt) * self.damping_rates[component],
                coefficient=float(self.dt) * float(self.viscosity),
            )
            for component in range(3)
        )

    def factorization(self) -> FastDiagonalPoisson:
        """Factorize the scalar Laplacian shared by the pressure and the potential.

        Both see the same homogeneous conditions, so one factorization serves
        both. Build it once outside a traced function and pass it to
        :func:`step`; the assembly reads concrete arrays and cannot be traced.
        """
        return fast_diagonal_poisson(self.grid, self.scalar_conditions)


def electric_state(
    velocity: tuple[Field, Field, Field],
    problem: ChannelProblem,
    factorization: FastDiagonalPoisson | None = None,
    field_scale=1.0,
) -> tuple[Field, tuple[Field, Field, Field]]:
    """Return the induced potential and the Lorentz force it carries.

    One place forms the face currents, so the projection step and the steady
    solve cannot drift apart in how they close the wall or scale the potential.
    ``field_scale`` multiplies the imposed field and may be traced.
    """
    factorization = problem.factorization() if factorization is None else factorization
    scalar = problem.scalar_conditions
    field = tuple(
        _constant(problem.grid, value).replace_data(field_scale * _constant(problem.grid, value).data)
        for value in problem.magnetic_field
    )
    conductivities = [
        face_conductivity(_constant(problem.grid, problem.conductivity), axis, scalar[axis])
        for axis in range(3)
    ]
    emfs = [face_electromotive_force(velocity, field, axis, scalar) for axis in range(3)]
    motional = tuple(
        wall_insulated(
            conductivities[axis].replace_data(conductivities[axis].data * emfs[axis].data),
            axis,
            scalar[axis],
        )
        for axis in range(3)
    )
    potential = _solve_potential(divergence(motional), problem, factorization)
    currents = tuple(
        _closed_current(potential, conductivities[axis], emfs[axis], axis, problem) for axis in range(3)
    )
    return potential, lorentz_force(currents, field, scalar)


def _closed_current(
    potential: Field, conductivity: Field, emf: Field, axis: int, problem: ChannelProblem
) -> Field:
    """Ohm's law inside, and whatever the wall itself conducts on the wall faces."""
    scalar = problem.scalar_conditions
    ohmic = wall_insulated(face_current(potential, conductivity, emf, axis, scalar[axis]), axis, scalar[axis])
    if not problem.conducting_walls:
        return ohmic
    wall = thin_wall_flux(potential, axis, scalar[axis], problem.wall_conductance[axis], scalar)
    return ohmic.replace_data(ohmic.data + wall.data)


def _charge_operator(potential: Field, problem: ChannelProblem) -> Field:
    """Return the charge balance of a potential alone, with no motional term."""
    scalar = problem.scalar_conditions
    zero = tuple(
        Field(
            jnp.zeros(problem.grid.face_shape(axis), dtype=potential.dtype), _face_offset(axis), problem.grid
        )
        for axis in range(3)
    )
    conductivities = [
        face_conductivity(_constant(problem.grid, problem.conductivity), axis, scalar[axis])
        for axis in range(3)
    ]
    currents = tuple(
        _closed_current(potential, conductivities[axis], zero[axis], axis, problem) for axis in range(3)
    )
    balance = divergence(currents)
    return balance.replace_data(-balance.data)


def _face_offset(axis: int) -> tuple[float, float, float]:
    return tuple(FACE if position == axis else CENTER for position in range(3))


def _solve_potential(source: Field, problem: ChannelProblem, factorization: FastDiagonalPoisson) -> Field:
    """Solve the charge equation for the potential.

    With insulating walls the operator is the scalar Laplacian and the exact
    factorization answers in three contractions. A conducting wall adds a
    tangential surface operator on the wall layer, which is not separable; the
    factorization then becomes the preconditioner of a Krylov solve, and the
    solve is wrapped in :func:`jax.lax.custom_linear_solve` so the adjoint runs
    on the transposed operator instead of through the iteration.
    """
    scale = 1.0 / float(problem.conductivity) if float(problem.conductivity) else 0.0

    def preconditioner(residual: Field) -> Field:
        return factorization.solve(residual.replace_data(scale * residual.data))

    if not problem.conducting_walls:
        return preconditioner(source)

    def operator(potential: Field) -> Field:
        return _charge_operator(potential, problem)

    def solve(matvec, target):
        return solvax.gmres(matvec, target, precond=preconditioner, rtol=1.0e-12, max_restarts=20).x

    return jax.lax.custom_linear_solve(operator, source, solve, solve)


def zero_velocity(problem: ChannelProblem) -> tuple[Field, Field, Field]:
    """Return a velocity at rest in the staggered layout."""
    return tuple(
        Field(
            jnp.zeros(problem.grid.offset_shape(velocity_offset(component)), dtype=jnp.result_type(float)),
            velocity_offset(component),
            problem.grid,
        )
        for component in range(3)
    )


def enforce_face_constraints(
    velocity: tuple[Field, Field, Field], problem: ChannelProblem
) -> tuple[Field, Field, Field]:
    """Impose impermeability at walls and face agreement across a periodic axis.

    Both are prerequisites for the discrete divergence to represent a flux
    balance; see the module docstring.
    """
    constrained = []
    for component, field in enumerate(velocity):
        data = field.data
        selection = (slice(None),) * component
        if problem.conditions[component].is_periodic:
            data = data.at[selection + (-1,)].set(data[selection + (0,)])
        else:
            data = data.at[selection + (0,)].set(0.0)
            data = data.at[selection + (-1,)].set(0.0)
        constrained.append(field.replace_data(data))
    return tuple(constrained)


def project(
    velocity: tuple[Field, Field, Field],
    problem: ChannelProblem,
    factorization: FastDiagonalPoisson | None = None,
) -> tuple[tuple[Field, Field, Field], Field]:
    """Remove the divergence from a velocity field and return the pressure that did it."""
    factorization = problem.factorization() if factorization is None else factorization
    velocity = enforce_face_constraints(velocity, problem)
    source = divergence(velocity)
    scale = problem.density / problem.dt
    pressure = factorization.solve(source.replace_data(scale * source.data))
    scalar = problem.scalar_conditions
    corrected = tuple(
        field.replace_data(
            field.data
            - (problem.dt / problem.density) * face_gradient(pressure, component, scalar[component]).data
        )
        for component, field in enumerate(velocity)
    )
    return enforce_face_constraints(corrected, problem), pressure


def step(
    velocity: tuple[Field, Field, Field],
    problem: ChannelProblem,
    factorization: FastDiagonalPoisson | None = None,
    viscous: tuple[FastDiagonalHelmholtz, ...] | None = None,
) -> tuple[tuple[Field, Field, Field], Field, Field]:
    """Advance one projection step and return velocity, pressure and potential.

    Passing ``viscous`` takes diffusion implicitly and lifts the step off
    :attr:`ChannelProblem.diffusive_step_limit`.
    """
    factorization = problem.factorization() if factorization is None else factorization
    scalar = problem.scalar_conditions
    potential, force = electric_state(velocity, problem, factorization)

    velocity_conditions = tuple(velocity_condition(problem.conditions, axis) for axis in range(3))
    transport = (
        None
        if problem.advection == "off"
        else momentum_advection(velocity, velocity_conditions, limited=problem.advection == "limited")
    )
    predicted = []
    for component, component_field in enumerate(velocity):
        body = face_interpolate(force[component], component, scalar[component])
        drive = (body.data + problem.forcing[component]) / problem.density
        if transport is not None:
            drive = drive - transport[component].data
        rate = problem.damping_rates[component]
        shift = 1.0 + problem.dt * rate
        if viscous is None:
            diffusion = staggered_laplacian(component_field, velocity_conditions)
            updated = component_field.data + problem.dt * (problem.viscosity * diffusion.data + drive) / shift
        else:
            source = component_field.replace_data(shift * component_field.data + problem.dt * drive)
            updated = viscous[component].solve(source).data
        predicted.append(component_field.replace_data(updated))

    corrected, pressure = project(tuple(predicted), problem, factorization)
    return corrected, pressure, potential


def _constant(grid: Grid, value: float) -> Field:
    with jax.ensure_compile_time_eval():
        data = jnp.full(grid.shape, value, dtype=jnp.result_type(float))
    return Field(data, (CENTER,) * 3, grid)
