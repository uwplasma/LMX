"""Fully developed pipe flow: the other geometry of the validation ladder."""

import jax.numpy as jnp
import numpy as np
import pytest

from lmx.grid import Grid, uniform_faces
from lmx.ops import divergence
from lmx.pipe import (
    PipeProblem,
    _face_currents,
    _potential,
    flow_rate,
    pipe_grid,
    pipe_problem,
    solve_pipe,
)

pytestmark = pytest.mark.numerical

# Hagen-Poiseuille in a unit pipe with unit forcing and unit viscosity: the
# profile is (1 - r^2)/4 and its mean over the section is exactly 1/8.
POISEUILLE = 0.125


def test_the_pipe_reproduces_poiseuille_without_a_field():
    """The one flow rate that is known exactly, and it fixes every normalisation at once."""
    errors = []
    for radial in (16, 32):
        velocity, _ = solve_pipe(pipe_problem(hartmann=0.0, radial=radial, azimuthal=16))
        errors.append(abs(flow_rate(velocity) - POISEUILLE) / POISEUILLE)
    assert errors[1] < 1.5e-3
    assert np.log2(errors[0] / errors[1]) > 1.9


def test_the_azimuthal_discretization_is_second_order():
    """The azimuth is uniform and periodic, so it should be clean second order."""
    rates = [
        flow_rate(solve_pipe(pipe_problem(hartmann=20.0, radial=32, azimuthal=count))[0])
        for count in (16, 32, 64)
    ]
    differences = [abs(rates[1] - rates[0]), abs(rates[2] - rates[1])]
    assert np.log2(differences[0] / differences[1]) > 1.8, differences


def test_the_field_slows_the_pipe_and_the_mesh_agrees_with_itself():
    coarse = flow_rate(solve_pipe(pipe_problem(hartmann=20.0, radial=32, azimuthal=64))[0])
    fine = flow_rate(solve_pipe(pipe_problem(hartmann=20.0, radial=48, azimuthal=64))[0])
    assert fine < 0.4 * POISEUILLE
    assert abs(coarse - fine) / fine < 0.01


def test_a_conducting_wall_short_circuits_the_pipe():
    """More wall conductance is more current returned through the wall, and less flow."""
    rates = [
        flow_rate(solve_pipe(pipe_problem(hartmann=20.0, radial=32, azimuthal=32, wall_conductance=c))[0])
        for c in (0.0, 0.027, 0.1)
    ]
    assert rates[0] > rates[1] > rates[2]


def test_the_face_currents_conserve_charge():
    """The potential equation is the divergence of the same currents the force uses."""
    problem = pipe_problem(hartmann=20.0, radial=24, azimuthal=32)
    factorization = problem.factorization()
    velocity, potential = solve_pipe(problem)
    currents = _face_currents(velocity, potential, problem)
    residual = float(jnp.max(jnp.abs(divergence(currents).data)))
    assert residual < 1e-8 * float(jnp.max(jnp.abs(velocity.data)))
    # The potential the solve reports is the one the residual was built on.
    assert float(jnp.max(jnp.abs(_potential(velocity, problem, factorization).data - potential.data))) < 1e-12


def test_the_pipe_states_what_it_needs():
    cartesian = Grid(uniform_faces(4, 0.0, 1.0), uniform_faces(4, 0.0, 1.0), uniform_faces(4, 0.0, 1.0))
    with pytest.raises(ValueError, match="a pipe needs a polar grid"):
        PipeProblem(cartesian, 1.0)
    grid = pipe_grid(8, 8, 0.0)
    with pytest.raises(ValueError, match="hartmann must not be negative"):
        PipeProblem(grid, -1.0)
    with pytest.raises(ValueError, match="wall conductance must not be negative"):
        PipeProblem(grid, 1.0, -0.5)


def test_the_mesh_resolves_the_layer_the_field_implies():
    coarse = np.diff(pipe_grid(32, 32, 0.0).x_faces)
    layered = np.diff(pipe_grid(32, 32, 200.0).x_faces)
    assert np.allclose(coarse, coarse[0])
    assert layered.min() < 1.0 / 200.0
    assert layered[-1] < layered[0]
