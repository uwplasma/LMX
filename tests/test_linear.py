import jax
import jax.numpy as jnp
import pytest

import lmx.linear as linear


pytestmark = pytest.mark.unit


def _poisson_coefficients():
    diagonal = jnp.ones((2, 2)) * 4.0
    west = jnp.ones((2, 2))
    east = jnp.ones((2, 2))
    south = jnp.ones((2, 2))
    north = jnp.ones((2, 2))
    rhs = jnp.zeros((2, 2))
    return diagonal, west, east, south, north, rhs


def test_solve_poisson_jacobi_can_stop_early_on_residual_tolerance():
    diagonal, west, east, south, north, rhs = _poisson_coefficients()
    solution, residual, iterations = linear.solve_poisson_jacobi_state(
        diagonal,
        west,
        east,
        south,
        north,
        rhs,
        anchor=(0, 0),
        iterations=50,
        tolerance=1e-6,
    )

    assert solution.shape == (2, 2)
    assert int(iterations) < 50
    assert float(residual) <= 2e-6


def test_solve_poisson_cg_converges_on_zero_rhs():
    diagonal, west, east, south, north, rhs = _poisson_coefficients()

    solution, residual, iterations = linear.solve_poisson_cg_state(
        diagonal,
        west,
        east,
        south,
        north,
        rhs,
        anchor=(0, 0),
        iterations=20,
        tolerance=1e-8,
    )

    assert solution.shape == (2, 2)
    assert int(iterations) == 0
    assert float(residual) == pytest.approx(0.0)


def test_poisson_residual_norm_is_zero_for_exact_zero_solution():
    diagonal, west, east, south, north, rhs = _poisson_coefficients()
    residual = linear.poisson_residual_norm(
        diagonal,
        west,
        east,
        south,
        north,
        rhs,
        jnp.zeros_like(rhs),
        anchor=(0, 0),
    )

    assert float(residual) == pytest.approx(0.0)


def test_solvax_pcg_recovers_manufactured_five_point_solution():
    scale = 1.0e-8
    diagonal = scale * jnp.full((3, 3), 6.0)
    west = scale * jnp.ones((3, 3))
    east = scale * jnp.ones((3, 3))
    south = scale * jnp.ones((3, 3))
    north = scale * jnp.ones((3, 3))
    known = jnp.arange(1.0, 10.0).reshape(3, 3)
    rhs = linear.apply_five_point_operator(diagonal, west, east, south, north, known)
    tolerance = max(1.0e-11, 100.0 * jnp.finfo(rhs.dtype).eps)

    solvax, residual, _ = linear.solve_five_point_solvax_pcg_state(
        diagonal,
        west,
        east,
        south,
        north,
        rhs,
        iterations=40,
        tolerance=tolerance,
    )

    assert float(residual) <= tolerance
    assert jnp.allclose(solvax, known, rtol=tolerance, atol=tolerance)


def test_solvax_pcg_five_point_gradient_is_implicit_and_matches_exact_solution():
    diagonal = jnp.full((2, 2), 4.0)
    zeros = jnp.zeros((2, 2))
    rhs_base = jnp.arange(1.0, 5.0).reshape(2, 2)
    exact_base = rhs_base / diagonal

    def objective(scale):
        field, _, _ = linear.solve_five_point_solvax_pcg_state(
            diagonal,
            zeros,
            zeros,
            zeros,
            zeros,
            scale * rhs_base,
            iterations=8,
            tolerance=1.0e-12,
            preconditioner="jacobi",
        )
        return jnp.sum(field**2)

    scale = 1.3
    expected = 2.0 * scale * jnp.sum(exact_base**2)
    assert jnp.allclose(jax.grad(objective)(scale), expected, rtol=1.0e-10)
