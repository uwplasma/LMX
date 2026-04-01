from types import SimpleNamespace

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


def test_solve_poisson_lineax_falls_back_without_lineax(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(linear, "lx", None)
    diagonal, west, east, south, north, rhs = _poisson_coefficients()
    solution, info = linear.solve_poisson_lineax(diagonal, west, east, south, north, rhs, anchor=(0, 0))

    assert solution.shape == (2, 2)
    assert info.backend == "jax-jacobi"
    assert info.iterations == 400
    assert info.residual == pytest.approx(0.0)


def test_solve_poisson_lineax_uses_lineax_backend(monkeypatch: pytest.MonkeyPatch):
    diagonal, west, east, south, north, rhs = _poisson_coefficients()

    class FakeLinearOperator:
        def __init__(self, mv, shape, tags=None):
            self.mv = mv
            self.shape = shape
            self.tags = tags

    class FakeSolver:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def fake_linear_solve(op, rhs_vec, solver=None):
        op.mv(jnp.ones_like(rhs_vec))
        return SimpleNamespace(value=jnp.zeros_like(rhs_vec), stats={"num_steps": 7})

    fake_lx = SimpleNamespace(
        FunctionLinearOperator=FakeLinearOperator,
        positive_semidefinite_tag=object(),
        CG=FakeSolver,
        linear_solve=fake_linear_solve,
    )
    monkeypatch.setattr(linear, "lx", fake_lx)

    solution, info = linear.solve_poisson_lineax(diagonal, west, east, south, north, rhs, anchor=(0, 0))

    assert solution.shape == (2, 2)
    assert info.backend == "lineax-cg"
    assert info.iterations == 7


def test_solve_poisson_jacobi_can_stop_early_on_residual_tolerance():
    diagonal, west, east, south, north, rhs = _poisson_coefficients()
    solution, info = linear.solve_poisson_jacobi(
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
    assert info.backend == "jax-jacobi"
    assert info.iterations < 50
    assert info.residual <= 2e-6


def test_solve_poisson_cg_converges_on_zero_rhs():
    diagonal, west, east, south, north, rhs = _poisson_coefficients()

    solution, info = linear.solve_poisson_cg(
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
    assert info.backend == "jax-cg"
    assert info.iterations == 0
    assert info.residual == pytest.approx(0.0)


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
