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
