import jax.numpy as jnp
import pytest

from lmx.cases import make_hartmann_case, make_shercliff_case
from lmx.solvers import solve_steady


pytestmark = pytest.mark.regression


_EXPECTED_HARTMANN_CENTERLINE = jnp.asarray(
    [
        0.0,
        0.03861319,
        0.05785941,
        0.06664725,
        0.07029065,
        0.07155155,
        0.07155155,
        0.07029065,
        0.06664725,
        0.05785941,
        0.03861317,
        0.0,
    ]
)

_EXPECTED_SHERCLIFF_CENTERLINE = jnp.asarray(
    [
        0.0,
        0.03756857,
        0.05726324,
        0.06647202,
        0.07027150,
        0.07155155,
        0.07155155,
        0.07027150,
        0.06647202,
        0.05726324,
        0.03756856,
        0.0,
    ]
)


def _centerline(solution):
    return solution.state.u[:, solution.state.u.shape[1] // 2]


def test_hartmann_centerline_regression():
    case = make_hartmann_case(ha=2.0, ny=12, nz=12)
    solution = solve_steady(case)
    assert jnp.allclose(_centerline(solution), _EXPECTED_HARTMANN_CENTERLINE, atol=1e-6)


def test_shercliff_centerline_regression():
    case = make_shercliff_case(ha=2.0, ny=12, nz=12)
    solution = solve_steady(case)
    assert jnp.allclose(_centerline(solution), _EXPECTED_SHERCLIFF_CENTERLINE, atol=1e-6)
