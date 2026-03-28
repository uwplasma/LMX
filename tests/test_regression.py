import jax.numpy as jnp
import pytest

from lmx.cases import make_hartmann_case, make_shercliff_case
from lmx.solvers import solve_steady


pytestmark = pytest.mark.regression


_EXPECTED_HARTMANN_CENTERLINE = jnp.asarray(
    [
        0.0,
        0.025262890,
        0.034946479,
        0.038111702,
        0.039012909,
        0.039234478,
        0.039234374,
        0.039012600,
        0.038111202,
        0.034945879,
        0.025262350,
        0.0,
    ]
)

_EXPECTED_SHERCLIFF_CENTERLINE = jnp.asarray(
    [
        0.0,
        0.024926130,
        0.034853220,
        0.038111661,
        0.038975261,
        0.039147109,
        0.039255910,
        0.039171938,
        0.038257360,
        0.034944300,
        0.024971290,
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
