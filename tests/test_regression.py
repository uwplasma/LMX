import jax.numpy as jnp
import pytest

from lmx.cases import make_hartmann_case, make_shercliff_case
from lmx.solvers import solve_steady


pytestmark = pytest.mark.regression


_EXPECTED_HARTMANN_CENTERLINE = jnp.asarray(
    [
        0.0,
        0.025192320,
        0.034829620,
        0.037971869,
        0.038864099,
        0.039082918,
        0.039082803,
        0.038863778,
        0.037971348,
        0.034829021,
        0.025191780,
        0.0,
    ]
)

_EXPECTED_SHERCLIFF_CENTERLINE = jnp.asarray(
    [
        0.0,
        0.024858040,
        0.034737770,
        0.037972290,
        0.038826790,
        0.038996019,
        0.039104210,
        0.039022319,
        0.038117018,
        0.034828179,
        0.024902860,
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
