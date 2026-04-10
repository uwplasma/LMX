from io import StringIO

import pytest

from lmx.cases import make_hartmann_case
from lmx.config import LoggingSpec
from lmx.runtime_logging import StreamingSolverLogger
from lmx.solvers import solve_steady


pytestmark = pytest.mark.unit


def test_streaming_solver_logger_prints_live_solver_sections():
    stream = StringIO()
    logger = StreamingSolverLogger(LoggingSpec(step_stride=1), stream=stream)
    case = make_hartmann_case(ha=5.0, ny=8, nz=8)

    solve_steady(case, logger=logger)

    text = stream.getvalue()
    assert "LMX Solver Run" in text
    assert "Create mesh for case" in text
    assert "Time =" in text
    assert "smoothSolver: potE" in text
    assert "smoothSolver: U" in text
    assert "currentScaledPressureProxy" in text
    assert "MHD integrals" in text
    assert "MHD conservation" in text
    assert "rawUpdateMax" in text
    assert "limitedFraction" in text
    assert "steadySolver" in text
    assert "ExecutionTime" in text
