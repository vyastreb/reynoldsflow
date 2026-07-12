"""Package-level public metadata and exception exports."""

import reynoldsflow
import pytest


pytestmark = pytest.mark.unit


def test_public_version_and_exceptions():
    assert reynoldsflow.__version__ == "0.1.1"
    assert issubclass(reynoldsflow.InvalidGapError, ValueError)
    assert issubclass(reynoldsflow.UnknownSolverError, ValueError)
    assert issubclass(reynoldsflow.SolverUnavailableError, ImportError)
    assert issubclass(reynoldsflow.ConvergenceError, RuntimeError)
