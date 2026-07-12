"""Efficient finite-volume Reynolds flow solvers."""

from ._exceptions import (
    ConvergenceError,
    InvalidGapError,
    ReynoldsFlowError,
    SolverUnavailableError,
    UnknownSolverError,
)


__version__ = "0.1.2"

__all__ = [
    "ConvergenceError",
    "InvalidGapError",
    "ReynoldsFlowError",
    "SolverUnavailableError",
    "UnknownSolverError",
    "__version__",
]
