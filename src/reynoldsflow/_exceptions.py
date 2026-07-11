"""Exceptions raised by ReynoldsFlow public and internal APIs."""


class ReynoldsFlowError(Exception):
    """Base class for ReynoldsFlow-specific errors."""


class InvalidGapError(ReynoldsFlowError, ValueError):
    """The supplied gap field cannot define the requested problem."""


class UnknownSolverError(ReynoldsFlowError, ValueError):
    """A solver specification is not recognized."""


class SolverUnavailableError(ReynoldsFlowError, ImportError):
    """A recognized solver is unavailable in the current environment."""


class ConvergenceError(ReynoldsFlowError, RuntimeError):
    """An iterative or direct backend failed to produce a valid solution."""
