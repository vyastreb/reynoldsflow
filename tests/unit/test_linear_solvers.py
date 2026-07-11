"""Shared solver registry, fallback, and diagnostics tests."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse import csr_matrix

from reynoldsflow import _linear_solvers as solvers
from reynoldsflow._exceptions import (
    ConvergenceError,
    SolverUnavailableError,
    UnknownSolverError,
)
from reynoldsflow.transport import solve_fluid_problem


pytestmark = pytest.mark.unit


def _small_spd_system():
    matrix = csr_matrix(np.array([[4.0, -1.0], [-1.0, 3.0]]))
    rhs = np.array([1.0, 2.0])
    return matrix, rhs


def test_scipy_direct_solver_reports_residual():
    matrix, rhs = _small_spd_system()
    result = solvers.solve_linear_system(matrix, rhs, solver="scipy-spsolve")

    assert result.solver == "scipy-spsolve"
    assert result.converged
    assert result.iterations is None
    assert result.relative_residual < 1e-14
    assert_allclose(matrix @ result.solution, rhs, rtol=1e-14, atol=1e-14)


def test_scipy_amg_solver_reports_iterations_and_residual():
    matrix, rhs = _small_spd_system()
    result = solvers.solve_linear_system(
        matrix, rhs, solver="scipy.amg-rs", rtol=1e-12
    )

    assert result.converged
    assert result.iterations is not None
    assert result.iterations > 0
    assert result.relative_residual < 1e-11


def test_auto_falls_back_to_base_scipy(monkeypatch):
    matrix, rhs = _small_spd_system()
    monkeypatch.setattr(
        solvers,
        "_module_available",
        lambda name: name == "pyamg",
    )

    result = solvers.solve_linear_system(matrix, rhs, solver="auto")

    assert result.solver == "scipy.amg-rs"


def test_auto_never_implicitly_imports_petsc(monkeypatch):
    monkeypatch.setattr(solvers, "_module_available", lambda name: True)

    assert solvers.automatic_solver_candidates() == ["scipy.amg-rs"]


@pytest.mark.parametrize(
    ("alias", "canonical"),
    [
        ("scipy", "scipy.amg-rs"),
        ("scipy.amg.rs", "scipy.amg-rs"),
        ("scipy.amg-sa", "scipy.amg-smooth_aggregation"),
        ("petsc", "petsc-cg.hypre"),
        ("petsc-cg.ilu", "petsc-gmres.ilu"),
        ("none", "auto"),
    ],
)
def test_legacy_solver_aliases(alias, canonical):
    assert solvers.normalize_solver_name(alias) == canonical


def test_unknown_solver_is_explicit():
    with pytest.raises(UnknownSolverError, match="Unknown solver"):
        solvers.normalize_solver_name("definitely-not-a-solver")


def test_unavailable_explicit_solver_is_explicit(monkeypatch):
    matrix, rhs = _small_spd_system()
    monkeypatch.setattr(solvers, "_module_available", lambda name: False)

    with pytest.raises(SolverUnavailableError, match="pypardiso"):
        solvers.solve_linear_system(matrix, rhs, solver="pardiso")


def test_public_solver_does_not_swallow_unknown_solver_error():
    gaps = np.ones((6, 6), dtype=np.float64)

    with pytest.raises(UnknownSolverError):
        solve_fluid_problem(gaps, solver="definitely-not-a-solver")


def test_reported_convergence_requires_true_residual(monkeypatch):
    matrix, rhs = _small_spd_system()
    bad_result = solvers.LinearSolveResult(
        solution=np.zeros(2),
        solver="scipy.amg-rs",
        converged=True,
        iterations=1,
        relative_residual=1e-2,
        convergence_reason="mocked",
    )
    monkeypatch.setattr(solvers, "_solve_once", lambda *args, **kwargs: bad_result)

    with pytest.raises(ConvergenceError, match="true relative residual"):
        solvers.solve_linear_system(
            matrix, rhs, solver="scipy.amg-rs", rtol=1e-10
        )
