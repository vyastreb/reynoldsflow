"""Shared sparse linear-solver registry and convergence diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import logging
import warnings

import numpy as np
from scipy.sparse.linalg import LinearOperator, MatrixRankWarning, cg, spsolve

from ._exceptions import (
    ConvergenceError,
    SolverUnavailableError,
    UnknownSolverError,
)


logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

DEFAULT_RTOL = 1e-12


@dataclass(frozen=True)
class LinearSolveResult:
    solution: np.ndarray
    solver: str
    converged: bool
    iterations: int | None
    relative_residual: float
    convergence_reason: str


_ALIASES = {
    "none": "auto",
    "petsc": "petsc-cg.hypre",
    "scipy.amg.rs": "scipy.amg-rs",
    "scipy.amg.sa": "scipy.amg-smooth_aggregation",
    "scipy.amg.smooth_aggregation": "scipy.amg-smooth_aggregation",
    "scipy.amg-sa": "scipy.amg-smooth_aggregation",
    "petsc-cg.ilu": "petsc-gmres.ilu",
    "cholesky": "cholesky",
}

_EXPLICIT_SOLVERS = {
    "scipy-spsolve",
    "scipy.amg-rs",
    "scipy.amg-smooth_aggregation",
    "pardiso",
    "cholesky",
    "petsc-cg.hypre",
    "petsc-cg.gamg",
    "petsc-gmres.ilu",
    "petsc-mumps",
}


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def normalize_solver_name(solver: str) -> str:
    if not isinstance(solver, str) or not solver.strip():
        raise UnknownSolverError("Solver name must be a non-empty string.")
    normalized = solver.strip().lower()
    normalized = _ALIASES.get(normalized, normalized)
    if normalized == "scipy":
        normalized = "scipy.amg-rs"
    elif normalized == "petsc-cg":
        normalized = "petsc-cg.hypre"

    if normalized != "auto" and normalized not in _EXPLICIT_SOLVERS:
        choices = ", ".join(sorted(_EXPLICIT_SOLVERS | {"auto"}))
        raise UnknownSolverError(
            f"Unknown solver {solver!r}. Supported solver names: {choices}."
        )
    return normalized


def automatic_solver_candidates() -> list[str]:
    """Return the portable base-dependency automatic solver.

    Optional native stacks are intentionally explicit-only: importing a locally
    installed but unusable MPI/PETSc or MKL/Pardiso stack can terminate the
    process before Python can catch an exception and attempt a fallback.
    """
    return ["scipy.amg-rs"]


def build_amg_preconditioner(matrix, method: str):
    if not _module_available("pyamg"):
        raise SolverUnavailableError(
            "PyAMG is required for SciPy AMG solvers; install the base package dependencies."
        )
    import pyamg

    max_coarse = max(10, matrix.shape[0] // 1000)
    try:
        if method == "amg-smooth_aggregation":
            hierarchy = pyamg.smoothed_aggregation_solver(
                matrix, max_coarse=max_coarse
            )
            return hierarchy.aspreconditioner(cycle="V")
        if method == "amg-rs":
            hierarchy = pyamg.ruge_stuben_solver(
                matrix, max_coarse=max_coarse, CF="RS"
            )
            return LinearOperator(
                matrix.shape,
                matvec=lambda vector: hierarchy.solve(
                    vector, tol=1e-2, maxiter=1
                ),
            )
    except Exception as exc:
        raise ConvergenceError(
            f"Failed to construct {method} preconditioner: {exc}"
        ) from exc
    raise UnknownSolverError(f"Unknown AMG preconditioner {method!r}.")


def _relative_residual(matrix, solution: np.ndarray, rhs: np.ndarray) -> float:
    residual = np.linalg.norm(matrix @ solution - rhs)
    scale = max(np.linalg.norm(rhs), np.finfo(np.float64).tiny)
    return float(residual / scale)


def _checked_result(
    matrix,
    rhs: np.ndarray,
    solution: np.ndarray,
    solver: str,
    iterations: int | None,
    reason: str,
) -> LinearSolveResult:
    solution = np.asarray(solution, dtype=np.float64).reshape(-1).copy()
    if solution.size != rhs.size or not np.all(np.isfinite(solution)):
        raise ConvergenceError(
            f"Solver {solver!r} returned an invalid or non-finite solution."
        )
    relative_residual = _relative_residual(matrix, solution, rhs)
    if not np.isfinite(relative_residual):
        raise ConvergenceError(f"Solver {solver!r} produced a non-finite residual.")
    return LinearSolveResult(
        solution=solution,
        solver=solver,
        converged=True,
        iterations=iterations,
        relative_residual=relative_residual,
        convergence_reason=reason,
    )


def _solve_once(
    matrix,
    rhs: np.ndarray,
    solver: str,
    rtol: float,
    external_preconditioner=None,
):
    if solver == "scipy-spsolve":
        matrix_csc = matrix.tocsc()
        with warnings.catch_warnings():
            warnings.simplefilter("error", MatrixRankWarning)
            try:
                solution = spsolve(matrix_csc, rhs)
            except Exception as exc:
                raise ConvergenceError(f"SciPy spsolve failed: {exc}") from exc
        return _checked_result(
            matrix_csc, rhs, solution, solver, None, "direct solve"
        )

    if solver.startswith("scipy."):
        method = solver.split(".", maxsplit=1)[1]
        matrix_csr = matrix.tocsr()
        preconditioner = external_preconditioner
        if preconditioner is None:
            preconditioner = build_amg_preconditioner(matrix_csr, method)
        iterations = 0

        def count_iteration(_):
            nonlocal iterations
            iterations += 1

        solution, info = cg(
            matrix_csr,
            rhs,
            M=preconditioner,
            rtol=rtol,
            atol=0.0,
            maxiter=6000,
            callback=count_iteration,
        )
        if info != 0:
            if info > 0:
                reason = f"iteration limit reached after {info} iterations"
            else:
                reason = f"illegal input or numerical breakdown ({info})"
            raise ConvergenceError(f"Solver {solver!r} did not converge: {reason}.")
        return _checked_result(
            matrix_csr, rhs, solution, solver, iterations, "converged"
        )

    if solver == "cholesky":
        if not _module_available("sksparse"):
            raise SolverUnavailableError(
                "CHOLMOD requires scikit-sparse; install reynoldsflow[solvers]."
            )
        from sksparse.cholmod import cholesky

        matrix_csc = matrix.tocsc()
        try:
            solution = cholesky(matrix_csc).solve_A(rhs)
        except Exception as exc:
            raise ConvergenceError(f"CHOLMOD failed: {exc}") from exc
        return _checked_result(
            matrix_csc, rhs, solution, solver, None, "direct solve"
        )

    if solver == "pardiso":
        if not _module_available("pypardiso"):
            raise SolverUnavailableError(
                "Pardiso requires pypardiso; install reynoldsflow[solvers]."
            )
        import pypardiso

        matrix_csr = matrix.tocsr()
        pardiso = pypardiso.PyPardisoSolver()
        pardiso.set_iparm(1, 1)
        pardiso.set_iparm(24, 1)
        pardiso.set_matrix_type(2)  # real symmetric positive definite
        try:
            solution = pardiso.solve(matrix_csr, rhs)
        except Exception as exc:
            raise ConvergenceError(f"Pardiso failed: {exc}") from exc
        return _checked_result(
            matrix_csr, rhs, solution, solver, None, "direct solve"
        )

    if solver.startswith("petsc-cg.") or solver.startswith("petsc-gmres."):
        if not _module_available("petsc4py"):
            raise SolverUnavailableError(
                "PETSc requires petsc4py; install reynoldsflow[solvers]."
            )
        from petsc4py import PETSc

        ksp_type = "gmres" if solver.startswith("petsc-gmres.") else "cg"
        preconditioner_name = solver.rsplit(".", maxsplit=1)[1]
        matrix_csr = matrix.tocsr()
        try:
            petsc_matrix = PETSc.Mat().createAIJ(
                size=matrix_csr.shape,
                csr=(matrix_csr.indptr, matrix_csr.indices, matrix_csr.data),
            )
            ksp = PETSc.KSP().create()
            ksp.setOperators(petsc_matrix)
            ksp.setType(ksp_type)
            if ksp_type == "gmres":
                try:
                    ksp.setGMRESRestart(200)
                except AttributeError:
                    pass
            pc = ksp.getPC()
            pc.setType(preconditioner_name)
            if preconditioner_name == "hypre":
                try:
                    pc.setHYPREType("boomeramg")
                except Exception:
                    pass
            ksp.setTolerances(rtol=rtol)
            ksp.setFromOptions()
            rhs_vector = PETSc.Vec().createWithArray(rhs)
            solution_vector = rhs_vector.duplicate()
            ksp.solve(rhs_vector, solution_vector)
            reason = int(ksp.getConvergedReason())
            iterations = int(ksp.getIterationNumber())
            solution = solution_vector.getArray().copy()
        except Exception as exc:
            raise SolverUnavailableError(
                f"PETSc solver {solver!r} could not be configured or run: {exc}"
            ) from exc
        if reason <= 0:
            raise ConvergenceError(
                f"PETSc solver {solver!r} did not converge (reason={reason})."
            )
        return _checked_result(
            matrix_csr,
            rhs,
            solution,
            solver,
            iterations,
            f"PETSc reason {reason}",
        )

    if solver == "petsc-mumps":
        if not _module_available("petsc4py"):
            raise SolverUnavailableError(
                "PETSc/MUMPS requires petsc4py; install reynoldsflow[solvers]."
            )
        from petsc4py import PETSc

        matrix_csr = matrix.tocsr()
        try:
            petsc_matrix = PETSc.Mat().createAIJ(
                size=matrix_csr.shape,
                csr=(matrix_csr.indptr, matrix_csr.indices, matrix_csr.data),
            )
            rhs_vector = PETSc.Vec().createWithArray(rhs)
            solution_vector = rhs_vector.duplicate()
            ksp = PETSc.KSP().create()
            ksp.setOperators(petsc_matrix)
            ksp.setType("preonly")
            pc = ksp.getPC()
            pc.setType("lu")
            pc.setFactorSolverType("mumps")
            ksp.setFromOptions()
            ksp.solve(rhs_vector, solution_vector)
            reason = int(ksp.getConvergedReason())
            solution = solution_vector.getArray().copy()
        except Exception as exc:
            raise SolverUnavailableError(
                f"PETSc/MUMPS could not be configured or run: {exc}"
            ) from exc
        if reason <= 0:
            raise ConvergenceError(
                f"PETSc/MUMPS did not converge (reason={reason})."
            )
        return _checked_result(
            matrix_csr,
            rhs,
            solution,
            solver,
            None,
            f"PETSc reason {reason}",
        )

    raise UnknownSolverError(f"Solver {solver!r} has no registered implementation.")


def solve_linear_system(
    matrix,
    rhs: np.ndarray,
    solver: str = "auto",
    rtol: float = DEFAULT_RTOL,
    preconditioner=None,
) -> LinearSolveResult:
    """Solve a symmetric sparse system with explicit diagnostics."""
    normalized = normalize_solver_name(solver)
    if rtol <= 0.0 or not np.isfinite(rtol):
        raise ValueError("rtol must be finite and strictly positive.")

    def require_true_residual(result: LinearSolveResult) -> LinearSolveResult:
        if result.iterations is not None:
            limit = max(10.0 * rtol, 100.0 * np.finfo(np.float64).eps)
            if result.relative_residual > limit:
                raise ConvergenceError(
                    f"Solver {result.solver!r} reported convergence but its "
                    f"true relative residual {result.relative_residual:.3e} "
                    f"exceeds acceptance limit {limit:.3e}."
                )
        return result

    if normalized != "auto":
        if preconditioner is not None and not normalized.startswith("scipy."):
            raise ValueError(
                "An external preconditioner is supported only by SciPy iterative solvers."
            )
        return require_true_residual(
            _solve_once(
                matrix,
                rhs,
                normalized,
                rtol,
                external_preconditioner=preconditioner,
            )
        )

    failures = []
    for candidate in automatic_solver_candidates():
        try:
            result = require_true_residual(
                _solve_once(
                    matrix,
                    rhs,
                    candidate,
                    rtol,
                    external_preconditioner=preconditioner,
                )
            )
            logger.info("Auto-selected linear solver %s.", result.solver)
            return result
        except (SolverUnavailableError, ConvergenceError) as exc:
            failures.append(f"{candidate}: {exc}")
            logger.warning("Auto solver candidate %s failed: %s", candidate, exc)
    raise ConvergenceError(
        "No automatic solver candidate succeeded. " + " | ".join(failures)
    )
