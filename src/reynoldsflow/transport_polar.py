"""
Finite-Difference Reynolds Fluid Flow Solver in Polar Coordinates

Author: Vladislav A. Yastrebov (CNRS, Mines Paris - PSL)
AI: Cursor, Claude, ChatGPT (polar extension by ChatGPT)
Date: Nov 2025
License: BSD 3-Clause
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from numba import jit, njit
from scipy.sparse import csr_matrix, save_npz

from ._connectivity import find_spanning_mask
from ._active_dofs import build_active_mapping, reconstruct_full_solution
from ._linear_solvers import (
    DEFAULT_RTOL,
    build_amg_preconditioner,
    normalize_solver_name,
    solve_linear_system,
)
from ._exceptions import InvalidGapError
from ._validation import validate_gap_array
from ._sparse import indptr_from_counts

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_STREAM_HANDLER_MARKER = "_reynoldsflow_stream_handler"


def setup_logging(level=logging.INFO):
    """Configure one module stream handler, even when called repeatedly."""
    handler = next(
        (
            existing
            for existing in logger.handlers
            if getattr(existing, _STREAM_HANDLER_MARKER, False)
        ),
        None,
    )
    if handler is None:
        handler = logging.StreamHandler()
        setattr(handler, _STREAM_HANDLER_MARKER, True)
        formatter = logging.Formatter(
            '/FS: %(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)


def set_verbosity(level: str = 'info'):
    levels = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL,
    }
    logger.setLevel(levels.get(level.lower(), logging.INFO))


THETA_BC_PERIODIC = 0
THETA_BC_SYMMETRY = 1


@njit
def face_k(a: float, b: float) -> float:
    """Harmonic mean of face conductivity; zero if either side blocked."""
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return 2.0 * (a ** 3) * (b ** 3) / (a ** 3 + b ** 3)


@njit
def _count_full_matrix_entries_polar(
    n_r, n_theta, gaps, theta_bc_code
):
    counts = np.empty(n_r * n_theta, dtype=np.int32)
    for i in range(n_r):
        for j in range(n_theta):
            row = i * n_theta + j
            if gaps[i, j] <= 0.0 or i == 0 or i == n_r - 1:
                counts[row] = 1
                continue
            count = 1
            if i + 1 < n_r - 1 and gaps[i + 1, j] > 0.0:
                count += 1
            if i - 1 > 0 and gaps[i - 1, j] > 0.0:
                count += 1
            if n_theta > 1:
                if theta_bc_code == THETA_BC_PERIODIC:
                    if gaps[i, (j + 1) % n_theta] > 0.0:
                        count += 1
                    if gaps[i, (j - 1) % n_theta] > 0.0:
                        count += 1
                else:
                    if j + 1 < n_theta and gaps[i, j + 1] > 0.0:
                        count += 1
                    if j > 0 and gaps[i, j - 1] > 0.0:
                        count += 1
            counts[row] = count
    return counts


@njit
def _fill_full_matrix_csr_polar(
    n_r,
    n_theta,
    gaps,
    r_inner,
    dr,
    dtheta,
    p_inner,
    p_outer,
    theta_bc_code,
    indptr,
):
    indices = np.empty(int(indptr[-1]), dtype=np.int32)
    data = np.empty(int(indptr[-1]), dtype=np.float64)
    b = np.zeros(n_r * n_theta, dtype=np.float64)
    dtheta_sq = dtheta * dtheta if n_theta > 1 else 1.0

    for i in range(n_r):
        r_i = r_inner + i * dr
        for j in range(n_theta):
            row = i * n_theta + j
            cursor = int(indptr[row])
            if gaps[i, j] <= 0.0:
                indices[cursor] = row
                data[cursor] = 1.0
                continue
            if i == 0:
                indices[cursor] = row
                data[cursor] = 1.0
                b[row] = p_inner
                continue
            if i == n_r - 1:
                indices[cursor] = row
                data[cursor] = 1.0
                b[row] = p_outer
                continue

            diag = 0.0
            coefficient = face_k(gaps[i, j], gaps[i + 1, j])
            if coefficient > 0.0:
                coefficient *= (
                    r_inner + (i + 0.5) * dr
                ) / (dr * dr)
                diag += coefficient
                if i + 1 == n_r - 1:
                    b[row] += coefficient * p_outer
                else:
                    indices[cursor] = (i + 1) * n_theta + j
                    data[cursor] = -coefficient
                    cursor += 1

            coefficient = face_k(gaps[i, j], gaps[i - 1, j])
            if coefficient > 0.0:
                coefficient *= (
                    r_inner + (i - 0.5) * dr
                ) / (dr * dr)
                diag += coefficient
                if i - 1 == 0:
                    b[row] += coefficient * p_inner
                else:
                    indices[cursor] = (i - 1) * n_theta + j
                    data[cursor] = -coefficient
                    cursor += 1

            if n_theta > 1 and dtheta > 0.0:
                if theta_bc_code == THETA_BC_PERIODIC:
                    j_plus = (j + 1) % n_theta
                    j_minus = (j - 1) % n_theta
                    coefficient = face_k(gaps[i, j], gaps[i, j_plus])
                    if coefficient > 0.0:
                        coefficient /= r_i * dtheta_sq
                        indices[cursor] = i * n_theta + j_plus
                        data[cursor] = -coefficient
                        cursor += 1
                        diag += coefficient
                    coefficient = face_k(gaps[i, j], gaps[i, j_minus])
                    if coefficient > 0.0:
                        coefficient /= r_i * dtheta_sq
                        indices[cursor] = i * n_theta + j_minus
                        data[cursor] = -coefficient
                        cursor += 1
                        diag += coefficient
                else:
                    if j + 1 < n_theta:
                        coefficient = face_k(gaps[i, j], gaps[i, j + 1])
                        if coefficient > 0.0:
                            coefficient /= r_i * dtheta_sq
                            indices[cursor] = i * n_theta + j + 1
                            data[cursor] = -coefficient
                            cursor += 1
                            diag += coefficient
                    if j > 0:
                        coefficient = face_k(gaps[i, j], gaps[i, j - 1])
                        if coefficient > 0.0:
                            coefficient /= r_i * dtheta_sq
                            indices[cursor] = i * n_theta + j - 1
                            data[cursor] = -coefficient
                            cursor += 1
                            diag += coefficient

            indices[cursor] = row
            data[cursor] = diag

    return indices, data, b


def create_diffusion_matrix_polar(
    gaps: np.ndarray,
    r_inner: float,
    r_outer: float,
    theta_extent: float,
    theta_bc_code: int,
    p_inner: float,
    p_outer: float,
) -> Tuple[csr_matrix, np.ndarray, float, float]:
    """Create sparse matrix for polar diffusion problem."""
    gaps = validate_gap_array(
        gaps, geometry="Polar", minimum_shape=(2, 1)
    )
    n_r, n_theta = gaps.shape

    if n_r < 2:
        raise ValueError("Need at least two radial nodes.")
    if r_outer <= r_inner:
        raise ValueError("Outer radius must exceed inner radius.")
    if r_inner <= 0.0:
        raise ValueError("Inner radius must be positive.")
    if theta_extent <= 0.0 or not np.isfinite(theta_extent):
        raise ValueError("Angular extent must be finite and positive.")

    dr = (r_outer - r_inner) / (n_r - 1)
    if n_theta <= 1:
        dtheta = theta_extent
    elif theta_bc_code == THETA_BC_PERIODIC:
        dtheta = theta_extent / n_theta
    else:
        dtheta = theta_extent / (n_theta - 1)

    total_dofs = n_r * n_theta
    if total_dofs > np.iinfo(np.int32).max:
        raise OverflowError("Polar grid exceeds int32 matrix index capacity.")
    counts = _count_full_matrix_entries_polar(
        n_r, n_theta, gaps, theta_bc_code
    )
    indptr = indptr_from_counts(counts)
    indices, data, b = _fill_full_matrix_csr_polar(
        n_r,
        n_theta,
        gaps,
        r_inner,
        dr,
        dtheta,
        p_inner,
        p_outer,
        theta_bc_code,
        indptr,
    )
    A = csr_matrix(
        (data, indices, indptr),
        shape=(total_dofs, total_dofs),
        dtype=np.float64,
    )
    A.sort_indices()
    return A, b, dr, dtheta


@njit
def _count_active_matrix_entries_polar(
    n_r, n_theta, theta_bc_code, grid_to_dof, dof_to_grid
):
    active_count = dof_to_grid.size
    counts = np.empty(active_count, dtype=np.int32)
    for row in range(active_count):
        grid_index = int(dof_to_grid[row])
        i = grid_index // n_theta
        j = grid_index - i * n_theta
        if i == 0 or i == n_r - 1:
            counts[row] = 1
            continue
        count = 1
        if i + 1 < n_r - 1:
            if grid_to_dof[(i + 1) * n_theta + j] >= 0:
                count += 1
        if i - 1 > 0:
            if grid_to_dof[(i - 1) * n_theta + j] >= 0:
                count += 1
        if n_theta > 1:
            if theta_bc_code == THETA_BC_PERIODIC:
                if grid_to_dof[i * n_theta + (j + 1) % n_theta] >= 0:
                    count += 1
                if grid_to_dof[i * n_theta + (j - 1) % n_theta] >= 0:
                    count += 1
            else:
                if j + 1 < n_theta:
                    if grid_to_dof[i * n_theta + j + 1] >= 0:
                        count += 1
                if j > 0:
                    if grid_to_dof[i * n_theta + j - 1] >= 0:
                        count += 1
        counts[row] = count
    return counts


@njit
def _fill_active_matrix_csr_polar_into(
    n_r,
    n_theta,
    gaps,
    r_inner,
    dr,
    dtheta,
    p_inner,
    p_outer,
    theta_bc_code,
    grid_to_dof,
    dof_to_grid,
    indptr,
    indices,
    data,
    b,
):
    active_count = dof_to_grid.size
    dtheta_sq = dtheta * dtheta if n_theta > 1 else 1.0

    for row in range(active_count):
        grid_index = int(dof_to_grid[row])
        i = grid_index // n_theta
        j = grid_index - i * n_theta
        cursor = int(indptr[row])

        if i == 0:
            indices[cursor] = row
            data[cursor] = 1.0
            b[row] = p_inner
            continue
        if i == n_r - 1:
            indices[cursor] = row
            data[cursor] = 1.0
            b[row] = p_outer
            continue

        r_i = r_inner + i * dr
        diag = 0.0
        coefficient = face_k(gaps[i, j], gaps[i + 1, j])
        if coefficient > 0.0:
            coefficient *= (
                r_inner + (i + 0.5) * dr
            ) / (dr * dr)
            diag += coefficient
            if i + 1 == n_r - 1:
                b[row] += coefficient * p_outer
            else:
                neighbor = int(grid_to_dof[(i + 1) * n_theta + j])
                if neighbor >= 0:
                    indices[cursor] = neighbor
                    data[cursor] = -coefficient
                    cursor += 1

        coefficient = face_k(gaps[i, j], gaps[i - 1, j])
        if coefficient > 0.0:
            coefficient *= (
                r_inner + (i - 0.5) * dr
            ) / (dr * dr)
            diag += coefficient
            if i - 1 == 0:
                b[row] += coefficient * p_inner
            else:
                neighbor = int(grid_to_dof[(i - 1) * n_theta + j])
                if neighbor >= 0:
                    indices[cursor] = neighbor
                    data[cursor] = -coefficient
                    cursor += 1

        if n_theta > 1 and dtheta > 0.0:
            if theta_bc_code == THETA_BC_PERIODIC:
                j_plus = (j + 1) % n_theta
                neighbor = int(grid_to_dof[i * n_theta + j_plus])
                if neighbor >= 0:
                    coefficient = face_k(gaps[i, j], gaps[i, j_plus])
                    coefficient /= r_i * dtheta_sq
                    indices[cursor] = neighbor
                    data[cursor] = -coefficient
                    cursor += 1
                    diag += coefficient

                j_minus = (j - 1) % n_theta
                neighbor = int(grid_to_dof[i * n_theta + j_minus])
                if neighbor >= 0:
                    coefficient = face_k(gaps[i, j], gaps[i, j_minus])
                    coefficient /= r_i * dtheta_sq
                    indices[cursor] = neighbor
                    data[cursor] = -coefficient
                    cursor += 1
                    diag += coefficient
            else:
                if j + 1 < n_theta:
                    neighbor = int(grid_to_dof[i * n_theta + j + 1])
                    if neighbor >= 0:
                        coefficient = face_k(gaps[i, j], gaps[i, j + 1])
                        coefficient /= r_i * dtheta_sq
                        indices[cursor] = neighbor
                        data[cursor] = -coefficient
                        cursor += 1
                        diag += coefficient
                if j > 0:
                    neighbor = int(grid_to_dof[i * n_theta + j - 1])
                    if neighbor >= 0:
                        coefficient = face_k(gaps[i, j], gaps[i, j - 1])
                        coefficient /= r_i * dtheta_sq
                        indices[cursor] = neighbor
                        data[cursor] = -coefficient
                        cursor += 1
                        diag += coefficient

        indices[cursor] = row
        data[cursor] = diag


@njit
def _fill_active_matrix_csr_polar(
    n_r,
    n_theta,
    gaps,
    r_inner,
    dr,
    dtheta,
    p_inner,
    p_outer,
    theta_bc_code,
    grid_to_dof,
    dof_to_grid,
    indptr,
):
    active_count = dof_to_grid.size
    indices = np.empty(int(indptr[-1]), dtype=np.int32)
    data = np.empty(int(indptr[-1]), dtype=np.float64)
    b = np.zeros(active_count, dtype=np.float64)
    _fill_active_matrix_csr_polar_into(
        n_r,
        n_theta,
        gaps,
        r_inner,
        dr,
        dtheta,
        p_inner,
        p_outer,
        theta_bc_code,
        grid_to_dof,
        dof_to_grid,
        indptr,
        indices,
        data,
        b,
    )
    return indices, data, b


def create_active_diffusion_matrix_polar(
    gaps: np.ndarray,
    r_inner: float,
    r_outer: float,
    theta_extent: float,
    theta_bc_code: int,
    p_inner: float,
    p_outer: float,
):
    """Create a compact polar matrix containing positive cells only."""
    gaps = validate_gap_array(
        gaps, geometry="Polar", minimum_shape=(2, 1)
    )
    if r_inner <= 0.0 or r_outer <= r_inner:
        raise ValueError("Require 0 < r_inner < r_outer.")
    if theta_extent <= 0.0 or not np.isfinite(theta_extent):
        raise ValueError("Angular extent must be finite and positive.")
    n_r, n_theta = gaps.shape
    dr = (r_outer - r_inner) / (n_r - 1)
    if n_theta <= 1:
        dtheta = theta_extent
    elif theta_bc_code == THETA_BC_PERIODIC:
        dtheta = theta_extent / n_theta
    else:
        dtheta = theta_extent / (n_theta - 1)

    grid_to_dof, dof_to_grid = build_active_mapping(gaps)
    active_count = dof_to_grid.size
    counts = _count_active_matrix_entries_polar(
        n_r, n_theta, theta_bc_code, grid_to_dof, dof_to_grid
    )
    indptr = indptr_from_counts(counts)
    indices, data, b = _fill_active_matrix_csr_polar(
        n_r,
        n_theta,
        gaps,
        r_inner,
        dr,
        dtheta,
        p_inner,
        p_outer,
        theta_bc_code,
        grid_to_dof,
        dof_to_grid,
        indptr,
    )
    matrix = csr_matrix(
        (data, indices, indptr),
        shape=(active_count, active_count),
        dtype=np.float64,
    )
    matrix.sort_indices()
    return matrix, b, dr, dtheta, dof_to_grid


def _create_solver_matrix_polar(
    gaps: np.ndarray,
    r_inner: float,
    r_outer: float,
    theta_extent: float,
    theta_bc_code: int,
    p_inner: float,
    p_outer: float,
):
    if np.all(gaps > 0.0):
        matrix, rhs, dr, dtheta = create_diffusion_matrix_polar(
            gaps,
            r_inner,
            r_outer,
            theta_extent,
            theta_bc_code,
            p_inner,
            p_outer,
        )
        return matrix, rhs, dr, dtheta, None
    return create_active_diffusion_matrix_polar(
        gaps,
        r_inner,
        r_outer,
        theta_extent,
        theta_bc_code,
        p_inner,
        p_outer,
    )


@njit
def _calculate_face_fluxes_polar(
    gaps: np.ndarray,
    pressure: np.ndarray,
    r_inner: float,
    dr: float,
    dtheta: float,
    theta_bc_code: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate conservative radial and angular face-flux densities."""
    n_r, n_theta = gaps.shape
    flux_r = np.zeros((n_r - 1, n_theta), dtype=np.float64)
    # Face j lies between angular samples j-1 and j. For periodic grids the
    # last column duplicates face 0; for symmetry sectors both endpoints are 0.
    flux_theta = np.zeros((n_r, n_theta + 1), dtype=np.float64)

    for i in range(n_r - 1):
        for j in range(n_theta):
            conductivity = face_k(gaps[i, j], gaps[i + 1, j])
            if conductivity > 0.0:
                flux_r[i, j] = -conductivity * (
                    pressure[i + 1, j] - pressure[i, j]
                ) / dr

    if n_theta > 1 and dtheta > 0.0:
        for i in range(n_r):
            r_i = r_inner + i * dr
            if theta_bc_code == THETA_BC_PERIODIC:
                conductivity = face_k(gaps[i, n_theta - 1], gaps[i, 0])
                if conductivity > 0.0:
                    flux_theta[i, 0] = -conductivity * (
                        pressure[i, 0] - pressure[i, n_theta - 1]
                    ) / (r_i * dtheta)
                flux_theta[i, n_theta] = flux_theta[i, 0]

            for j in range(1, n_theta):
                conductivity = face_k(gaps[i, j - 1], gaps[i, j])
                if conductivity > 0.0:
                    flux_theta[i, j] = -conductivity * (
                        pressure[i, j] - pressure[i, j - 1]
                    ) / (r_i * dtheta)

    return flux_r, flux_theta


@njit
def _cell_flux_from_faces_polar(
    gaps: np.ndarray,
    flux_r: np.ndarray,
    flux_theta: np.ndarray,
    r_inner: float,
    dr: float,
) -> np.ndarray:
    """Construct legacy node-shaped polar flux while conserving boundary flow."""
    n_r, n_theta = gaps.shape
    flux = np.empty((n_r, n_theta, 2), dtype=np.float64)
    r_outer = r_inner + (n_r - 1) * dr

    for i in range(n_r):
        r_i = r_inner + i * dr
        for j in range(n_theta):
            if gaps[i, j] <= 0.0:
                flux[i, j, 0] = np.nan
                flux[i, j, 1] = np.nan
                continue

            if i == 0:
                r_face = r_inner + 0.5 * dr
                flux[i, j, 0] = flux_r[0, j] * r_face / r_inner
            elif i == n_r - 1:
                r_face = r_outer - 0.5 * dr
                flux[i, j, 0] = flux_r[n_r - 2, j] * r_face / r_outer
            else:
                r_minus = r_i - 0.5 * dr
                r_plus = r_i + 0.5 * dr
                flux[i, j, 0] = 0.5 * (
                    flux_r[i - 1, j] * r_minus
                    + flux_r[i, j] * r_plus
                ) / r_i

            flux[i, j, 1] = 0.5 * (
                flux_theta[i, j] + flux_theta[i, j + 1]
            )

    return flux


def solve_diffusion_polar(
    gaps: np.ndarray,
    r_inner: float,
    r_outer: float,
    theta_extent: float,
    theta_bc_code: int,
    solver: str = "auto",
    rtol: Optional[float] = None,
    p_inner: float = 1.0,
    p_outer: float = 0.0,
    save_matrix: bool = False,
    save_matrix_type: str = "coo",
) -> Tuple[np.ndarray, float, float]:
    """Solve the diffusion problem in polar coordinates."""
    gaps = validate_gap_array(
        gaps, geometry="Polar", minimum_shape=(2, 1)
    )
    if not np.any(gaps > 0.0):
        raise InvalidGapError("Polar gap field has no positive cells.")

    A, b, dr, dtheta, dof_to_grid = _create_solver_matrix_polar(
        gaps, r_inner, r_outer, theta_extent, theta_bc_code, p_inner, p_outer
    )

    if save_matrix:
        logger.info(f"Saving transport matrix and RHS to npz files in {save_matrix_type} format.")
        if save_matrix_type == "coo":
            save_npz("transport_matrix_polar.npz", A.tocoo(), compressed=True)
        elif save_matrix_type == "csr":
            save_npz("transport_matrix_polar.npz", A.tocsr(), compressed=True)
        elif save_matrix_type == "csc":
            save_npz("transport_matrix_polar.npz", A.tocsc(), compressed=True)
        else:
            logger.warning(f"Unknown save_matrix_type: {save_matrix_type}, defaulting to 'coo'.")
            save_npz("transport_matrix_polar.npz", A.tocoo(), compressed=True)
        np.savez_compressed("transport_rhs_polar.npz", b=b)

    effective_rtol = DEFAULT_RTOL if rtol is None else rtol
    result = solve_linear_system(A, b, solver=solver, rtol=effective_rtol)
    logger.info(
        "Linear solver %s finished (iterations=%s, relative residual=%.3e).",
        result.solver,
        result.iterations,
        result.relative_residual,
    )
    pressure = reconstruct_full_solution(
        gaps.shape, result.solution, dof_to_grid
    )
    return pressure, dr, dtheta


def connectivity_analysis_polar(gaps: np.ndarray, theta_bc_code: int) -> Optional[np.ndarray]:
    """Detect percolation between inner and outer radius."""
    periodic_axis = 1 if theta_bc_code == THETA_BC_PERIODIC else None
    mask = find_spanning_mask(
        gaps, transport_axis=0, periodic_axis=periodic_axis
    )
    if mask is None:
        logger.info("No percolation detected between inner and outer boundaries.")
        return None
    logger.info("Radially percolating channel detected.")
    return np.where(mask, gaps, 0.0)


@dataclass
class PreparedPolarProblem:
    """Reusable polar topology and CSR sparsity for gap-value sequences."""

    shape: tuple[int, int]
    original_open_mask: np.ndarray
    spanning_mask: np.ndarray
    grid_to_dof: np.ndarray
    dof_to_grid: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    r_inner: float
    r_outer: float
    theta_extent: float
    theta_bc_code: int
    theta_bc_name: str
    p_inner: float
    p_outer: float
    dr: float
    dtheta: float
    _amg_preconditioner: object = field(default=None, init=False, repr=False)
    _amg_method: str | None = field(default=None, init=False, repr=False)

    def _filtered_values(self, gaps) -> np.ndarray:
        values = validate_gap_array(
            gaps, geometry="Polar", minimum_shape=(2, 1)
        )
        if values.shape != self.shape:
            raise InvalidGapError(
                f"Prepared polar topology has shape {self.shape}, got {values.shape}."
            )
        if not np.array_equal(values > 0.0, self.original_open_mask):
            raise InvalidGapError(
                "Gap open/closed topology changed; prepare a new polar problem."
            )
        return np.where(self.spanning_mask, values, 0.0)

    def assemble(self, gaps):
        """Update coefficients/RHS while reusing mappings and CSR structure."""
        values = self._filtered_values(gaps)
        data = np.empty(self.indices.size, dtype=np.float64)
        rhs = np.zeros(self.dof_to_grid.size, dtype=np.float64)
        _fill_active_matrix_csr_polar_into(
            self.shape[0],
            self.shape[1],
            values,
            self.r_inner,
            self.dr,
            self.dtheta,
            self.p_inner,
            self.p_outer,
            self.theta_bc_code,
            self.grid_to_dof,
            self.dof_to_grid,
            self.indptr,
            self.indices,
            data,
            rhs,
        )
        matrix = csr_matrix(
            (data, self.indices, self.indptr),
            shape=(self.dof_to_grid.size, self.dof_to_grid.size),
            copy=False,
        )
        matrix.sort_indices()
        return matrix, rhs, values

    def clear_preconditioner(self):
        """Discard a cached AMG hierarchy before a strongly changed sequence."""
        self._amg_preconditioner = None
        self._amg_method = None

    def solve_with_diagnostics(
        self,
        gaps,
        solver="auto",
        rtol=DEFAULT_RTOL,
        reuse_preconditioner=False,
    ):
        matrix, rhs, values = self.assemble(gaps)
        preconditioner = None
        if reuse_preconditioner:
            normalized = normalize_solver_name(solver)
            if normalized == "auto":
                normalized = "scipy.amg-rs"
            if not normalized.startswith("scipy.amg-"):
                raise ValueError(
                    "Prepared preconditioner reuse requires a SciPy AMG solver."
                )
            method = normalized.split(".", maxsplit=1)[1]
            if self._amg_preconditioner is None or self._amg_method != method:
                self._amg_preconditioner = build_amg_preconditioner(
                    matrix, method
                )
                self._amg_method = method
            preconditioner = self._amg_preconditioner
        result = solve_linear_system(
            matrix,
            rhs,
            solver=solver,
            rtol=rtol,
            preconditioner=preconditioner,
        )
        pressure = reconstruct_full_solution(
            values.shape, result.solution, self.dof_to_grid
        )
        inner_mask = values[0, :] > 0.0
        outer_mask = values[-1, :] > 0.0
        pressure[0, inner_mask] = self.p_inner
        pressure[-1, outer_mask] = self.p_outer
        flux_r, flux_theta = _calculate_face_fluxes_polar(
            values,
            pressure,
            self.r_inner,
            self.dr,
            self.dtheta,
            self.theta_bc_code,
        )
        flux = _cell_flux_from_faces_polar(
            values, flux_r, flux_theta, self.r_inner, self.dr
        )
        flux[~self.spanning_mask] = np.nan
        return values, pressure, flux, self.dr, self.dtheta, result

    def solve(
        self,
        gaps,
        solver="auto",
        rtol=DEFAULT_RTOL,
        reuse_preconditioner=False,
    ):
        values, pressure, flux, dr, dtheta, _ = self.solve_with_diagnostics(
            gaps,
            solver=solver,
            rtol=rtol,
            reuse_preconditioner=reuse_preconditioner,
        )
        return values, pressure, flux, dr, dtheta


def prepare_fluid_problem_polar(
    gaps,
    r_inner: float,
    r_outer: float,
    *,
    theta_extent: float = 2.0 * np.pi,
    theta_bc: str = "auto",
    p_inner: float = 1.0,
    p_outer: float = 0.0,
):
    """Prepare reusable polar connectivity, mappings, and CSR topology."""
    values = validate_gap_array(
        gaps, geometry="Polar", minimum_shape=(2, 1)
    )
    if r_inner <= 0.0 or r_outer <= r_inner:
        raise ValueError("Require 0 < r_inner < r_outer.")
    if theta_extent <= 0.0 or not np.isfinite(theta_extent):
        raise ValueError("Angular extent must be finite and positive.")
    if theta_bc == "auto":
        theta_bc = (
            "periodic" if np.isclose(theta_extent, 2.0 * np.pi) else "symmetry"
        )
    theta_bc_name = theta_bc.lower()
    if theta_bc_name not in {"periodic", "symmetry"}:
        raise ValueError("theta_bc must be 'auto', 'periodic', or 'symmetry'.")
    theta_bc_code = (
        THETA_BC_PERIODIC
        if theta_bc_name == "periodic"
        else THETA_BC_SYMMETRY
    )

    filtered = connectivity_analysis_polar(values, theta_bc_code)
    if filtered is None:
        return None
    n_r, n_theta = values.shape
    dr = (r_outer - r_inner) / (n_r - 1)
    if n_theta == 1:
        dtheta = theta_extent
    elif theta_bc_code == THETA_BC_PERIODIC:
        dtheta = theta_extent / n_theta
    else:
        dtheta = theta_extent / (n_theta - 1)

    grid_to_dof, dof_to_grid = build_active_mapping(filtered)
    counts = _count_active_matrix_entries_polar(
        n_r, n_theta, theta_bc_code, grid_to_dof, dof_to_grid
    )
    indptr = indptr_from_counts(counts)
    indices, _, _ = _fill_active_matrix_csr_polar(
        n_r,
        n_theta,
        filtered,
        r_inner,
        dr,
        dtheta,
        p_inner,
        p_outer,
        theta_bc_code,
        grid_to_dof,
        dof_to_grid,
        indptr,
    )
    return PreparedPolarProblem(
        shape=values.shape,
        original_open_mask=(values > 0.0).copy(),
        spanning_mask=(filtered > 0.0).copy(),
        grid_to_dof=grid_to_dof,
        dof_to_grid=dof_to_grid,
        indptr=indptr,
        indices=indices,
        r_inner=r_inner,
        r_outer=r_outer,
        theta_extent=theta_extent,
        theta_bc_code=theta_bc_code,
        theta_bc_name=theta_bc_name,
        p_inner=p_inner,
        p_outer=p_outer,
        dr=dr,
        dtheta=dtheta,
    )


@jit(nopython=True)
def _dilate_gaps_polar(gaps: np.ndarray, iterations: int, theta_bc_code: int) -> np.ndarray:
    n_r, n_theta = gaps.shape
    dilated = gaps.copy()

    for _ in range(iterations):
        temp = dilated.copy()
        for i in range(n_r):
            for j in range(n_theta):
                if dilated[i, j] == 0.0:
                    max_neighbor = 0.0
                    if i > 0 and dilated[i - 1, j] > max_neighbor:
                        max_neighbor = dilated[i - 1, j]
                    if i < n_r - 1 and dilated[i + 1, j] > max_neighbor:
                        max_neighbor = dilated[i + 1, j]

                    if n_theta > 1:
                        if theta_bc_code == THETA_BC_PERIODIC:
                            j_minus = (j - 1 + n_theta) % n_theta
                            j_plus = (j + 1) % n_theta
                            if dilated[i, j_minus] > max_neighbor:
                                max_neighbor = dilated[i, j_minus]
                            if dilated[i, j_plus] > max_neighbor:
                                max_neighbor = dilated[i, j_plus]
                        else:
                            if j > 0 and dilated[i, j - 1] > max_neighbor:
                                max_neighbor = dilated[i, j - 1]
                            if j < n_theta - 1 and dilated[i, j + 1] > max_neighbor:
                                max_neighbor = dilated[i, j + 1]

                    if max_neighbor > 0.0:
                        temp[i, j] = max_neighbor
        dilated = temp

    return dilated


def solve_fluid_problem_polar(
    gaps: np.ndarray,
    r_inner: float,
    r_outer: float,
    solver: str = "auto",
    rtol: Optional[float] = None,
    theta_extent: float = 2.0 * np.pi,
    theta_bc: str = "auto",
    p_inner: float = 1.0,
    p_outer: float = 0.0,
    dilation_iterations: int = 0,
    save_matrix: bool = False,
    save_matrix_type: str = "coo",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Solve the Reynolds equation on a polar grid defined between r_inner and r_outer.

    Parameters
    ----------
    gaps : np.ndarray
        Gap field array of shape (n_r, n_theta).
    r_inner : float
        Inner radius (> 0).
    r_outer : float
        Outer radius (> r_inner).
    solver : str, optional
        Solver specification (see `solve_diffusion_polar`).
    rtol : float, optional
        Relative tolerance for iterative solvers.
    theta_extent : float, optional
        Angular extent of the computational domain in radians. Default is full circle (2π).
    theta_bc : str, optional
        Angular boundary condition: 'auto', 'periodic', or 'symmetry'. 'auto' selects
        'periodic' when theta_extent ~ 2π, otherwise 'symmetry'.
    p_inner : float, optional
        Pressure at the inner radius.
    p_outer : float, optional
        Pressure at the outer radius.
    dilation_iterations : int, optional
        Number of explicit geometry-dilation iterations (default 0). Positive
        values modify the solved channel and are retained for experiments.
    save_matrix : bool, optional
        If True, store matrix and RHS to disk for debugging.
    save_matrix_type : str, optional
        Storage type for matrix saving ('coo', 'csr', 'csc').

    Returns
    -------
    (gaps_filtered, pressure, flux, dr, dtheta)
    """
    logger.info("Starting polar fluid solver.")

    gaps = validate_gap_array(
        gaps, geometry="Polar", minimum_shape=(2, 1)
    )

    n_r, n_theta = gaps.shape
    if n_r < 2:
        raise ValueError("Polar solver requires at least two radial nodes.")
    if r_inner <= 0.0:
        raise ValueError("Inner radius must be positive for polar solver.")
    if r_outer <= r_inner:
        raise ValueError("Outer radius must exceed inner radius.")
    if theta_extent <= 0.0:
        raise ValueError("Angular extent must be positive.")
    if not isinstance(dilation_iterations, (int, np.integer)) or dilation_iterations < 0:
        raise ValueError("dilation_iterations must be a non-negative integer.")

    if theta_bc == "auto":
        theta_bc = "periodic" if np.isclose(theta_extent, 2.0 * np.pi) else "symmetry"

    theta_bc_lower = theta_bc.lower()
    if theta_bc_lower not in {"periodic", "symmetry"}:
        raise ValueError("theta_bc must be 'auto', 'periodic', or 'symmetry'.")
    theta_bc_code = THETA_BC_PERIODIC if theta_bc_lower == "periodic" else THETA_BC_SYMMETRY

    logger.info("Checking connectivity between inner and outer radii.")
    start = time.time()
    gaps_filtered = connectivity_analysis_polar(gaps, theta_bc_code)
    logger.info("Connectivity analysis: CPU time = %.3f sec", time.time() - start)

    if gaps_filtered is None:
        logger.warning("No percolating channel detected. Returning None results.")
        return None, None, None, None, None

    if dilation_iterations > 0:
        logger.info("Applying dilation (%d iterations) to preserve boundary channels.", dilation_iterations)
        gaps_dilated = _dilate_gaps_polar(gaps_filtered, dilation_iterations, theta_bc_code)
    else:
        gaps_dilated = gaps_filtered

    logger.info("Solving diffusion problem in polar coordinates.")
    start_time = time.time()
    pressure, dr, dtheta = solve_diffusion_polar(
        gaps_dilated,
        r_inner,
        r_outer,
        theta_extent,
        theta_bc_code,
        solver=solver,
        rtol=rtol,
        p_inner=p_inner,
        p_outer=p_outer,
        save_matrix=save_matrix,
        save_matrix_type=save_matrix_type,
    )
    logger.info("Fluid solver: CPU time = %.3f sec", time.time() - start_time)

    # Explicitly enforce Dirichlet boundary values (useful after iterative solves)
    if n_theta > 0:
        inner_mask = gaps_dilated[0, :] > 0.0
        outer_mask = gaps_dilated[-1, :] > 0.0
        if np.any(inner_mask):
            pressure[0, inner_mask] = p_inner
        if np.any(outer_mask):
            pressure[-1, outer_mask] = p_outer

    logger.info("Calculating conservative face and visualization flux.")
    flux_r, flux_theta = _calculate_face_fluxes_polar(
        gaps_dilated,
        pressure,
        r_inner,
        dr,
        dtheta,
        theta_bc_code,
    )
    flux = _cell_flux_from_faces_polar(
        gaps_dilated, flux_r, flux_theta, r_inner, dr
    )

    # Mask flux outside the original percolating channel
    channel_mask = gaps_filtered > 0.0
    flux[~channel_mask] = np.nan

    logger.info("Polar fluid solver finished.")

    return gaps_filtered, pressure, flux, dr, dtheta


def get_preconditioner(A, method="amg-rs"):
    """Backward-compatible wrapper around the shared AMG implementation."""
    aliases = {
        "amg.rs": "amg-rs",
        "amg-sa": "amg-smooth_aggregation",
    }
    return build_amg_preconditioner(A, aliases.get(method, method))


def compute_total_flux_polar(
    filtered_gaps: np.ndarray,
    flux: np.ndarray,
    r_inner: float,
    r_outer: float,
    dtheta: float,
    theta_bc: str = "periodic",
) -> Tuple[float, float]:
    """
    Compute total flux and conservation error across inner and outer boundaries.
    Returns Q_total and flux_conservation_error.
    """
    if filtered_gaps is None or flux is None:
        raise ValueError("Flux computation requires valid gaps and flux arrays.")

    filtered_gaps = np.asarray(filtered_gaps)
    flux = np.asarray(flux)
    if filtered_gaps.ndim != 2 or flux.shape != filtered_gaps.shape + (2,):
        raise ValueError(
            "Expected gaps with shape (n_r, n_theta) and flux with shape "
            "(n_r, n_theta, 2)."
        )
    if r_inner <= 0.0 or r_outer <= r_inner:
        raise ValueError("Require 0 < r_inner < r_outer.")
    if dtheta <= 0.0 or not np.isfinite(dtheta):
        raise ValueError("dtheta must be finite and positive.")

    n_r, n_theta = filtered_gaps.shape
    if n_r < 2:
        raise ValueError("Flux computation requires at least two radial nodes.")

    theta_bc_lower = theta_bc.lower()
    if theta_bc_lower not in {"periodic", "symmetry"}:
        raise ValueError("theta_bc must be 'periodic' or 'symmetry'.")

    flux_inner = 0.0
    active_inner = 0
    for j in range(n_theta):
        if not np.isnan(flux[0, j, 0]) and filtered_gaps[0, j] > 0:
            weight = 1.0
            if theta_bc_lower == "symmetry" and n_theta > 1:
                if j == 0 or j == n_theta - 1:
                    weight = 0.5
            flux_inner += flux[0, j, 0] * r_inner * dtheta * weight
            active_inner += 1

    flux_outer = 0.0
    active_outer = 0
    for j in range(n_theta):
        if not np.isnan(flux[-1, j, 0]) and filtered_gaps[-1, j] > 0:
            weight = 1.0
            if theta_bc_lower == "symmetry" and n_theta > 1:
                if j == 0 or j == n_theta - 1:
                    weight = 0.5
            flux_outer += flux[-1, j, 0] * r_outer * dtheta * weight
            active_outer += 1

    Q_total = 0.5 * (abs(flux_inner) + abs(flux_outer))
    flux_conservation_error = abs(flux_inner - flux_outer) / max(Q_total, 1e-15)

    logger.info("> Flux computation (polar) <")
    logger.info(
        "Inner flux (r = %.3e):     %.6e [Active cells: %d]", r_inner, flux_inner, active_inner
    )
    logger.info(
        "Outer flux (r = %.3e):     %.6e [Active cells: %d]", r_outer, flux_outer, active_outer
    )
    logger.info("Total average flux (Q_total): %.6e", Q_total)
    logger.info(
        "Conservation error:   %.2e (%.2f%%)",
        flux_conservation_error,
        flux_conservation_error * 100.0,
    )

    return Q_total, flux_conservation_error


def warmup_numba_functions_polar():
    """Explicitly compile polar Numba kernels on tiny arrays."""
    logger.info("Warming up polar Numba kernels.")
    n_r, n_theta = 3, 4
    gaps = np.full((n_r, n_theta), 0.1, dtype=np.float64)
    create_diffusion_matrix_polar(
        gaps,
        1.0,
        1.2,
        2.0 * np.pi,
        THETA_BC_PERIODIC,
        1.0,
        0.0,
    )

    blocked = gaps.copy()
    blocked[1, 1] = 0.0
    create_active_diffusion_matrix_polar(
        blocked,
        1.0,
        1.2,
        2.0 * np.pi,
        THETA_BC_PERIODIC,
        1.0,
        0.0,
    )
    _dilate_gaps_polar(blocked, 1, THETA_BC_PERIODIC)

    pressure = np.repeat(
        np.linspace(1.0, 0.0, n_r, dtype=np.float64)[:, None],
        n_theta,
        axis=1,
    )
    dr = 0.1
    dtheta = 2.0 * np.pi / n_theta
    flux_r, flux_theta = _calculate_face_fluxes_polar(
        gaps,
        pressure,
        1.0,
        dr,
        dtheta,
        THETA_BC_PERIODIC,
    )
    _cell_flux_from_faces_polar(gaps, flux_r, flux_theta, 1.0, dr)
    logger.info("Polar Numba kernels warmed up.")


_warmup_numba_functions_polar = warmup_numba_functions_polar
