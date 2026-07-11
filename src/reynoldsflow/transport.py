"""
Finite-Difference Reynolds Fluid Flow Solver

Author: Vladislav A. Yastrebov (CNRS, Mines Paris - PSL)
AI: Cursor, Claude, ChatGPT
Date: Aug 2024-Mar 2026
License: BSD 3-Clause
"""

# TODO: adapt for compressible fluids (requires only postprocessing)

import numpy as np
from dataclasses import dataclass, field
from scipy.sparse import csr_matrix, save_npz
from numba import jit, njit
import time
import logging

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

# If need control over number of threads
# os.environ['MKL_NUM_THREADS'] = '4'
# os.environ['OMP_NUM_THREADS'] = '4'

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

def set_verbosity(level='info'):
    levels = {
        'debug': logging.DEBUG,
        'info': logging.INFO,
        'warning': logging.WARNING,
        'error': logging.ERROR,
        'critical': logging.CRITICAL,
    }
    logger.setLevel(levels.get(level.lower(), logging.INFO))

# new version of matrix builder
@njit
def face_k(a, b):
    # harmonic mean; if either side blocked -> 0
    if a <= 0.0 or b <= 0.0:
        return 0.0
    return 2.0 * (a**3) * (b**3) / (a**3 + b**3)

@njit
def _count_full_matrix_entries(n, g):
    counts = np.empty(n * n, dtype=np.int32)
    for i in range(n):
        for j in range(n):
            row = i * n + j
            if g[i, j] <= 0.0:
                counts[row] = 1
                continue
            count = 1  # diagonal
            if i > 0 and g[i - 1, j] > 0.0:
                count += 1
            if i + 1 < n and g[i + 1, j] > 0.0:
                count += 1
            if g[i, (j + 1) % n] > 0.0:
                count += 1
            if g[i, (j - 1) % n] > 0.0:
                count += 1
            counts[row] = count
    return counts


@njit
def _fill_full_matrix_csr(n, g, indptr, p_west, p_east):
    nnz = int(indptr[-1])
    indices = np.empty(nnz, dtype=np.int32)
    data = np.empty(nnz, dtype=np.float64)
    b = np.zeros(n * n, dtype=np.float64)

    for i in range(n):
        for j in range(n):
            row = i * n + j
            cursor = int(indptr[row])
            if g[i, j] <= 0.0:
                indices[cursor] = row
                data[cursor] = 1.0
                continue

            diag = 0.0
            if i == 0:
                coefficient = 2.0 * g[i, j] ** 3
                diag += coefficient
                b[row] += coefficient * p_west
            elif g[i - 1, j] > 0.0:
                coefficient = face_k(g[i, j], g[i - 1, j])
                indices[cursor] = (i - 1) * n + j
                data[cursor] = -coefficient
                cursor += 1
                diag += coefficient

            if i == n - 1:
                coefficient = 2.0 * g[i, j] ** 3
                diag += coefficient
                b[row] += coefficient * p_east
            elif g[i + 1, j] > 0.0:
                coefficient = face_k(g[i, j], g[i + 1, j])
                indices[cursor] = (i + 1) * n + j
                data[cursor] = -coefficient
                cursor += 1
                diag += coefficient

            j_plus = (j + 1) % n
            if g[i, j_plus] > 0.0:
                coefficient = face_k(g[i, j], g[i, j_plus])
                indices[cursor] = i * n + j_plus
                data[cursor] = -coefficient
                cursor += 1
                diag += coefficient

            j_minus = (j - 1) % n
            if g[i, j_minus] > 0.0:
                coefficient = face_k(g[i, j], g[i, j_minus])
                indices[cursor] = i * n + j_minus
                data[cursor] = -coefficient
                cursor += 1
                diag += coefficient

            indices[cursor] = row
            data[cursor] = diag

    return indices, data, b


def create_diffusion_matrix(
    n, g, penalty=None, p_west=0.0, p_east=1.0
):
    """
    Create the sparse matrix for the diffusion problem with non-homogeneous gap field, properly handling zero or near-zero gap regions.
    """
    g = validate_gap_array(
        g,
        geometry="Cartesian",
        require_square=True,
        minimum_shape=(2, 2),
    )
    if n != g.shape[0]:
        raise InvalidGapError(
            f"Matrix size n={n} does not match gap shape {g.shape}."
        )
    N = n * n
    if N > np.iinfo(np.int32).max:
        raise OverflowError("Cartesian grid exceeds int32 matrix index capacity.")
    counts = _count_full_matrix_entries(n, g)
    indptr = indptr_from_counts(counts)
    if not np.isfinite(p_west) or not np.isfinite(p_east):
        raise ValueError("Reservoir pressures must be finite.")
    indices, data, b = _fill_full_matrix_csr(
        n, g, indptr, p_west, p_east
    )
    A = csr_matrix((data, indices, indptr), shape=(N, N), dtype=np.float64)
    A.sort_indices()
    return A, b


@njit
def _count_active_matrix_entries(n, grid_to_dof, dof_to_grid):
    active_count = dof_to_grid.size
    counts = np.empty(active_count, dtype=np.int32)
    for row in range(active_count):
        grid_index = int(dof_to_grid[row])
        i = grid_index // n
        j = grid_index - i * n
        count = 1
        if i > 0 and grid_to_dof[(i - 1) * n + j] >= 0:
            count += 1
        if i + 1 < n and grid_to_dof[(i + 1) * n + j] >= 0:
            count += 1
        if grid_to_dof[i * n + (j + 1) % n] >= 0:
            count += 1
        if grid_to_dof[i * n + (j - 1) % n] >= 0:
            count += 1
        counts[row] = count
    return counts


@njit
def _fill_active_matrix_csr_into(
    n,
    g,
    grid_to_dof,
    dof_to_grid,
    indptr,
    indices,
    data,
    b,
    p_west,
    p_east,
):
    active_count = dof_to_grid.size

    for row in range(active_count):
        grid_index = int(dof_to_grid[row])
        i = grid_index // n
        j = grid_index - i * n
        cursor = int(indptr[row])
        diag = 0.0

        if i == 0:
            coefficient = 2.0 * g[i, j] ** 3
            diag += coefficient
            b[row] += coefficient * p_west
        else:
            neighbor = int(grid_to_dof[(i - 1) * n + j])
            if neighbor >= 0:
                coefficient = face_k(g[i, j], g[i - 1, j])
                indices[cursor] = neighbor
                data[cursor] = -coefficient
                cursor += 1
                diag += coefficient

        if i == n - 1:
            coefficient = 2.0 * g[i, j] ** 3
            diag += coefficient
            b[row] += coefficient * p_east
        else:
            neighbor = int(grid_to_dof[(i + 1) * n + j])
            if neighbor >= 0:
                coefficient = face_k(g[i, j], g[i + 1, j])
                indices[cursor] = neighbor
                data[cursor] = -coefficient
                cursor += 1
                diag += coefficient

        j_plus = (j + 1) % n
        neighbor = int(grid_to_dof[i * n + j_plus])
        if neighbor >= 0:
            coefficient = face_k(g[i, j], g[i, j_plus])
            indices[cursor] = neighbor
            data[cursor] = -coefficient
            cursor += 1
            diag += coefficient

        j_minus = (j - 1) % n
        neighbor = int(grid_to_dof[i * n + j_minus])
        if neighbor >= 0:
            coefficient = face_k(g[i, j], g[i, j_minus])
            indices[cursor] = neighbor
            data[cursor] = -coefficient
            cursor += 1
            diag += coefficient

        indices[cursor] = row
        data[cursor] = diag


@njit
def _fill_active_matrix_csr(
    n,
    g,
    grid_to_dof,
    dof_to_grid,
    indptr,
    p_west=0.0,
    p_east=1.0,
):
    active_count = dof_to_grid.size
    indices = np.empty(int(indptr[-1]), dtype=np.int32)
    data = np.empty(int(indptr[-1]), dtype=np.float64)
    b = np.zeros(active_count, dtype=np.float64)
    _fill_active_matrix_csr_into(
        n,
        g,
        grid_to_dof,
        dof_to_grid,
        indptr,
        indices,
        data,
        b,
        p_west,
        p_east,
    )
    return indices, data, b


def create_active_diffusion_matrix(g, p_west=0.0, p_east=1.0):
    """Create a compact Cartesian matrix containing positive cells only."""
    g = validate_gap_array(
        g,
        geometry="Cartesian",
        require_square=True,
        minimum_shape=(2, 2),
    )
    n = g.shape[0]
    if not np.isfinite(p_west) or not np.isfinite(p_east):
        raise ValueError("Reservoir pressures must be finite.")
    grid_to_dof, dof_to_grid = build_active_mapping(g)
    active_count = dof_to_grid.size
    counts = _count_active_matrix_entries(n, grid_to_dof, dof_to_grid)
    indptr = indptr_from_counts(counts)
    indices, data, b = _fill_active_matrix_csr(
        n,
        g,
        grid_to_dof,
        dof_to_grid,
        indptr,
        p_west,
        p_east,
    )
    matrix = csr_matrix(
        (data, indices, indptr),
        shape=(active_count, active_count),
        dtype=np.float64,
    )
    matrix.sort_indices()
    return matrix, b, dof_to_grid


def _create_solver_matrix(g, p_west=0.0, p_east=1.0):
    """Avoid mapping overhead for fully open grids; compact all others."""
    if np.all(g > 0.0):
        matrix, rhs = create_diffusion_matrix(
            g.shape[0], g, p_west=p_west, p_east=p_east
        )
        return matrix, rhs, None
    return create_active_diffusion_matrix(
        g, p_west=p_west, p_east=p_east
    )

@jit(nopython=True)
def _threshold_numba(matrix, z0):
    """Numba-accelerated threshold function"""
    result = np.zeros_like(matrix, dtype=np.int32)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] > z0:
                result[i, j] = 1
    return result

def threshold(matrix, z0):
    return _threshold_numba(matrix, z0)

@njit
def _calculate_face_fluxes_numba(gaps, pressure, p_west=0.0, p_east=1.0):
    """Return conservative Cartesian face-flux densities.

    ``flux_x[i, j]`` is the x-directed flux on face ``i`` where faces 0 and
    ``n`` are the reservoir boundaries. ``flux_y[i, j]`` is the flux from
    angular/periodic cell ``j`` toward ``(j + 1) % n``.
    """
    n = gaps.shape[0]
    dx = 1.0 / n
    dy = 1.0 / n
    flux_x = np.zeros((n + 1, n), dtype=np.float64)
    flux_y = np.zeros((n, n), dtype=np.float64)

    for j in range(n):
        if gaps[0, j] > 0.0:
            flux_x[0, j] = -(gaps[0, j] ** 3) * (
                pressure[0, j] - p_west
            ) / (0.5 * dx)

        for i in range(1, n):
            conductivity = face_k(gaps[i - 1, j], gaps[i, j])
            if conductivity > 0.0:
                flux_x[i, j] = -conductivity * (
                    pressure[i, j] - pressure[i - 1, j]
                ) / dx

        if gaps[n - 1, j] > 0.0:
            flux_x[n, j] = -(gaps[n - 1, j] ** 3) * (
                p_east - pressure[n - 1, j]
            ) / (0.5 * dx)

    for i in range(n):
        for j in range(n):
            j_plus = (j + 1) % n
            conductivity = face_k(gaps[i, j], gaps[i, j_plus])
            if conductivity > 0.0:
                flux_y[i, j] = -conductivity * (
                    pressure[i, j_plus] - pressure[i, j]
                ) / dy

    return flux_x, flux_y


@njit
def _cell_flux_from_faces_numba(gaps, flux_x, flux_y):
    """Construct the legacy cell-shaped visualization flux from face fluxes."""
    n = gaps.shape[0]
    flux = np.zeros((n, n, 2), dtype=np.float64)

    for i in range(n):
        for j in range(n):
            if gaps[i, j] <= 0.0:
                continue

            if i == 0:
                flux[i, j, 0] = flux_x[0, j]
            elif i == n - 1:
                flux[i, j, 0] = flux_x[n, j]
            else:
                flux[i, j, 0] = 0.5 * (
                    flux_x[i, j] + flux_x[i + 1, j]
                )

            j_minus = (j - 1) % n
            flux[i, j, 1] = 0.5 * (
                flux_y[i, j_minus] + flux_y[i, j]
            )

    return flux

def solve_diffusion(
    n,
    g,
    solver="auto",
    rtol=None,
    save_matrix=False,
    save_matrix_type="coo",
    p_west=0.0,
    p_east=1.0,
):
    """Solve diffusion with external reservoir boundary conditions"""
    g = validate_gap_array(
        g,
        geometry="Cartesian",
        require_square=True,
        minimum_shape=(2, 2),
    )
    if n != g.shape[0]:
        raise InvalidGapError(
            f"Solver size n={n} does not match gap shape {g.shape}."
        )
    if not np.any(g > 0.0):
        raise InvalidGapError("Cartesian gap field has no positive cells.")
    
    A, b, dof_to_grid = _create_solver_matrix(g, p_west, p_east)

    # Save matrix and RHS for debugging
    if save_matrix:
        logger.info(f"Saving transport matrix and RHS to npz files in {save_matrix_type} format.")
        if save_matrix_type == "coo":
            save_npz("transport_matrix.npz", A.tocoo(), compressed=True)
        elif save_matrix_type == "csr":
            save_npz("transport_matrix.npz", A.tocsr(), compressed=True)
        elif save_matrix_type == "csc":
            save_npz("transport_matrix.npz", A.tocsc(), compressed=True)
        else:   
            logger.warning(f"Unknown save_matrix_type: {save_matrix_type}, defaulting to 'coo'.")
            save_npz("transport_matrix.npz", A.tocoo(), compressed=True)
        np.savez_compressed("transport_rhs.npz", b=b)

    effective_rtol = DEFAULT_RTOL if rtol is None else rtol
    result = solve_linear_system(A, b, solver=solver, rtol=effective_rtol)
    logger.info(
        "Linear solver %s finished (iterations=%s, relative residual=%.3e).",
        result.solver,
        result.iterations,
        result.relative_residual,
    )
    return reconstruct_full_solution(g.shape, result.solution, dof_to_grid)

def connectivity_analysis(gaps):
    mask = find_spanning_mask(gaps, transport_axis=0, periodic_axis=1)
    if mask is None:
        logger.info("No percolation detected.")
        return None
    logger.info("Percolation detected.")
    return np.where(mask, gaps, 0.0)


@dataclass
class PreparedCartesianProblem:
    """Reusable Cartesian topology and CSR sparsity for gap-value sequences."""

    n: int
    original_open_mask: np.ndarray
    spanning_mask: np.ndarray
    grid_to_dof: np.ndarray
    dof_to_grid: np.ndarray
    indptr: np.ndarray
    indices: np.ndarray
    p_west: float
    p_east: float
    _amg_preconditioner: object = field(default=None, init=False, repr=False)
    _amg_method: str | None = field(default=None, init=False, repr=False)

    def _filtered_values(self, gaps) -> np.ndarray:
        values = validate_gap_array(
            gaps,
            geometry="Cartesian",
            require_square=True,
            minimum_shape=(2, 2),
        )
        if values.shape != (self.n, self.n):
            raise InvalidGapError(
                f"Prepared topology has shape {(self.n, self.n)}, got {values.shape}."
            )
        if not np.array_equal(values > 0.0, self.original_open_mask):
            raise InvalidGapError(
                "Gap open/closed topology changed; prepare a new Cartesian problem."
            )
        return np.where(self.spanning_mask, values, 0.0)

    def assemble(self, gaps):
        """Update coefficients/RHS while reusing mappings and CSR structure."""
        values = self._filtered_values(gaps)
        data = np.empty(self.indices.size, dtype=np.float64)
        rhs = np.zeros(self.dof_to_grid.size, dtype=np.float64)
        _fill_active_matrix_csr_into(
            self.n,
            values,
            self.grid_to_dof,
            self.dof_to_grid,
            self.indptr,
            self.indices,
            data,
            rhs,
            self.p_west,
            self.p_east,
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
        flux_x, flux_y = _calculate_face_fluxes_numba(
            values, pressure, self.p_west, self.p_east
        )
        flux = _cell_flux_from_faces_numba(values, flux_x, flux_y)
        return values, pressure, flux, result

    def solve(
        self,
        gaps,
        solver="auto",
        rtol=DEFAULT_RTOL,
        reuse_preconditioner=False,
    ):
        values, pressure, flux, _ = self.solve_with_diagnostics(
            gaps,
            solver=solver,
            rtol=rtol,
            reuse_preconditioner=reuse_preconditioner,
        )
        return values, pressure, flux


def prepare_fluid_problem(gaps, *, p_west=0.0, p_east=1.0):
    """Prepare reusable Cartesian connectivity, mappings, and CSR topology."""
    values = validate_gap_array(
        gaps,
        geometry="Cartesian",
        require_square=True,
        minimum_shape=(2, 2),
    )
    if not np.isfinite(p_west) or not np.isfinite(p_east):
        raise ValueError("Reservoir pressures must be finite.")
    filtered = connectivity_analysis(values)
    if filtered is None:
        return None
    grid_to_dof, dof_to_grid = build_active_mapping(filtered)
    counts = _count_active_matrix_entries(
        values.shape[0], grid_to_dof, dof_to_grid
    )
    indptr = indptr_from_counts(counts)
    indices, _, _ = _fill_active_matrix_csr(
        values.shape[0],
        filtered,
        grid_to_dof,
        dof_to_grid,
        indptr,
        p_west,
        p_east,
    )
    return PreparedCartesianProblem(
        n=values.shape[0],
        original_open_mask=(values > 0.0).copy(),
        spanning_mask=(filtered > 0.0).copy(),
        grid_to_dof=grid_to_dof,
        dof_to_grid=dof_to_grid,
        indptr=indptr,
        indices=indices,
        p_west=p_west,
        p_east=p_east,
    )

def solve_fluid_problem(
    gaps,
    solver="auto",
    rtol=None,
    save_matrix=False,
    save_matrix_type="coo",
    p_west=0.0,
    p_east=1.0,
):
    logger.info("Starting fluid solver.")

    gaps = validate_gap_array(
        gaps,
        geometry="Cartesian",
        require_square=True,
        minimum_shape=(2, 2),
    )
    n = gaps.shape[0]

    logger.info("Checking connectivity.")

    start = time.time()
    gaps_original = connectivity_analysis(gaps)
    logger.info("Connectivity analysis: CPU time  = {1:.3f} sec".format(n, time.time() - start))

    if gaps_original is None:
        logger.info("No percolating path found. Returning None.")
        return None, None, None

    logger.info("Solving diffusion problem.")
    start_time = time.time()
    p = solve_diffusion(
        n,
        gaps_original,
        solver=solver,
        rtol=rtol,
        save_matrix=save_matrix,
        save_matrix_type=save_matrix_type,
        p_west=p_west,
        p_east=p_east,
    )
    logger.info("Fluid solver: CPU time for n = {0:d}: {1:.3f} sec".format(n, time.time() - start_time))

    logger.info("Fluid solver finished.")
    logger.info("Calculating flux.")

    flux_x, flux_y = _calculate_face_fluxes_numba(
        gaps_original, p, p_west, p_east
    )
    flux = _cell_flux_from_faces_numba(gaps_original, flux_x, flux_y)

    logger.info("finished.")

    return gaps_original, p, flux

def get_preconditioner(A, method="amg-rs"):
    """Backward-compatible wrapper around the shared AMG implementation."""
    aliases = {
        "amg.rs": "amg-rs",
        "amg-sa": "amg-smooth_aggregation",
    }
    return build_amg_preconditioner(A, aliases.get(method, method))

# Total flux calculation
def compute_total_flux(filtered_gaps, flux, N0):
    """Integrate exact boundary-face flux stored in the boundary cells."""
    filtered_gaps = np.asarray(filtered_gaps)
    flux = np.asarray(flux)
    if filtered_gaps.ndim != 2 or flux.shape != filtered_gaps.shape + (2,):
        raise ValueError(
            "Expected gaps with shape (n, n) and flux with shape (n, n, 2)."
        )
    if filtered_gaps.shape != (N0, N0):
        raise ValueError(
            f"N0={N0} does not match Cartesian gap shape {filtered_gaps.shape}."
        )
    if N0 < 1:
        raise ValueError("N0 must be positive.")
    dy = 1.0 / N0

    flux_inlet = 0.0
    active_inlet_cells = 0
    for j in range(N0):
        if not np.isnan(flux[0, j, 0]) and filtered_gaps[0, j] > 0:
            flux_inlet += flux[0, j, 0] * dy
            active_inlet_cells += 1

    flux_outlet = 0.0
    active_outlet_cells = 0
    for j in range(N0):
        if not np.isnan(flux[N0-1, j, 0]) and filtered_gaps[N0-1, j] > 0:
            flux_outlet += flux[N0-1, j, 0] * dy
            active_outlet_cells += 1

    Q_total = 0.5 * (abs(flux_inlet) + abs(flux_outlet))
    flux_conservation_error = abs(flux_inlet - flux_outlet) / max(Q_total, 1e-15)

    logger.info("> Flux computation <")
    logger.info(f"Inlet flux (x=0):     {flux_inlet:.6e} [Active cells: {active_inlet_cells}]")
    logger.info(f"Outlet flux (x=1):    {flux_outlet:.6e} [Active cells: {active_outlet_cells}]")
    logger.info(f"Total average flux (Q_total): {Q_total:.6e}")
    logger.info(f"Conservation error:   {flux_conservation_error:.2e} ({flux_conservation_error*100:.2f}%)")

    return Q_total, flux_conservation_error


def warmup_numba_functions():
    """Explicitly compile Cartesian Numba kernels on tiny arrays."""
    logger.info("Warming up Cartesian Numba kernels.")
    n = 3
    gaps = np.full((n, n), 0.1, dtype=np.float64)
    create_diffusion_matrix(n, gaps)

    blocked = gaps.copy()
    blocked[1, 1] = 0.0
    create_active_diffusion_matrix(blocked)
    _threshold_numba(gaps, 0.05)

    pressure = np.repeat(
        ((np.arange(n, dtype=np.float64) + 0.5) / n)[:, None],
        n,
        axis=1,
    )
    flux_x, flux_y = _calculate_face_fluxes_numba(gaps, pressure)
    _cell_flux_from_faces_numba(gaps, flux_x, flux_y)
    logger.info("Cartesian Numba kernels warmed up.")


_warmup_numba_functions = warmup_numba_functions
