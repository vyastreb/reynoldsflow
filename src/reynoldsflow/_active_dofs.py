"""Compact active-degree-of-freedom mappings for grid-based solvers."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit
def _fill_active_mapping(gaps: np.ndarray, active_count: int):
    flat_gaps = gaps.ravel()
    grid_to_dof = np.full(flat_gaps.size, -1, dtype=np.int32)
    dof_to_grid = np.empty(active_count, dtype=np.int32)
    dof = 0
    for grid_index in range(flat_gaps.size):
        if flat_gaps[grid_index] > 0.0:
            grid_to_dof[grid_index] = dof
            dof_to_grid[dof] = grid_index
            dof += 1
    return grid_to_dof, dof_to_grid


def build_active_mapping(gaps: np.ndarray):
    """Return flat int32 grid-to-DOF and DOF-to-grid maps."""
    active_count = int(np.count_nonzero(gaps > 0.0))
    if active_count == 0:
        raise ValueError("Cannot build a linear system without active cells.")
    int32_max = np.iinfo(np.int32).max
    if gaps.size > int32_max or active_count > int32_max:
        raise OverflowError(
            "Active mapping currently requires grid and DOF counts to fit int32."
        )
    return _fill_active_mapping(gaps, active_count)


def reconstruct_full_solution(
    shape: tuple[int, int],
    solution: np.ndarray,
    dof_to_grid: np.ndarray | None,
) -> np.ndarray:
    """Reconstruct a full grid, or reshape an already full-grid solution."""
    solution = np.asarray(solution, dtype=np.float64).reshape(-1)
    if dof_to_grid is None:
        return solution.copy().reshape(shape)
    full = np.zeros(int(np.prod(shape)), dtype=np.float64)
    full[dof_to_grid] = solution
    return full.reshape(shape)
