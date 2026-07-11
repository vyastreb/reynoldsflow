"""Compact active-DOF matrices must match full-grid reference systems."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse.linalg import spsolve

from reynoldsflow._active_dofs import reconstruct_full_solution
from reynoldsflow.transport import (
    create_active_diffusion_matrix,
    create_diffusion_matrix,
)
from reynoldsflow.transport_polar import (
    THETA_BC_PERIODIC,
    create_active_diffusion_matrix_polar,
    create_diffusion_matrix_polar,
)


pytestmark = pytest.mark.unit


def test_compact_cartesian_system_matches_full_grid_reference():
    n = 14
    coordinates = (np.arange(n, dtype=np.float64) + 0.5) / n
    x, y = np.meshgrid(coordinates, coordinates, indexing="ij")
    gaps = 0.6 + 0.3 * x
    gaps[(x - 0.5) ** 2 + (y - 0.5) ** 2 < 0.12**2] = 0.0

    full_matrix, full_rhs = create_diffusion_matrix(n, gaps)
    full_pressure = spsolve(full_matrix.tocsc(), full_rhs).reshape(gaps.shape)

    compact_matrix, compact_rhs, dof_to_grid = create_active_diffusion_matrix(
        gaps
    )
    compact_solution = spsolve(compact_matrix.tocsc(), compact_rhs)
    compact_pressure = reconstruct_full_solution(
        gaps.shape, compact_solution, dof_to_grid
    )

    assert compact_matrix.shape[0] == np.count_nonzero(gaps > 0.0)
    assert_allclose(compact_pressure, full_pressure, rtol=2e-13, atol=2e-13)


def test_compact_polar_system_matches_full_grid_reference():
    n_r = 16
    n_theta = 32
    gaps = np.ones((n_r, n_theta), dtype=np.float64)
    gaps[4:12, 10:16] = 0.0

    full_matrix, full_rhs, _, _ = create_diffusion_matrix_polar(
        gaps,
        1.0,
        2.0,
        2.0 * np.pi,
        THETA_BC_PERIODIC,
        1.0,
        0.0,
    )
    full_pressure = spsolve(full_matrix.tocsc(), full_rhs).reshape(gaps.shape)

    compact_matrix, compact_rhs, _, _, dof_to_grid = (
        create_active_diffusion_matrix_polar(
            gaps,
            1.0,
            2.0,
            2.0 * np.pi,
            THETA_BC_PERIODIC,
            1.0,
            0.0,
        )
    )
    compact_solution = spsolve(compact_matrix.tocsc(), compact_rhs)
    compact_pressure = reconstruct_full_solution(
        gaps.shape, compact_solution, dof_to_grid
    )

    assert compact_matrix.shape[0] == np.count_nonzero(gaps > 0.0)
    assert_allclose(compact_pressure, full_pressure, rtol=3e-13, atol=3e-13)
    difference = compact_matrix.tocsr() - compact_matrix.tocsr().T
    if difference.nnz:
        assert np.max(np.abs(difference.data)) < 1e-13
