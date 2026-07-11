"""Exact-size CSR structure and matrix-property regressions."""

import numpy as np
import pytest

from reynoldsflow.transport import (
    create_active_diffusion_matrix,
    create_diffusion_matrix,
)
from reynoldsflow.transport_polar import (
    THETA_BC_PERIODIC,
    THETA_BC_SYMMETRY,
    create_active_diffusion_matrix_polar,
    create_diffusion_matrix_polar,
)


pytestmark = pytest.mark.unit


def _assert_valid_symmetric_csr(matrix, maximum_entries_per_row=5):
    assert matrix.format == "csr"
    assert matrix.indptr.dtype == np.int32
    assert matrix.indices.dtype == np.int32
    assert matrix.indptr.size == matrix.shape[0] + 1
    assert matrix.nnz <= maximum_entries_per_row * matrix.shape[0]
    assert np.all(np.diff(matrix.indptr) >= 1)
    assert matrix.has_sorted_indices
    difference = matrix - matrix.T
    difference.eliminate_zeros()
    if difference.nnz:
        scale = max(np.max(np.abs(matrix.data)), 1.0)
        assert np.max(np.abs(difference.data)) < 5e-15 * scale


def test_cartesian_full_and_active_builders_return_exact_csr():
    n = 10
    rng = np.random.default_rng(42)
    gaps = 0.3 + rng.random((n, n))
    gaps[3:7, 4:6] = 0.0

    full_matrix, full_rhs = create_diffusion_matrix(n, gaps)
    active_matrix, active_rhs, dof_to_grid = create_active_diffusion_matrix(gaps)

    _assert_valid_symmetric_csr(full_matrix)
    _assert_valid_symmetric_csr(active_matrix)
    assert full_matrix.shape == (gaps.size, gaps.size)
    assert active_matrix.shape[0] == np.count_nonzero(gaps > 0.0)
    assert active_matrix.shape[0] == dof_to_grid.size
    assert full_rhs.shape == (gaps.size,)
    assert active_rhs.shape == (dof_to_grid.size,)


@pytest.mark.parametrize("theta_bc_code", [THETA_BC_PERIODIC, THETA_BC_SYMMETRY])
def test_polar_full_and_active_builders_return_exact_csr(theta_bc_code):
    n_r, n_theta = 9, 12
    theta_extent = 2.0 * np.pi if theta_bc_code == THETA_BC_PERIODIC else 1.2
    gaps = np.ones((n_r, n_theta), dtype=np.float64)
    gaps[3:6, 4:7] = 0.0

    full_matrix, full_rhs, _, _ = create_diffusion_matrix_polar(
        gaps,
        1.0,
        2.0,
        theta_extent,
        theta_bc_code,
        1.0,
        0.0,
    )
    active_matrix, active_rhs, _, _, dof_to_grid = (
        create_active_diffusion_matrix_polar(
            gaps,
            1.0,
            2.0,
            theta_extent,
            theta_bc_code,
            1.0,
            0.0,
        )
    )

    _assert_valid_symmetric_csr(full_matrix)
    _assert_valid_symmetric_csr(active_matrix)
    assert full_matrix.shape == (gaps.size, gaps.size)
    assert active_matrix.shape[0] == np.count_nonzero(gaps > 0.0)
    assert active_matrix.shape[0] == dof_to_grid.size
    assert full_rhs.shape == (gaps.size,)
    assert active_rhs.shape == (dof_to_grid.size,)
