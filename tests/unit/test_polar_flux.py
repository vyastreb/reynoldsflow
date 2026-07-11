"""Conservative polar matrix and face-flux regressions."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse.linalg import spsolve

from reynoldsflow.transport_polar import (
    THETA_BC_PERIODIC,
    THETA_BC_SYMMETRY,
    _calculate_face_fluxes_polar,
    _cell_flux_from_faces_polar,
    compute_total_flux_polar,
    create_diffusion_matrix_polar,
)


pytestmark = pytest.mark.unit


def _solve_constant_gap(
    n_r: int,
    n_theta: int,
    theta_extent: float,
    theta_bc_code: int,
):
    gaps = np.ones((n_r, n_theta), dtype=np.float64)
    matrix, rhs, dr, dtheta = create_diffusion_matrix_polar(
        gaps,
        r_inner=1.0,
        r_outer=2.0,
        theta_extent=theta_extent,
        theta_bc_code=theta_bc_code,
        p_inner=1.0,
        p_outer=0.0,
    )
    solution = spsolve(matrix.tocsc(), rhs).reshape(gaps.shape)
    flux_r, flux_theta = _calculate_face_fluxes_polar(
        gaps, solution, 1.0, dr, dtheta, theta_bc_code
    )
    flux = _cell_flux_from_faces_polar(
        gaps, flux_r, flux_theta, 1.0, dr
    )
    return gaps, matrix.tocsr(), solution, flux_r, flux_theta, flux, dr, dtheta


def test_row_scaled_polar_matrix_is_symmetric():
    gaps = np.ones((18, 36), dtype=np.float64)
    radial_scale = np.linspace(0.5, 1.0, gaps.shape[0])
    gaps *= radial_scale[:, None]
    matrix, _, _, _ = create_diffusion_matrix_polar(
        gaps,
        r_inner=1.0,
        r_outer=2.0,
        theta_extent=2.0 * np.pi,
        theta_bc_code=THETA_BC_PERIODIC,
        p_inner=1.0,
        p_outer=0.0,
    )

    difference = matrix.tocsr() - matrix.tocsr().T
    if difference.nnz:
        assert np.max(np.abs(difference.data)) < 1e-13


def test_full_annulus_has_conservative_radial_flux():
    gaps, _, _, flux_r, flux_theta, flux, _, dtheta = _solve_constant_gap(
        32, 64, 2.0 * np.pi, THETA_BC_PERIODIC
    )
    total_flux, conservation_error = compute_total_flux_polar(
        gaps, flux, 1.0, 2.0, dtheta
    )

    expected = 2.0 * np.pi / np.log(2.0)
    face_totals = np.sum(flux_r, axis=1) * dtheta * (
        1.0 + (np.arange(flux_r.shape[0]) + 0.5) / (gaps.shape[0] - 1)
    )
    assert_allclose(face_totals, face_totals[0], rtol=2e-12, atol=2e-12)
    assert_allclose(flux_theta, 0.0, rtol=0.0, atol=1e-12)
    assert_allclose(total_flux, expected, rtol=5e-4)
    assert conservation_error < 2e-12


def test_symmetry_sector_uses_endpoint_half_weights():
    theta_extent = 0.5 * np.pi
    gaps, _, _, _, _, flux, _, dtheta = _solve_constant_gap(
        32, 33, theta_extent, THETA_BC_SYMMETRY
    )
    total_flux, conservation_error = compute_total_flux_polar(
        gaps,
        flux,
        1.0,
        2.0,
        dtheta,
        theta_bc="symmetry",
    )

    assert_allclose(total_flux, theta_extent / np.log(2.0), rtol=5e-4)
    assert conservation_error < 2e-12


def test_single_angular_sample_uses_requested_extent():
    theta_extent = 1.25
    gaps, _, _, _, _, flux, _, dtheta = _solve_constant_gap(
        32, 1, theta_extent, THETA_BC_PERIODIC
    )
    total_flux, conservation_error = compute_total_flux_polar(
        gaps, flux, 1.0, 2.0, dtheta
    )

    assert_allclose(dtheta, theta_extent, rtol=0.0, atol=0.0)
    assert_allclose(total_flux, theta_extent / np.log(2.0), rtol=5e-4)
    assert conservation_error < 2e-12
