"""Characterization tests for Cartesian flux reconstruction."""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.sparse.linalg import spsolve

from reynoldsflow.transport import (
    _calculate_face_fluxes_numba,
    _cell_flux_from_faces_numba,
    compute_total_flux,
    create_diffusion_matrix,
    solve_fluid_problem,
)


pytestmark = pytest.mark.unit


def test_unit_gap_has_unit_total_flux():
    """A unit gap under a unit pressure drop should have unit conductance."""
    n = 24
    gaps = np.ones((n, n), dtype=np.float64)

    matrix, rhs = create_diffusion_matrix(n, gaps)
    pressure = spsolve(matrix.tocsc(), rhs).reshape(gaps.shape)

    flux_x, flux_y = _calculate_face_fluxes_numba(gaps, pressure)
    flux = _cell_flux_from_faces_numba(gaps, flux_x, flux_y)
    total_flux, conservation_error = compute_total_flux(gaps, flux, n)

    assert_allclose(flux_x, -1.0, rtol=1e-12, atol=1e-12)
    assert_allclose(flux_y, 0.0, rtol=0.0, atol=1e-12)
    assert_allclose(total_flux, 1.0, rtol=1e-12, atol=1e-12)
    assert conservation_error < 1e-12


def test_one_dimensional_variable_gap_has_constant_face_flux():
    n = 32
    x = (np.arange(n, dtype=np.float64) + 0.5) / n
    gap_line = 0.5 + 0.5 * x
    gaps = np.repeat(gap_line[:, None], n, axis=1)

    matrix, rhs = create_diffusion_matrix(n, gaps)
    pressure = spsolve(matrix.tocsc(), rhs).reshape(gaps.shape)
    flux_x, flux_y = _calculate_face_fluxes_numba(gaps, pressure)

    assert_allclose(flux_x, flux_x[0, 0], rtol=1e-11, atol=1e-12)
    assert_allclose(flux_y, 0.0, rtol=0.0, atol=1e-12)


def test_cartesian_reservoir_pressures_are_configurable():
    n = 20
    gaps = np.ones((n, n), dtype=np.float64)
    filtered, pressure, flux = solve_fluid_problem(
        gaps,
        solver="scipy-spsolve",
        p_west=2.0,
        p_east=5.0,
    )
    total_flux, conservation_error = compute_total_flux(
        filtered, flux, n
    )

    expected_profile = 2.0 + 3.0 * (
        np.arange(n, dtype=np.float64) + 0.5
    ) / n
    assert_allclose(pressure[:, 0], expected_profile, rtol=1e-12, atol=1e-12)
    assert_allclose(flux[:, :, 0], -3.0, rtol=1e-12, atol=1e-12)
    assert_allclose(total_flux, 3.0, rtol=1e-12, atol=1e-12)
    assert conservation_error < 1e-12
