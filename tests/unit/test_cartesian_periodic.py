"""Cartesian representative-cell pressure-gradient tests."""

import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from reynoldsflow._connectivity import label_periodic_components
from reynoldsflow.transport import (
    _calculate_periodic_face_fluxes_numba,
    compute_total_flux,
    prepare_fluid_problem,
    solve_fluid_problem,
)


pytestmark = pytest.mark.unit


@pytest.mark.parametrize("gradient", [0.0, 2.5, -1.75])
def test_open_periodic_cell_has_uniform_gradient_driven_flux(gradient):
    gaps = np.full((12, 12), 0.8)

    filtered, pressure, flux = solve_fluid_problem(
        gaps,
        solver="scipy-spsolve",
        boundary_mode="periodic",
        pressure_gradient=gradient,
    )

    assert_array_equal(filtered, gaps)
    assert_allclose(pressure, 0.0, atol=1e-14)
    assert_allclose(flux[..., 0], 0.8**3 * gradient, atol=1e-13)
    assert_allclose(flux[..., 1], 0.0, atol=1e-14)
    total_flux, conservation_error = compute_total_flux(
        filtered, flux, gaps.shape[0], boundary_mode="periodic"
    )
    assert_allclose(total_flux, abs(0.8**3 * gradient), atol=1e-13)
    assert conservation_error < 1e-13


def test_heterogeneous_periodic_flux_is_conservative_and_matrix_is_symmetric():
    rng = np.random.default_rng(451)
    n = 15
    gaps = 0.25 + rng.random((n, n))
    gaps[4:8, 6:10] = 0.0
    gradient = 1.4
    prepared = prepare_fluid_problem(
        gaps, boundary_mode="periodic", pressure_gradient=gradient
    )

    matrix, _, _ = prepared.assemble(gaps)
    asymmetry = matrix - matrix.T
    assert asymmetry.nnz == 0 or np.max(np.abs(asymmetry.data)) < 1e-14

    filtered, pressure, _ = prepared.solve(gaps, solver="scipy-spsolve")
    flux_x, flux_y = _calculate_periodic_face_fluxes_numba(
        filtered, pressure, gradient
    )
    dx = 1.0 / n
    divergence = (flux_x[1:] - flux_x[:-1]) / dx
    divergence += (flux_y - np.roll(flux_y, 1, axis=1)) / dx

    assert_allclose(pressure[filtered > 0.0].mean(), 0.0, atol=1e-14)
    assert np.max(np.abs(divergence[filtered > 0.0])) < 2e-11
    assert np.ptp(flux_x[:-1].sum(axis=1) / n) < 2e-13


def test_periodic_pockets_are_retained_gauged_and_carry_no_flux():
    n = 10
    gaps = np.zeros((n, n))
    gaps[:, 2] = 1.0
    gaps[0, 7] = 0.6
    gaps[-1, 7] = 0.6

    filtered, pressure, flux = solve_fluid_problem(
        gaps,
        solver="scipy-spsolve",
        boundary_mode="periodic",
        pressure_gradient=1.0,
    )
    labels, winding = label_periodic_components(gaps)
    pocket_label = int(labels[0, 7])

    assert_array_equal(filtered, gaps)
    assert not winding[pocket_label]
    for component in range(1, int(labels.max()) + 1):
        assert_allclose(pressure[labels == component].mean(), 0.0, atol=1e-14)
    assert_allclose(flux[labels == pocket_label], 0.0, atol=1e-13)


def test_seam_crossing_pocket_is_not_mistaken_for_x_winding():
    gaps = np.zeros((9, 9))
    gaps[0, 4] = 1.0
    gaps[-1, 4] = 1.0

    assert prepare_fluid_problem(gaps, boundary_mode="periodic") is None
    assert solve_fluid_problem(gaps, boundary_mode="periodic") == (
        None,
        None,
        None,
    )


def test_prepared_periodic_problem_matches_one_shot_solve():
    gaps = np.ones((11, 11))
    gaps[3:7, 4:6] = 0.0
    kwargs = dict(
        solver="scipy-spsolve",
        boundary_mode="periodic",
        pressure_gradient=-0.7,
    )

    expected = solve_fluid_problem(gaps, **kwargs)
    prepared = prepare_fluid_problem(
        gaps,
        boundary_mode=kwargs["boundary_mode"],
        pressure_gradient=kwargs["pressure_gradient"],
    )
    actual = prepared.solve(gaps, solver=kwargs["solver"])

    for expected_value, actual_value in zip(expected, actual):
        assert_allclose(actual_value, expected_value, rtol=1e-13, atol=1e-13)


def test_periodic_boundary_parameters_are_validated():
    gaps = np.ones((5, 5))
    with pytest.raises(ValueError, match="boundary_mode"):
        solve_fluid_problem(gaps, boundary_mode="unknown")
    with pytest.raises(ValueError, match="Pressure gradient must be finite"):
        solve_fluid_problem(
            gaps, boundary_mode="periodic", pressure_gradient=np.nan
        )
    with pytest.raises(ValueError, match="p_west and p_east"):
        solve_fluid_problem(
            gaps, boundary_mode="periodic", p_west=2.0
        )
    with pytest.raises(ValueError, match="only available"):
        solve_fluid_problem(gaps, pressure_gradient=2.0)
