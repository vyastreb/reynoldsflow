"""Public gap and geometry validation regressions."""

import numpy as np
import pytest

from reynoldsflow._exceptions import InvalidGapError
from reynoldsflow.transport import (
    compute_total_flux,
    create_diffusion_matrix,
    solve_diffusion,
    solve_fluid_problem,
)
from reynoldsflow.transport_polar import (
    compute_total_flux_polar,
    solve_fluid_problem_polar,
)


pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "gaps",
    [
        np.ones(5),
        np.ones((4, 5)),
        np.ones((1, 1)),
        np.array([[1.0, np.nan], [1.0, 1.0]]),
        [["not", "numeric"], ["values", "here"]],
    ],
)
def test_cartesian_public_api_rejects_invalid_gap_arrays(gaps):
    with pytest.raises(InvalidGapError):
        solve_fluid_problem(gaps, solver="scipy-spsolve")


def test_cartesian_matrix_rejects_mismatched_n():
    with pytest.raises(InvalidGapError, match="does not match"):
        create_diffusion_matrix(5, np.ones((4, 4)))


def test_direct_diffusion_rejects_no_positive_cells():
    with pytest.raises(InvalidGapError, match="no positive cells"):
        solve_diffusion(4, np.zeros((4, 4)), solver="scipy-spsolve")


def test_public_no_percolation_remains_a_normal_none_result():
    result = solve_fluid_problem(
        np.zeros((4, 4)), solver="scipy-spsolve"
    )
    assert result == (None, None, None)


def test_polar_public_api_accepts_numeric_nested_lists():
    gaps = [[1.0] * 6 for _ in range(5)]
    filtered, pressure, flux, dr, dtheta = solve_fluid_problem_polar(
        gaps,
        1.0,
        2.0,
        solver="scipy-spsolve",
        dilation_iterations=0,
    )

    assert filtered.shape == (5, 6)
    assert pressure.shape == (5, 6)
    assert flux.shape == (5, 6, 2)
    assert dr > 0.0
    assert dtheta > 0.0


def test_polar_dilation_iterations_must_be_nonnegative_integer():
    with pytest.raises(ValueError, match="dilation_iterations"):
        solve_fluid_problem_polar(
            np.ones((5, 6)),
            1.0,
            2.0,
            solver="scipy-spsolve",
            dilation_iterations=-1,
        )


def test_polar_default_does_not_dilate_geometry():
    gaps = np.ones((8, 12), dtype=np.float64)
    gaps[3:6, 4:7] = 0.0

    default = solve_fluid_problem_polar(
        gaps, 1.0, 2.0, solver="scipy-spsolve"
    )
    explicit_zero = solve_fluid_problem_polar(
        gaps,
        1.0,
        2.0,
        solver="scipy-spsolve",
        dilation_iterations=0,
    )

    for default_value, explicit_value in zip(default, explicit_zero):
        assert np.allclose(default_value, explicit_value, equal_nan=True)


def test_flux_integrators_reject_inconsistent_shapes():
    with pytest.raises(ValueError, match="Expected gaps"):
        compute_total_flux(np.ones((4, 4)), np.ones((4, 4)), 4)

    with pytest.raises(ValueError, match="Expected gaps"):
        compute_total_flux_polar(
            np.ones((4, 5)), np.ones((4, 5)), 1.0, 2.0, 0.1
        )


def test_cartesian_reservoir_pressures_must_be_finite():
    with pytest.raises(ValueError, match="pressures must be finite"):
        solve_fluid_problem(
            np.ones((4, 4)),
            solver="scipy-spsolve",
            p_west=np.nan,
        )
