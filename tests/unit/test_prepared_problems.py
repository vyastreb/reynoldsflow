"""Prepared topology must match fresh solves and reject stale topology."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from reynoldsflow._exceptions import InvalidGapError
from reynoldsflow.transport import prepare_fluid_problem, solve_fluid_problem
from reynoldsflow.transport_polar import (
    prepare_fluid_problem_polar,
    solve_fluid_problem_polar,
)


pytestmark = pytest.mark.unit


def _assert_result_arrays_match(prepared_result, fresh_result):
    assert len(prepared_result) == len(fresh_result)
    for prepared_value, fresh_value in zip(prepared_result, fresh_result):
        assert_allclose(
            prepared_value,
            fresh_value,
            rtol=3e-12,
            atol=3e-12,
            equal_nan=True,
        )


def test_prepared_cartesian_sequence_matches_fresh_solve():
    n = 24
    coordinate = (np.arange(n, dtype=np.float64) + 0.5) / n
    x, y = np.meshgrid(coordinate, coordinate, indexing="ij")
    gaps = np.ones((n, n), dtype=np.float64)
    gaps[(x - 0.5) ** 2 + (y - 0.5) ** 2 < 0.16**2] = 0.0
    prepared = prepare_fluid_problem(gaps)

    varied = gaps * (0.6 + 0.4 * x)
    prepared_result = prepared.solve(varied, solver="scipy-spsolve")
    fresh_result = solve_fluid_problem(varied, solver="scipy-spsolve")

    _assert_result_arrays_match(prepared_result, fresh_result)
    _, _, _, diagnostics = prepared.solve_with_diagnostics(
        varied, solver="scipy-spsolve"
    )
    assert diagnostics.relative_residual < 1e-12


def test_prepared_cartesian_rejects_topology_change():
    gaps = np.ones((8, 8), dtype=np.float64)
    prepared = prepare_fluid_problem(gaps)
    changed = gaps.copy()
    changed[3, 3] = 0.0

    with pytest.raises(InvalidGapError, match="topology changed"):
        prepared.solve(changed, solver="scipy-spsolve")


def test_prepared_cartesian_can_reuse_amg_hierarchy():
    n = 20
    gaps = np.ones((n, n), dtype=np.float64)
    prepared = prepare_fluid_problem(gaps)

    first = prepared.solve_with_diagnostics(
        gaps,
        solver="scipy.amg-rs",
        rtol=1e-10,
        reuse_preconditioner=True,
    )
    cached_id = id(prepared._amg_preconditioner)
    varied = gaps * np.linspace(0.9, 1.1, n)[:, None]
    second = prepared.solve_with_diagnostics(
        varied,
        solver="scipy.amg-rs",
        rtol=1e-10,
        reuse_preconditioner=True,
    )
    fresh = solve_fluid_problem(
        varied, solver="scipy-spsolve"
    )

    assert id(prepared._amg_preconditioner) == cached_id
    for prepared_value, fresh_value in zip(second[:3], fresh):
        assert_allclose(
            prepared_value,
            fresh_value,
            rtol=2e-9,
            atol=5e-9,
            equal_nan=True,
        )
    assert first[-1].converged
    assert second[-1].converged
    prepared.clear_preconditioner()
    assert prepared._amg_preconditioner is None


def test_prepared_cartesian_returns_none_without_percolation():
    gaps = np.zeros((8, 8), dtype=np.float64)
    gaps[:4, 2] = 1.0
    assert prepare_fluid_problem(gaps) is None


def test_prepared_polar_sequence_matches_fresh_solve():
    n_r, n_theta = 18, 36
    gaps = np.ones((n_r, n_theta), dtype=np.float64)
    gaps[5:13, 12:17] = 0.0
    prepared = prepare_fluid_problem_polar(gaps, 1.0, 2.0)

    radial_scale = np.linspace(0.6, 1.0, n_r)[:, None]
    varied = gaps * radial_scale
    prepared_result = prepared.solve(varied, solver="scipy-spsolve")
    fresh_result = solve_fluid_problem_polar(
        varied,
        1.0,
        2.0,
        solver="scipy-spsolve",
        dilation_iterations=0,
    )

    _assert_result_arrays_match(prepared_result, fresh_result)
    *_, diagnostics = prepared.solve_with_diagnostics(
        varied, solver="scipy-spsolve"
    )
    assert diagnostics.relative_residual < 1e-12


def test_prepared_polar_rejects_topology_change():
    gaps = np.ones((8, 12), dtype=np.float64)
    prepared = prepare_fluid_problem_polar(gaps, 1.0, 2.0)
    changed = gaps.copy()
    changed[3, 3] = 0.0

    with pytest.raises(InvalidGapError, match="topology changed"):
        prepared.solve(changed, solver="scipy-spsolve")


def test_prepared_polar_can_reuse_amg_hierarchy():
    gaps = np.ones((14, 28), dtype=np.float64)
    prepared = prepare_fluid_problem_polar(gaps, 1.0, 2.0)
    prepared.solve_with_diagnostics(
        gaps,
        solver="scipy.amg-rs",
        rtol=1e-10,
        reuse_preconditioner=True,
    )
    cached_id = id(prepared._amg_preconditioner)
    varied = gaps * np.linspace(0.95, 1.05, gaps.shape[0])[:, None]
    prepared_result = prepared.solve_with_diagnostics(
        varied,
        solver="scipy.amg-rs",
        rtol=1e-10,
        reuse_preconditioner=True,
    )
    fresh_result = solve_fluid_problem_polar(
        varied, 1.0, 2.0, solver="scipy-spsolve"
    )

    assert id(prepared._amg_preconditioner) == cached_id
    for prepared_value, fresh_value in zip(prepared_result[:5], fresh_result):
        assert_allclose(
            prepared_value,
            fresh_value,
            rtol=2e-9,
            atol=2e-10,
            equal_nan=True,
        )
    assert prepared_result[-1].converged
