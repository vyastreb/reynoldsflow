"""Connectivity behavior and known-defect characterization tests."""

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from reynoldsflow.transport import connectivity_analysis
from reynoldsflow.transport_polar import (
    THETA_BC_PERIODIC,
    connectivity_analysis_polar,
)


pytestmark = pytest.mark.unit


def test_all_independent_spanning_channels_are_retained():
    n = 12
    gaps = np.zeros((n, n), dtype=np.float64)
    gaps[:, 2] = 1.0
    gaps[:, 8] = 0.5

    filtered = connectivity_analysis(gaps)

    assert_array_equal(filtered, gaps)


def test_all_independent_polar_channels_are_retained():
    gaps = np.zeros((12, 12), dtype=np.float64)
    gaps[:, 2] = 1.0
    gaps[:, 8] = 0.5

    filtered = connectivity_analysis_polar(gaps, THETA_BC_PERIODIC)

    assert_array_equal(filtered, gaps)


def test_channel_can_cross_the_periodic_seam():
    n = 12
    middle = n // 2
    gaps = np.zeros((n, n), dtype=np.float64)
    gaps[: middle + 1, 0] = 1.0
    gaps[middle:, -1] = 1.0

    filtered = connectivity_analysis(gaps)

    assert_array_equal(filtered, gaps)


def test_nonpercolating_channel_returns_none():
    gaps = np.zeros((10, 10), dtype=np.float64)
    gaps[:5, 4] = 1.0

    assert connectivity_analysis(gaps) is None
