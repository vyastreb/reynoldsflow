"""Small deterministic checks for the homogenization study workflow."""

import numpy as np
from numpy.testing import assert_allclose

from homogenization_sims.study import (
    _rms_error,
    _short_wave_cutoff_ratio,
    bruggeman_transmissivity,
    gap_at_contact_fraction,
    realization_seed,
    simulation_tasks,
)


def test_gap_quantile_produces_requested_geometrical_contact():
    surface = np.arange(100, dtype=float).reshape(10, 10)
    gap, threshold, actual = gap_at_contact_fraction(surface, 0.20)

    assert threshold == 19.8
    assert actual == 0.20
    assert np.count_nonzero(gap == 0.0) == 20
    assert np.all(gap >= 0.0)


def test_bruggeman_constant_and_binary_fields():
    assert_allclose(bruggeman_transmissivity(np.full((8, 8), 2.0)), 8.0)
    assert_allclose(
        bruggeman_transmissivity(np.full((8, 8), 2.0), m0=4.0), 1.0
    )

    gap = np.ones((10, 10))
    gap[:2] = 0.0
    # For a 0/1 mixture in 2-D, K_eff = 1 - 2*contact_fraction.
    assert_allclose(bruggeman_transmissivity(gap), 0.6, rtol=1e-12)


def test_task_order_and_paired_seeds_are_stable():
    tasks = simulation_tasks((1, 2), 2, (0.1,))
    assert len(tasks) == 8
    assert tasks[0].mode == "pressure"
    assert tasks[1].mode == "mean-flux"
    assert realization_seed(123, 0) == realization_seed(123, 0)
    assert realization_seed(123, 0) != realization_seed(123, 1)


def test_rms_error_is_population_deviation_from_ensemble_mean():
    assert_allclose(_rms_error((1.0, 2.0, 3.0)), np.sqrt(2.0 / 3.0))


def test_short_wave_cutoff_can_reference_k0_or_k1():
    assert _short_wave_cutoff_ratio(8, 64, "k0") == 64
    assert _short_wave_cutoff_ratio(8, 64, "k1") == 512
