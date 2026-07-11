"""Tests for benchmark reporting rather than performance thresholds."""

import pytest

from benchmarks.benchmark_solver import _summarize_runs


pytestmark = pytest.mark.unit


def test_run_summary_separates_cold_from_steady_state():
    runs = [
        {"timings_s": {"assembly": 2.0, "linear_solve": 8.0}},
        {"timings_s": {"assembly": 1.0, "linear_solve": 2.0}},
        {"timings_s": {"assembly": 1.0, "linear_solve": 4.0}},
    ]

    summary = _summarize_runs(runs)

    assert summary["cold_total_s"] == 10.0
    assert summary["cold_linear_solve_s"] == 8.0
    assert summary["steady_runs"] == 2
    assert summary["steady_total_median_s"] == 4.0
    assert summary["steady_linear_solve_median_s"] == 3.0


def test_run_summary_treats_every_recorded_run_as_warm_after_warmup():
    runs = [
        {"timings_s": {"assembly": 1.0, "linear_solve": 2.0}},
        {"timings_s": {"assembly": 1.0, "linear_solve": 4.0}},
    ]

    summary = _summarize_runs(runs, includes_cold_start=False)

    assert "cold_total_s" not in summary
    assert summary["steady_runs"] == 2
    assert summary["steady_total_median_s"] == 4.0
