"""Tests for benchmark reporting rather than performance thresholds."""

import pytest
import numpy as np

from benchmarks.cases import build_cartesian_case
from benchmarks.benchmark_solver import _environment, _summarize_runs
from benchmarks.benchmark_suite import _text_tail


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


def test_timeout_output_bytes_are_json_safe_text():
    assert _text_tail(b"native stderr", limit=6) == "stderr"
    assert _text_tail(None) == ""


def test_historical_rough_case_is_deterministic_and_contains_contact():
    first = build_cartesian_case("historical-rough-contact", 32, seed=23_349)
    second = build_cartesian_case("historical-rough-contact", 32, seed=23_349)

    assert np.array_equal(first, second)
    assert np.any(first == 0.0)
    assert np.any(first > 0.0)


def test_benchmark_environment_records_hardware_and_blas():
    environment = _environment()

    assert environment["cpu_model"]
    assert "name" in environment["numpy_blas"]
    assert "pypardiso" in environment["packages"]
    assert "petsc4py" in environment["packages"]
    assert "scikit-sparse" in environment["packages"]
