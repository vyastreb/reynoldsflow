"""Pytest configuration for safe opt-in execution of expensive test groups."""

import pytest


def pytest_addoption(parser):
    group = parser.getgroup("reynoldsflow")
    group.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run tests marked 'slow'",
    )
    group.addoption(
        "--run-backend",
        action="store_true",
        default=False,
        help="run tests requiring optional solver backends",
    )
    group.addoption(
        "--run-benchmark",
        action="store_true",
        default=False,
        help="run tests marked 'benchmark'",
    )


def pytest_collection_modifyitems(config, items):
    gates = (
        ("slow", "--run-slow"),
        ("backend", "--run-backend"),
        ("benchmark", "--run-benchmark"),
    )

    for item in items:
        for marker_name, option_name in gates:
            if item.get_closest_marker(marker_name) and not config.getoption(option_name):
                item.add_marker(
                    pytest.mark.skip(
                        reason=f"requires explicit {option_name} opt-in"
                    )
                )
