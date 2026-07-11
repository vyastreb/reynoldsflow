"""Deterministic gap fields used by correctness and performance benchmarks."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter


CARTESIAN_CASES = (
    "constant",
    "linear",
    "circle",
    "periodic-seam",
    "parallel-channels",
    "nonpercolating",
    "rough-contact",
)

POLAR_CASES = (
    "constant",
    "linear",
    "blocked-sector",
)


def build_cartesian_case(
    name: str,
    n: int,
    *,
    seed: int = 23_349,
) -> np.ndarray:
    """Build a square Cartesian gap array without timing-dependent randomness."""
    if n < 4:
        raise ValueError("Cartesian benchmark cases require n >= 4.")
    if name not in CARTESIAN_CASES:
        raise ValueError(
            f"Unknown Cartesian case {name!r}; choose from {CARTESIAN_CASES}."
        )

    coordinates = (np.arange(n, dtype=np.float64) + 0.5) / n
    x, y = np.meshgrid(coordinates, coordinates, indexing="ij")

    if name == "constant":
        return np.ones((n, n), dtype=np.float64)

    if name == "linear":
        return 0.5 + 0.5 * x

    if name == "circle":
        radius = np.sqrt((x - 0.5) ** 2 + (y - 0.5) ** 2)
        return (radius > 0.2).astype(np.float64)

    if name == "periodic-seam":
        gaps = np.zeros((n, n), dtype=np.float64)
        middle = n // 2
        width = max(1, n // 32)
        gaps[: middle + 1, :width] = 1.0
        gaps[middle:, -width:] = 1.0
        return gaps

    if name == "parallel-channels":
        gaps = np.zeros((n, n), dtype=np.float64)
        width = max(1, n // 16)
        first = n // 4
        second = 3 * n // 4
        gaps[:, first : first + width] = 1.0
        gaps[:, second : second + width] = 0.7
        return gaps

    if name == "nonpercolating":
        gaps = np.zeros((n, n), dtype=np.float64)
        width = max(1, n // 16)
        gaps[: n // 2, n // 2 : n // 2 + width] = 1.0
        return gaps

    rng = np.random.default_rng(seed)
    field = gaussian_filter(
        rng.standard_normal((n, n)),
        sigma=max(1.0, n / 48.0),
        mode=("reflect", "wrap"),
    )
    field -= np.mean(field)
    standard_deviation = np.std(field)
    if standard_deviation > 0.0:
        field /= standard_deviation
    return np.clip(field + 0.35, 0.0, None)


def build_polar_case(
    name: str,
    n_r: int,
    n_theta: int,
) -> np.ndarray:
    """Build a deterministic annular gap array."""
    if n_r < 3:
        raise ValueError("Polar benchmark cases require n_r >= 3.")
    if n_theta < 2:
        raise ValueError("Polar benchmark cases require n_theta >= 2.")
    if name not in POLAR_CASES:
        raise ValueError(f"Unknown polar case {name!r}; choose from {POLAR_CASES}.")

    if name == "constant":
        return np.ones((n_r, n_theta), dtype=np.float64)

    radial_coordinate = np.linspace(0.0, 1.0, n_r, dtype=np.float64)
    if name == "linear":
        radial_gap = 0.5 + 0.5 * radial_coordinate
        return np.repeat(radial_gap[:, None], n_theta, axis=1)

    gaps = np.ones((n_r, n_theta), dtype=np.float64)
    start = n_theta // 3
    stop = 2 * n_theta // 3
    radial_stop = max(1, 2 * n_r // 3)
    gaps[:radial_stop, start:stop] = 0.0
    return gaps
