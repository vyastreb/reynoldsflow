"""Input validation shared by ReynoldsFlow geometry modules."""

from __future__ import annotations

import numpy as np

from ._exceptions import InvalidGapError


def validate_gap_array(
    gaps,
    *,
    geometry: str,
    require_square: bool = False,
    minimum_shape: tuple[int, int] = (2, 1),
) -> np.ndarray:
    """Return a finite float64 2D gap array satisfying geometry constraints."""
    try:
        array = np.asarray(gaps, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise InvalidGapError(
            f"{geometry} gap field must contain numeric values."
        ) from exc
    if array.ndim != 2:
        raise InvalidGapError(
            f"{geometry} gap field must be a 2D array; got shape {array.shape}."
        )
    if array.shape[0] < minimum_shape[0] or array.shape[1] < minimum_shape[1]:
        raise InvalidGapError(
            f"{geometry} gap field must have shape at least {minimum_shape}; "
            f"got {array.shape}."
        )
    if require_square and array.shape[0] != array.shape[1]:
        raise InvalidGapError(
            f"{geometry} solver currently requires a square gap field; "
            f"got {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise InvalidGapError(f"{geometry} gap field must contain only finite values.")
    return array
