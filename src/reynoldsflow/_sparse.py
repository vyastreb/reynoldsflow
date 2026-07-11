"""Sparse-array construction helpers."""

import numpy as np


def indptr_from_counts(counts: np.ndarray) -> np.ndarray:
    """Build an int32 CSR row pointer with an explicit overflow check."""
    total = int(np.sum(counts, dtype=np.int64))
    if total > np.iinfo(np.int32).max:
        raise OverflowError("Matrix nonzero count exceeds int32 CSR capacity.")
    indptr = np.empty(counts.size + 1, dtype=np.int32)
    indptr[0] = 0
    np.cumsum(counts, dtype=np.int32, out=indptr[1:])
    return indptr
