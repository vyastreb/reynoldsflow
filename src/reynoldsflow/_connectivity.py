"""Connected-component helpers shared by Cartesian and polar solvers."""

from __future__ import annotations

from typing import Optional

import numpy as np
from skimage.measure import label


def _normalize_axis(axis: int, ndim: int) -> int:
    normalized = axis + ndim if axis < 0 else axis
    if normalized < 0 or normalized >= ndim:
        raise ValueError(f"Axis {axis} is out of bounds for an array of rank {ndim}.")
    return normalized


def find_spanning_mask(
    gaps: np.ndarray,
    transport_axis: int,
    periodic_axis: Optional[int] = None,
) -> Optional[np.ndarray]:
    """Return the union of all positive components spanning two boundaries.

    Local components use face/4-connectivity. When ``periodic_axis`` is set,
    component labels touching opposite samples of that axis are merged with a
    union-find table. Python work scales with the number of labels and seam
    pairs, not with the total number of grid cells.
    """
    gaps = np.asarray(gaps)
    if gaps.ndim != 2:
        raise ValueError("Connectivity analysis requires a 2D gap array.")

    transport_axis = _normalize_axis(transport_axis, gaps.ndim)
    if periodic_axis is not None:
        periodic_axis = _normalize_axis(periodic_axis, gaps.ndim)
        if periodic_axis == transport_axis:
            raise ValueError("Transport and periodic axes must be different.")

    labels = label(gaps > 0.0, connectivity=1)
    max_label = int(labels.max())
    if max_label == 0:
        return None

    parent = np.arange(max_label + 1, dtype=labels.dtype)
    rank = np.zeros(max_label + 1, dtype=np.uint8)

    def find(component: int) -> int:
        root = component
        while int(parent[root]) != root:
            root = int(parent[root])
        while component != root:
            next_component = int(parent[component])
            parent[component] = root
            component = next_component
        return root

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        if rank[first_root] < rank[second_root]:
            first_root, second_root = second_root, first_root
        parent[second_root] = first_root
        if rank[first_root] == rank[second_root]:
            rank[first_root] += 1

    if periodic_axis is not None and gaps.shape[periodic_axis] > 1:
        first_seam = np.take(labels, 0, axis=periodic_axis).ravel()
        last_seam = np.take(labels, -1, axis=periodic_axis).ravel()
        valid = (first_seam > 0) & (last_seam > 0)
        if np.any(valid):
            seam_pairs = np.unique(
                np.column_stack((first_seam[valid], last_seam[valid])), axis=0
            )
            for first, second in seam_pairs:
                union(int(first), int(second))

    roots = np.arange(max_label + 1, dtype=labels.dtype)
    for component in range(1, max_label + 1):
        roots[component] = find(component)

    first_boundary = roots[np.take(labels, 0, axis=transport_axis)]
    last_boundary = roots[np.take(labels, -1, axis=transport_axis)]
    first_roots = np.unique(first_boundary[first_boundary > 0])
    last_roots = np.unique(last_boundary[last_boundary > 0])
    spanning_roots = np.intersect1d(first_roots, last_roots, assume_unique=True)
    if spanning_roots.size == 0:
        return None

    keep_root = np.zeros(max_label + 1, dtype=np.bool_)
    keep_root[spanning_roots] = True
    np.take(roots, labels, out=labels)
    return keep_root[labels]
