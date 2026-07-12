"""Connected-component helpers shared by Cartesian and polar solvers."""

from __future__ import annotations

from typing import Optional

import numpy as np
from numba import njit
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


@njit
def _label_periodic_components_numba(open_mask: np.ndarray):
    """Label 4-connected components on a 2D torus and detect x winding."""
    n_x, n_y = open_mask.shape
    size = n_x * n_y
    labels = np.zeros((n_x, n_y), dtype=np.int32)
    lifted_x = np.zeros(size, dtype=np.int64)
    stack = np.empty(size, dtype=np.int64)
    winding = np.zeros(size + 1, dtype=np.bool_)
    component = 0

    for start_i in range(n_x):
        for start_j in range(n_y):
            if not open_mask[start_i, start_j] or labels[start_i, start_j] != 0:
                continue
            component += 1
            start = start_i * n_y + start_j
            labels[start_i, start_j] = component
            lifted_x[start] = 0
            stack_size = 1
            stack[0] = start

            while stack_size:
                stack_size -= 1
                grid_index = int(stack[stack_size])
                i = grid_index // n_y
                j = grid_index - i * n_y
                current_x = lifted_x[grid_index]

                for direction in range(4):
                    neighbor_i = i
                    neighbor_j = j
                    x_step = 0
                    if direction == 0:
                        neighbor_i = (i - 1) % n_x
                        x_step = -1
                    elif direction == 1:
                        neighbor_i = (i + 1) % n_x
                        x_step = 1
                    elif direction == 2:
                        neighbor_j = (j - 1) % n_y
                    else:
                        neighbor_j = (j + 1) % n_y

                    if not open_mask[neighbor_i, neighbor_j]:
                        continue
                    neighbor_index = neighbor_i * n_y + neighbor_j
                    candidate_x = current_x + x_step
                    if labels[neighbor_i, neighbor_j] == 0:
                        labels[neighbor_i, neighbor_j] = component
                        lifted_x[neighbor_index] = candidate_x
                        stack[stack_size] = neighbor_index
                        stack_size += 1
                    elif (
                        labels[neighbor_i, neighbor_j] == component
                        and lifted_x[neighbor_index] != candidate_x
                    ):
                        winding[component] = True

    return labels, winding[: component + 1]


def label_periodic_components(gaps: np.ndarray):
    """Return toroidal component labels and x-winding flags.

    ``winding[label]`` is true when that component contains a closed path
    whose lifted x coordinate changes by a nonzero multiple of the cell size.
    """
    gaps = np.asarray(gaps)
    if gaps.ndim != 2:
        raise ValueError("Connectivity analysis requires a 2D gap array.")
    return _label_periodic_components_numba(gaps > 0.0)
