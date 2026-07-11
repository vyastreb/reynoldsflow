"""Saved debugging matrices honor the requested sparse format."""

import numpy as np
import pytest
from scipy.sparse import load_npz

from reynoldsflow.transport import solve_diffusion
from reynoldsflow.transport_polar import (
    THETA_BC_PERIODIC,
    solve_diffusion_polar,
)


pytestmark = pytest.mark.unit


@pytest.mark.parametrize("sparse_format", ["coo", "csr", "csc"])
def test_cartesian_saved_matrix_format(tmp_path, monkeypatch, sparse_format):
    monkeypatch.chdir(tmp_path)
    gaps = np.ones((6, 6), dtype=np.float64)
    solve_diffusion(
        6,
        gaps,
        solver="scipy-spsolve",
        save_matrix=True,
        save_matrix_type=sparse_format,
    )

    matrix = load_npz(tmp_path / "transport_matrix.npz")
    rhs = np.load(tmp_path / "transport_rhs.npz")["b"]
    assert matrix.format == sparse_format
    assert rhs.shape == (36,)


@pytest.mark.parametrize("sparse_format", ["coo", "csr", "csc"])
def test_polar_saved_matrix_format(tmp_path, monkeypatch, sparse_format):
    monkeypatch.chdir(tmp_path)
    gaps = np.ones((6, 8), dtype=np.float64)
    solve_diffusion_polar(
        gaps,
        1.0,
        2.0,
        2.0 * np.pi,
        THETA_BC_PERIODIC,
        solver="scipy-spsolve",
        save_matrix=True,
        save_matrix_type=sparse_format,
    )

    matrix = load_npz(tmp_path / "transport_matrix_polar.npz")
    rhs = np.load(tmp_path / "transport_rhs_polar.npz")["b"]
    assert matrix.format == sparse_format
    assert rhs.shape == (48,)
