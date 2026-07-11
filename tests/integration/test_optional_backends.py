"""Subprocess-isolated checks for optional native solver backends."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
import textwrap

import pytest


pytestmark = pytest.mark.backend


BACKENDS = (
    ("cholesky", "sksparse"),
    ("pardiso", "pypardiso"),
    ("petsc-cg.hypre", "petsc4py"),
    ("petsc-cg.gamg", "petsc4py"),
    ("petsc-gmres.ilu", "petsc4py"),
    ("petsc-mumps", "petsc4py"),
)


_BACKEND_SCRIPT = textwrap.dedent(
    """
    import json
    import sys
    import numpy as np
    from reynoldsflow import transport

    solver = sys.argv[1]
    gaps = np.ones((16, 16), dtype=np.float64)
    filtered, pressure, flux = transport.solve_fluid_problem(
        gaps, solver=solver, rtol=1e-10
    )
    total_flux, conservation_error = transport.compute_total_flux(
        filtered, flux, gaps.shape[0]
    )
    print(json.dumps({
        "total_flux": total_flux,
        "conservation_error": conservation_error,
        "pressure_min": float(np.min(pressure)),
        "pressure_max": float(np.max(pressure)),
    }))
    """
)


@pytest.mark.parametrize(("solver", "required_module"), BACKENDS)
def test_optional_backend_in_isolated_process(solver, required_module):
    if importlib.util.find_spec(required_module) is None:
        pytest.skip(f"{required_module} is not installed")

    try:
        completed = subprocess.run(
            [sys.executable, "-u", "-c", _BACKEND_SCRIPT, solver],
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        pytest.fail(f"Backend {solver} timed out after {exc.timeout} seconds")

    if completed.returncode != 0:
        pytest.fail(
            f"Backend {solver} exited with {completed.returncode}.\n"
            f"stdout:\n{completed.stdout[-4000:]}\n"
            f"stderr:\n{completed.stderr[-4000:]}"
        )

    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload["total_flux"] == pytest.approx(1.0, rel=1e-8, abs=1e-10)
    assert payload["conservation_error"] < 1e-7
    assert 0.0 < payload["pressure_min"] < payload["pressure_max"] < 1.0
