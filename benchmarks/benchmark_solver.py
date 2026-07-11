"""Run deterministic, staged ReynoldsFlow accuracy/performance benchmarks.

The default is SciPy's direct solver for a stable numerical reference. Base
SciPy/PyAMG and explicit optional backends can be selected for comparisons.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import resource
import sys
from statistics import median
from time import perf_counter
from typing import Any, Callable, TypeVar

import numpy as np
from reynoldsflow import transport as cartesian
from reynoldsflow import transport_polar as polar
from reynoldsflow._linear_solvers import solve_linear_system

try:
    from .cases import (
        CARTESIAN_CASES,
        POLAR_CASES,
        build_cartesian_case,
        build_polar_case,
    )
except ImportError:  # Support `python benchmarks/benchmark_solver.py`.
    from cases import (  # type: ignore[no-redef]
        CARTESIAN_CASES,
        POLAR_CASES,
        build_cartesian_case,
        build_polar_case,
    )


T = TypeVar("T")


def _timed(function: Callable[[], T]) -> tuple[T, float]:
    start = perf_counter()
    result = function()
    return result, perf_counter() - start


def _version(distribution: str) -> str | None:
    try:
        return metadata.version(distribution)
    except metadata.PackageNotFoundError:
        return None


def _peak_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reports KiB; macOS reports bytes.
    return int(value if sys.platform == "darwin" else value * 1024)


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", maxsplit=1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def _numpy_blas() -> dict[str, Any]:
    configuration = getattr(np.__config__, "CONFIG", {})
    blas = configuration.get("Build Dependencies", {}).get("blas", {})
    return {
        key: blas.get(key)
        for key in ("name", "version", "openblas configuration")
    }


def _environment() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_model": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "numpy_blas": _numpy_blas(),
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        },
        "packages": {
            name: _version(name)
            for name in (
                "reynoldsflow",
                "numpy",
                "scipy",
                "numba",
                "scikit-image",
                "pyamg",
                "pypardiso",
                "petsc4py",
                "scikit-sparse",
            )
        },
    }


def _relative_residual(matrix, solution: np.ndarray, rhs: np.ndarray) -> float:
    residual = np.linalg.norm(matrix @ solution - rhs)
    scale = max(np.linalg.norm(rhs), np.finfo(np.float64).tiny)
    return float(residual / scale)


def _total_timed_seconds(run: dict[str, Any]) -> float:
    return float(sum(run.get("timings_s", {}).values()))


def _summarize_runs(
    runs: list[dict[str, Any]], *, includes_cold_start: bool = True
) -> dict[str, Any]:
    """Keep cold-start and steady-state timings visibly separate."""
    if not runs:
        return {}
    summary: dict[str, Any] = {"recorded_runs": len(runs)}
    if includes_cold_start:
        cold = runs[0]
        steady = runs[1:]
        summary.update(
            {
                "cold_total_s": _total_timed_seconds(cold),
                "cold_linear_solve_s": cold.get("timings_s", {}).get(
                    "linear_solve"
                ),
            }
        )
    else:
        steady = runs
    if steady:
        summary.update(
            {
                "steady_runs": len(steady),
                "steady_total_median_s": median(
                    _total_timed_seconds(run) for run in steady
                ),
                "steady_linear_solve_median_s": median(
                    run["timings_s"]["linear_solve"] for run in steady
                ),
            }
        )
    return summary


def benchmark_cartesian(
    gaps: np.ndarray,
    *,
    compact_system: bool = True,
    solver: str = "scipy-spsolve",
    rtol: float = 1e-10,
) -> dict[str, Any]:
    n = gaps.shape[0]
    timings: dict[str, float] = {}

    filtered, timings["connectivity"] = _timed(
        lambda: cartesian.connectivity_analysis(gaps)
    )
    if filtered is None:
        return {
            "shape": list(gaps.shape),
            "cells": int(gaps.size),
            "input_open_cells": int(np.count_nonzero(gaps > 0.0)),
            "percolates": False,
            "timings_s": timings,
            "peak_rss_bytes": _peak_rss_bytes(),
        }

    if compact_system:
        (matrix_coo, rhs, dof_to_grid), timings["assembly"] = _timed(
            lambda: cartesian._create_solver_matrix(filtered)
        )
    else:
        (matrix_coo, rhs), timings["assembly"] = _timed(
            lambda: cartesian.create_diffusion_matrix(n, filtered)
        )
        dof_to_grid = None
    matrix_csc, timings["format_conversion"] = _timed(matrix_coo.tocsc)
    solve_result, timings["linear_solve"] = _timed(
        lambda: solve_linear_system(
            matrix_csc, rhs, solver=solver, rtol=rtol
        )
    )
    solution = solve_result.solution
    pressure = cartesian.reconstruct_full_solution(
        filtered.shape, solution, dof_to_grid
    )

    (flux_x, flux_y), timings["face_flux"] = _timed(
        lambda: cartesian._calculate_face_fluxes_numba(filtered, pressure)
    )
    flux, timings["flux_field"] = _timed(
        lambda: cartesian._cell_flux_from_faces_numba(
            filtered, flux_x, flux_y
        )
    )
    (total_flux, conservation_error), timings["flux_integration"] = _timed(
        lambda: cartesian.compute_total_flux(filtered, flux, n)
    )

    return {
        "shape": list(gaps.shape),
        "cells": int(gaps.size),
        "input_open_cells": int(np.count_nonzero(gaps > 0.0)),
        "active_cells": int(np.count_nonzero(filtered > 0.0)),
        "active_fraction": float(np.count_nonzero(filtered > 0.0) / gaps.size),
        "system_dofs": int(matrix_coo.shape[0]),
        "matrix_nnz": int(matrix_coo.nnz),
        "percolates": True,
        "relative_residual": _relative_residual(matrix_csc, solution, rhs),
        "linear_solver": solve_result.solver,
        "linear_iterations": solve_result.iterations,
        "total_flux": float(total_flux),
        "conservation_error": float(conservation_error),
        "pressure_min": float(np.min(pressure)),
        "pressure_max": float(np.max(pressure)),
        "timings_s": timings,
        "peak_rss_bytes": _peak_rss_bytes(),
    }


def benchmark_polar(
    gaps: np.ndarray,
    *,
    r_inner: float,
    r_outer: float,
    dilation_iterations: int,
    compact_system: bool = True,
    solver: str = "scipy-spsolve",
    rtol: float = 1e-10,
) -> dict[str, Any]:
    timings: dict[str, float] = {}
    theta_bc_code = polar.THETA_BC_PERIODIC
    theta_extent = 2.0 * np.pi

    filtered, timings["connectivity"] = _timed(
        lambda: polar.connectivity_analysis_polar(gaps, theta_bc_code)
    )
    if filtered is None:
        return {
            "shape": list(gaps.shape),
            "cells": int(gaps.size),
            "input_open_cells": int(np.count_nonzero(gaps > 0.0)),
            "percolates": False,
            "timings_s": timings,
            "peak_rss_bytes": _peak_rss_bytes(),
        }

    solve_gaps = filtered
    if dilation_iterations > 0:
        solve_gaps, timings["dilation"] = _timed(
            lambda: polar._dilate_gaps_polar(
                filtered, dilation_iterations, theta_bc_code
            )
        )

    if compact_system:
        (matrix_coo, rhs, dr, dtheta, dof_to_grid), timings["assembly"] = _timed(
            lambda: polar._create_solver_matrix_polar(
                solve_gaps,
                r_inner,
                r_outer,
                theta_extent,
                theta_bc_code,
                1.0,
                0.0,
            )
        )
    else:
        (matrix_coo, rhs, dr, dtheta), timings["assembly"] = _timed(
            lambda: polar.create_diffusion_matrix_polar(
                solve_gaps,
                r_inner,
                r_outer,
                theta_extent,
                theta_bc_code,
                1.0,
                0.0,
            )
        )
        dof_to_grid = None
    matrix_csc, timings["format_conversion"] = _timed(matrix_coo.tocsc)
    solve_result, timings["linear_solve"] = _timed(
        lambda: solve_linear_system(
            matrix_csc, rhs, solver=solver, rtol=rtol
        )
    )
    solution = solve_result.solution
    pressure = polar.reconstruct_full_solution(
        solve_gaps.shape, solution, dof_to_grid
    )

    n_r, n_theta = solve_gaps.shape
    (flux_r, flux_theta), timings["face_flux"] = _timed(
        lambda: polar._calculate_face_fluxes_polar(
            solve_gaps,
            pressure,
            r_inner,
            dr,
            dtheta,
            theta_bc_code,
        )
    )
    flux, timings["flux_field"] = _timed(
        lambda: polar._cell_flux_from_faces_polar(
            solve_gaps, flux_r, flux_theta, r_inner, dr
        )
    )
    flux[filtered <= 0.0] = np.nan
    (total_flux, conservation_error), timings["flux_integration"] = _timed(
        lambda: polar.compute_total_flux_polar(
            filtered, flux, r_inner, r_outer, dtheta
        )
    )

    return {
        "shape": list(gaps.shape),
        "cells": int(gaps.size),
        "input_open_cells": int(np.count_nonzero(gaps > 0.0)),
        "active_cells": int(np.count_nonzero(filtered > 0.0)),
        "solve_open_cells": int(np.count_nonzero(solve_gaps > 0.0)),
        "active_fraction": float(np.count_nonzero(filtered > 0.0) / gaps.size),
        "system_dofs": int(matrix_coo.shape[0]),
        "matrix_nnz": int(matrix_coo.nnz),
        "percolates": True,
        "relative_residual": _relative_residual(matrix_csc, solution, rhs),
        "linear_solver": solve_result.solver,
        "linear_iterations": solve_result.iterations,
        "total_flux": float(total_flux),
        "conservation_error": float(conservation_error),
        "pressure_min": float(np.min(pressure)),
        "pressure_max": float(np.max(pressure)),
        "dr": float(dr),
        "dtheta": float(dtheta),
        "timings_s": timings,
        "peak_rss_bytes": _peak_rss_bytes(),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry", choices=("cartesian", "polar"), default="cartesian"
    )
    parser.add_argument("--case", default="constant")
    parser.add_argument("--size", type=int, default=128, help="n or n_r")
    parser.add_argument(
        "--n-theta", type=int, default=None, help="polar angular resolution"
    )
    parser.add_argument("--seed", type=int, default=23_349)
    parser.add_argument("--repeat", type=int, default=1)
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="discard this many complete runs before recording --repeat runs",
    )
    parser.add_argument("--solver", default="scipy-spsolve")
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--r-inner", type=float, default=1.0)
    parser.add_argument("--r-outer", type=float, default=2.0)
    parser.add_argument("--dilation-iterations", type=int, default=0)
    parser.add_argument(
        "--full-grid-system",
        action="store_true",
        help="benchmark the legacy full-grid matrix instead of active DOFs",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--list-cases", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.list_cases:
        print("Cartesian:", ", ".join(CARTESIAN_CASES))
        print("Polar:", ", ".join(POLAR_CASES))
        return 0
    if args.repeat < 1:
        raise ValueError("--repeat must be at least 1.")
    if args.warmup < 0:
        raise ValueError("--warmup cannot be negative.")

    if args.geometry == "cartesian":
        gaps = build_cartesian_case(args.case, args.size, seed=args.seed)
        run = lambda: benchmark_cartesian(
            gaps,
            compact_system=not args.full_grid_system,
            solver=args.solver,
            rtol=args.rtol,
        )
        shape = list(gaps.shape)
    else:
        n_theta = args.n_theta if args.n_theta is not None else 2 * args.size
        gaps = build_polar_case(args.case, args.size, n_theta)
        run = lambda: benchmark_polar(
            gaps,
            r_inner=args.r_inner,
            r_outer=args.r_outer,
            dilation_iterations=args.dilation_iterations,
            compact_system=not args.full_grid_system,
            solver=args.solver,
            rtol=args.rtol,
        )
        shape = list(gaps.shape)

    for _ in range(args.warmup):
        run()
        gc.collect()

    runs = []
    for _ in range(args.repeat):
        runs.append(run())
        gc.collect()

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(),
        "configuration": {
            "geometry": args.geometry,
            "case": args.case,
            "shape": shape,
            "seed": args.seed,
            "repeat": args.repeat,
            "warmup": args.warmup,
            "solver": args.solver,
            "rtol": args.rtol,
            "compact_system": not args.full_grid_system,
            "r_inner": args.r_inner if args.geometry == "polar" else None,
            "r_outer": args.r_outer if args.geometry == "polar" else None,
            "dilation_iterations": (
                args.dilation_iterations if args.geometry == "polar" else None
            ),
        },
        "runs": runs,
        "summary": _summarize_runs(
            runs, includes_cold_start=args.warmup == 0
        ),
    }
    serialized = json.dumps(report, indent=2, sort_keys=True)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
