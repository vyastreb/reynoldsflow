"""Benchmark repeated solves with fresh or prepared ReynoldsFlow topology."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from reynoldsflow import transport as cartesian
from reynoldsflow import transport_polar as polar

from .benchmark_solver import _environment, _peak_rss_bytes
from .cases import build_cartesian_case, build_polar_case


def _cartesian_variation(base: np.ndarray, step: int, steps: int) -> np.ndarray:
    n = base.shape[0]
    x = (np.arange(n, dtype=np.float64) + 0.5) / n
    phase = 2.0 * np.pi * step / max(steps, 1)
    factor = 1.0 + 0.08 * np.sin(2.0 * np.pi * x + phase)
    return base * factor[:, None]


def _polar_variation(base: np.ndarray, step: int, steps: int) -> np.ndarray:
    n_r = base.shape[0]
    radial = np.linspace(0.0, 1.0, n_r, dtype=np.float64)
    phase = 2.0 * np.pi * step / max(steps, 1)
    factor = 1.0 + 0.08 * np.sin(2.0 * np.pi * radial + phase)
    return base * factor[:, None]


def _run_cartesian(args):
    base = build_cartesian_case("rough-contact", args.size, seed=args.seed)
    prepared = None
    preparation_s = 0.0
    if not args.fresh:
        start = perf_counter()
        prepared = cartesian.prepare_fluid_problem(base)
        preparation_s = perf_counter() - start

    runs = []
    for step in range(args.steps):
        gaps = _cartesian_variation(base, step, args.steps)
        start = perf_counter()
        if prepared is None:
            filtered, pressure, flux = cartesian.solve_fluid_problem(
                gaps, solver=args.solver, rtol=args.rtol
            )
            diagnostics = None
        else:
            filtered, pressure, flux, diagnostics = (
                prepared.solve_with_diagnostics(
                    gaps,
                    solver=args.solver,
                    rtol=args.rtol,
                    reuse_preconditioner=args.reuse_preconditioner,
                )
            )
        elapsed = perf_counter() - start
        total_flux, conservation_error = cartesian.compute_total_flux(
            filtered, flux, gaps.shape[0]
        )
        runs.append(
            {
                "step": step,
                "elapsed_s": elapsed,
                "total_flux": float(total_flux),
                "conservation_error": float(conservation_error),
                "iterations": None if diagnostics is None else diagnostics.iterations,
                "relative_residual": (
                    None if diagnostics is None else diagnostics.relative_residual
                ),
            }
        )
    return base.shape, preparation_s, runs


def _run_polar(args):
    n_theta = args.n_theta if args.n_theta is not None else 2 * args.size
    base = build_polar_case("blocked-sector", args.size, n_theta)
    prepared = None
    preparation_s = 0.0
    if not args.fresh:
        start = perf_counter()
        prepared = polar.prepare_fluid_problem_polar(base, 1.0, 2.0)
        preparation_s = perf_counter() - start

    runs = []
    for step in range(args.steps):
        gaps = _polar_variation(base, step, args.steps)
        start = perf_counter()
        if prepared is None:
            filtered, pressure, flux, dr, dtheta = (
                polar.solve_fluid_problem_polar(
                    gaps,
                    1.0,
                    2.0,
                    solver=args.solver,
                    rtol=args.rtol,
                )
            )
            diagnostics = None
        else:
            filtered, pressure, flux, dr, dtheta, diagnostics = (
                prepared.solve_with_diagnostics(
                    gaps,
                    solver=args.solver,
                    rtol=args.rtol,
                    reuse_preconditioner=args.reuse_preconditioner,
                )
            )
        elapsed = perf_counter() - start
        total_flux, conservation_error = polar.compute_total_flux_polar(
            filtered, flux, 1.0, 2.0, dtheta
        )
        runs.append(
            {
                "step": step,
                "elapsed_s": elapsed,
                "total_flux": float(total_flux),
                "conservation_error": float(conservation_error),
                "iterations": None if diagnostics is None else diagnostics.iterations,
                "relative_residual": (
                    None if diagnostics is None else diagnostics.relative_residual
                ),
            }
        )
    return base.shape, preparation_s, runs


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--geometry", choices=("cartesian", "polar"), default="cartesian"
    )
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--n-theta", type=int)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=23_349)
    parser.add_argument("--solver", default="scipy.amg-rs")
    parser.add_argument("--rtol", type=float, default=1e-10)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--reuse-preconditioner", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.steps < 1:
        raise ValueError("--steps must be at least 1.")
    if args.fresh and args.reuse_preconditioner:
        raise ValueError("--reuse-preconditioner requires prepared mode.")

    if args.geometry == "cartesian":
        shape, preparation_s, runs = _run_cartesian(args)
    else:
        shape, preparation_s, runs = _run_polar(args)

    report = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "environment": _environment(),
        "configuration": {
            "geometry": args.geometry,
            "shape": list(shape),
            "steps": args.steps,
            "solver": args.solver,
            "rtol": args.rtol,
            "mode": "fresh" if args.fresh else "prepared",
            "reuse_preconditioner": args.reuse_preconditioner,
        },
        "preparation_s": preparation_s,
        "runs": runs,
        "mean_step_s": float(np.mean([run["elapsed_s"] for run in runs])),
        "peak_rss_bytes": _peak_rss_bytes(),
    }
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
