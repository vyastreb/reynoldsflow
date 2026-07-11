"""Run every linear-solver benchmark in a crash-isolated subprocess."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from time import perf_counter
from typing import Any


DEFAULT_SOLVERS = (
    "scipy-spsolve",
    "scipy.amg-rs",
    "scipy.amg-smooth_aggregation",
    "cholesky",
    "pardiso",
    "petsc-cg.hypre",
    "petsc-cg.gamg",
    "petsc-mumps",
)


def _text_tail(value: str | bytes | None, limit: int = 2000) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    return value[-limit:]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--case", default="circle")
    parser.add_argument("--rtol", type=float, default=1e-8)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--solvers", nargs="+", default=list(DEFAULT_SOLVERS))
    parser.add_argument("--full-grid-system", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _run_solver(
    solver: str,
    args: argparse.Namespace,
    result_path: Path,
) -> dict[str, Any]:
    command = [
        sys.executable,
        "-m",
        "benchmarks.benchmark_solver",
        "--geometry",
        "cartesian",
        "--case",
        args.case,
        "--size",
        str(args.size),
        "--solver",
        solver,
        "--rtol",
        str(args.rtol),
        "--repeat",
        str(args.repeat),
        "--output",
        str(result_path),
    ]
    if args.full_grid_system:
        command.append("--full-grid-system")

    environment = os.environ.copy()
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        environment[name] = str(args.threads)

    start = perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=Path(__file__).resolve().parents[1],
            env=environment,
            capture_output=True,
            text=True,
            timeout=args.timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "solver": solver,
            "status": "timeout",
            "elapsed_process_s": perf_counter() - start,
            "error": f"exceeded {args.timeout:g} seconds",
            "stderr_tail": _text_tail(exc.stderr),
        }

    result: dict[str, Any] = {
        "solver": solver,
        "elapsed_process_s": perf_counter() - start,
        "returncode": completed.returncode,
    }
    if completed.returncode == 0 and result_path.exists():
        result["status"] = "ok"
        result["benchmark"] = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        result["status"] = "crashed" if completed.returncode < 0 else "failed"
        result["error"] = f"worker exited with code {completed.returncode}"
    if completed.stderr:
        result["stderr_tail"] = _text_tail(completed.stderr)
    return result


def _summary_row(result: dict[str, Any]) -> dict[str, Any]:
    row = {
        "solver": result["solver"],
        "status": result["status"],
        "elapsed_process_s": result["elapsed_process_s"],
    }
    benchmark = result.get("benchmark")
    if benchmark is None:
        row["error"] = result.get("error")
        return row
    timing = benchmark["summary"]
    first_run = benchmark["runs"][0]
    row.update(
        {
            "cold_total_s": timing.get("cold_total_s"),
            "steady_total_median_s": timing.get("steady_total_median_s"),
            "cold_linear_solve_s": timing.get("cold_linear_solve_s"),
            "steady_linear_solve_median_s": timing.get(
                "steady_linear_solve_median_s"
            ),
            "iterations": first_run.get("linear_iterations"),
            "relative_residual": first_run.get("relative_residual"),
            "conservation_error": first_run.get("conservation_error"),
            "total_flux": first_run.get("total_flux"),
            "system_dofs": first_run.get("system_dofs"),
            "matrix_nnz": first_run.get("matrix_nnz"),
            "peak_rss_bytes": max(
                run.get("peak_rss_bytes", 0) for run in benchmark["runs"]
            ),
        }
    )
    return row


def main() -> int:
    args = _parse_args()
    if args.repeat < 1:
        raise ValueError("--repeat must be at least 1.")
    if args.timeout <= 0.0:
        raise ValueError("--timeout must be positive.")
    if args.threads < 1:
        raise ValueError("--threads must be at least 1.")

    created_utc = datetime.now(timezone.utc).isoformat()
    with tempfile.TemporaryDirectory(prefix="reynoldsflow-bench-") as directory:
        temporary_directory = Path(directory)
        results = []
        for index, solver in enumerate(args.solvers):
            print(
                f"[{index + 1}/{len(args.solvers)}] benchmarking {solver}",
                file=sys.stderr,
                flush=True,
            )
            result = _run_solver(
                solver, args, temporary_directory / f"result-{index}.json"
            )
            results.append(result)
            print(
                f"[{index + 1}/{len(args.solvers)}] {solver}: {result['status']}",
                file=sys.stderr,
                flush=True,
            )
            if args.output is not None:
                partial_report = _build_report(args, created_utc, results)
                _write_report(args.output, partial_report)

    report = _build_report(args, created_utc, results)
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        _write_report(args.output, report)
    print(serialized)
    return 0 if all(result["status"] == "ok" for result in results) else 1


def _build_report(
    args: argparse.Namespace,
    created_utc: str,
    results: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_utc": created_utc,
        "configuration": {
            "case": args.case,
            "size": args.size,
            "rtol": args.rtol,
            "repeat": args.repeat,
            "timeout_s": args.timeout,
            "threads": args.threads,
            "compact_system": not args.full_grid_system,
        },
        "summary": [_summary_row(result) for result in results],
        "results": results,
    }


def _write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    raise SystemExit(main())
