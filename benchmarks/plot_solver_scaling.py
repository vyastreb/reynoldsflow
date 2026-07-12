"""Plot ReynoldsFlow solver scaling from benchmark-suite JSON reports."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


LABELS = {
    "scipy-spsolve": "SciPy SuperLU",
    "scipy.amg-rs": "PyAMG Ruge–Stuben",
    "scipy.amg-smooth_aggregation": "PyAMG smoothed aggregation",
    "cholesky": "CHOLMOD",
    "pardiso": "oneMKL Pardiso",
    "petsc-cg.hypre": "PETSc CG + Hypre",
    "petsc-cg.gamg": "PETSc CG + GAMG",
    "petsc-mumps": "PETSc + MUMPS",
}

STYLES = {
    "scipy-spsolve": ("#7f7f7f", "X"),
    "scipy.amg-rs": ("#9467bd", "o"),
    "scipy.amg-smooth_aggregation": ("#e377c2", "P"),
    "cholesky": ("#1f77b4", "v"),
    "pardiso": ("#ff7f0e", "o"),
    "petsc-cg.hypre": ("#d62728", "o"),
    "petsc-cg.gamg": ("#2ca02c", "o"),
    "petsc-mumps": ("#17becf", "s"),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path)
    return parser.parse_args()


def _load_reports(
    paths: list[Path],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    series: dict[str, list[dict[str, Any]]] = {}
    reference_configuration: dict[str, Any] | None = None
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        configuration = report["configuration"]
        comparable = {
            key: configuration[key]
            for key in ("case", "rtol", "threads", "compact_system")
        }
        if reference_configuration is None:
            reference_configuration = comparable
        elif comparable != reference_configuration:
            raise ValueError(f"Incompatible benchmark configuration in {path}")

        for source_row in report["summary"]:
            row = dict(source_row)
            if row["status"] != "ok":
                continue
            row["grid_size"] = configuration["size"]
            series.setdefault(row["solver"], []).append(row)

    for rows in series.values():
        rows.sort(key=lambda row: row["system_dofs"])
    return series, reference_configuration or {}


def _write_csv(path: Path, series: dict[str, list[dict[str, Any]]]) -> None:
    fields = (
        "grid_size",
        "system_dofs",
        "matrix_nnz",
        "solver",
        "steady_total_median_s",
        "steady_linear_solve_median_s",
        "peak_rss_bytes",
        "iterations",
        "relative_residual",
        "conservation_error",
        "total_flux",
    )
    rows = sorted(
        (row for solver_rows in series.values() for row in solver_rows),
        key=lambda row: (row["grid_size"], row["solver"]),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = _parse_args()
    series, configuration = _load_reports(args.reports)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.35,
            "grid.linewidth": 0.7,
        }
    )
    figure, (time_axis, memory_axis) = plt.subplots(1, 2, figsize=(12, 5))

    for solver, rows in series.items():
        color, marker = STYLES[solver]
        dofs = [row["system_dofs"] for row in rows]
        times = [row["steady_total_median_s"] for row in rows]
        memory = [row["peak_rss_bytes"] / 2**30 for row in rows]
        options = {
            "color": color,
            "marker": marker,
            "linewidth": 1.8,
            "markersize": 6,
            "label": LABELS[solver],
        }
        time_axis.plot(dofs, times, **options)
        memory_axis.plot(dofs, memory, **options)

    for axis in (time_axis, memory_axis):
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_xlabel("Active degrees of freedom")
        axis.grid(which="minor", alpha=0.15)

    time_axis.set_ylabel("Steady end-to-end wall time (s)")
    time_axis.set_title("Runtime")
    memory_axis.set_ylabel("Peak process RSS (GiB)")
    memory_axis.set_title("Memory")
    memory_axis.legend(loc="upper left", fontsize=8, ncol=1, framealpha=0.95)

    figure.suptitle(
        "ReynoldsFlow v0.1.1 — deterministic rough-contact benchmark",
        fontsize=12,
    )
    figure.text(
        0.5,
        0.01,
        (
            f"case={configuration.get('case')}, rtol={configuration.get('rtol'):.0e}, "
            f"native threads={configuration.get('threads')}; medians exclude cold run"
        ),
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.0, 0.045, 1.0, 0.95))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=220, bbox_inches="tight")
    plt.close(figure)
    if args.csv_output is not None:
        _write_csv(args.csv_output, series)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
