"""Generate and homogenize self-affine rough gaps.

The production study is intentionally split into small, restartable tasks.  A
single 4096 x 4096 solve can be expensive, and independent task files are safer
than a shared CSV when the jobs are submitted as an HPC array.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from scipy.optimize import brentq


SCHEMA_VERSION = 1
DEFAULT_SIZE = 4096
DEFAULT_HURST = 0.8
DEFAULT_ROLLOFF_RATIOS = (1, 2, 4, 6, 8)
DEFAULT_SHORT_WAVE_RATIO = 64
DEFAULT_REALIZATIONS = 5
DEFAULT_CONTACT_FRACTIONS = (0.10, 0.20, 0.30)
DEFAULT_BASE_SEED = 20260923
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "output"
MODES = ("pressure", "mean-flux")


@dataclass(frozen=True)
class SurfaceTask:
    rolloff_ratio: int
    realization: int


@dataclass(frozen=True)
class SimulationTask:
    rolloff_ratio: int
    realization: int
    contact_fraction: float
    mode: str


@dataclass(frozen=True)
class TheoryTask:
    rolloff_ratio: int
    realization: int
    contact_fraction: float


def _csv_numbers(value: str, cast) -> tuple:
    return tuple(cast(part.strip()) for part in value.split(",") if part.strip())


def surface_tasks(
    ratios: Sequence[int], realizations: int
) -> list[SurfaceTask]:
    return [
        SurfaceTask(ratio, realization)
        for ratio in ratios
        for realization in range(realizations)
    ]


def simulation_tasks(
    ratios: Sequence[int],
    realizations: int,
    contacts: Sequence[float],
    modes: Sequence[str] = MODES,
) -> list[SimulationTask]:
    return [
        SimulationTask(ratio, realization, contact, mode)
        for contact in contacts
        for ratio in ratios
        for realization in range(realizations)
        for mode in modes
    ]


def theory_tasks(
    ratios: Sequence[int], realizations: int, contacts: Sequence[float]
) -> list[TheoryTask]:
    return [
        TheoryTask(ratio, realization, contact)
        for contact in contacts
        for ratio in ratios
        for realization in range(realizations)
    ]


def _contact_tag(contact_fraction: float) -> str:
    return f"c{100.0 * contact_fraction:05.1f}".replace(".", "p")


def _surface_stem(task: SurfaceTask) -> str:
    return f"surface_r{task.rolloff_ratio:02d}_i{task.realization:02d}"


def surface_path(output: Path, task: SurfaceTask) -> Path:
    return output / "surfaces" / f"{_surface_stem(task)}.npy"


def surface_metadata_path(output: Path, task: SurfaceTask) -> Path:
    return output / "surfaces" / f"{_surface_stem(task)}.json"


def simulation_result_path(output: Path, task: SimulationTask) -> Path:
    return (
        output
        / "results"
        / (
            f"result_r{task.rolloff_ratio:02d}_i{task.realization:02d}_"
            f"{_contact_tag(task.contact_fraction)}_{task.mode}.json"
        )
    )


def theory_result_path(output: Path, task: TheoryTask) -> Path:
    return (
        output
        / "theory"
        / (
            f"bruggeman_r{task.rolloff_ratio:02d}_i{task.realization:02d}_"
            f"{_contact_tag(task.contact_fraction)}.json"
        )
    )


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.replace(temporary, path)


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        np.save(stream, array, allow_pickle=False)
    os.replace(temporary, path)


def realization_seed(base_seed: int, realization: int) -> int:
    """Return a stable seed shared across roll-offs for paired ensembles."""
    sequence = np.random.SeedSequence([base_seed, realization])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _short_wave_cutoff_ratio(
    rolloff_ratio: int, short_wave_ratio: int, reference: str
) -> int:
    """Return ``k2/k0`` when the supplied ratio references ``k0`` or ``k1``."""
    if reference == "k0":
        cutoff = short_wave_ratio
    elif reference == "k1":
        cutoff = short_wave_ratio * rolloff_ratio
    else:
        raise ValueError("short-wave reference must be 'k0' or 'k1'")
    if cutoff <= rolloff_ratio:
        raise ValueError("short-wave cutoff k2 must exceed roll-off k1")
    return cutoff


def generate_surface(
    task: SurfaceTask,
    *,
    output: Path,
    size: int,
    hurst: float,
    short_wave_ratio: int,
    base_seed: int,
    short_wave_reference: str = "k0",
    force: bool = False,
) -> Path:
    """Generate one unit-RMS periodic surface and its provenance metadata."""
    from rfgen import selfaffine_field

    if task.rolloff_ratio < 1:
        raise ValueError("roll-off ratios must be positive integers")
    k2_over_k0 = _short_wave_cutoff_ratio(
        task.rolloff_ratio, short_wave_ratio, short_wave_reference
    )
    if k2_over_k0 / size > 0.5:
        raise ValueError(
            "short-wave cutoff exceeds the discrete Nyquist limit; require "
            "size >= 2 * (k2/k0)"
        )
    path = surface_path(output, task)
    metadata_path = surface_metadata_path(output, task)
    if path.exists() and metadata_path.exists() and not force:
        print(f"skip existing {path}")
        return path

    seed = realization_seed(base_seed, task.realization)
    started = time.perf_counter()
    surface = selfaffine_field(
        dim=2,
        N=size,
        Hurst=hurst,
        k_low=task.rolloff_ratio / size,
        k_high=k2_over_k0 / size,
        plateau=True,
        noise=True,
        rng=np.random.default_rng(seed),
        dtype=np.float32,
    )
    surface -= np.mean(surface, dtype=np.float64)
    rms_before = float(np.std(surface, dtype=np.float64))
    if not np.isfinite(rms_before) or rms_before <= 0.0:
        raise RuntimeError("rfgen returned a degenerate surface")
    surface /= rms_before
    normalized_mean = float(np.mean(surface, dtype=np.float64))
    normalized_m0 = float(np.var(surface, dtype=np.float64))
    elapsed = time.perf_counter() - started
    _atomic_npy(path, np.asarray(surface, dtype=np.float32))

    try:
        from importlib.metadata import version

        rfgen_version = version("rfgen")
    except Exception:  # pragma: no cover - metadata is normally available
        rfgen_version = "unknown"
    _atomic_json(
        metadata_path,
        {
            "schema_version": SCHEMA_VERSION,
            "shape": [size, size],
            "dtype": "float32",
            "hurst": hurst,
            "rolloff_ratio_k1_over_k0": task.rolloff_ratio,
            "short_wave_reference": short_wave_reference,
            "specified_short_wave_ratio": short_wave_ratio,
            "short_wave_ratio_k2_over_k0": k2_over_k0,
            "short_wave_ratio_k2_over_k1": (
                k2_over_k0 / task.rolloff_ratio
            ),
            "rfgen_k_low_cycles_per_sample": task.rolloff_ratio / size,
            "rfgen_k_high_cycles_per_sample": k2_over_k0 / size,
            "shortest_wavelength_grid_cells": size / k2_over_k0,
            "plateau": True,
            "noise": True,
            "realization": task.realization,
            "base_seed": base_seed,
            "seed": seed,
            "paired_seed_across_rolloffs": True,
            "normalization": "zero mean, unit RMS height",
            "normalized_mean": normalized_mean,
            "normalized_m0": normalized_m0,
            "rfgen_version": rfgen_version,
            "generation_seconds": elapsed,
        },
    )
    print(f"generated {path} in {elapsed:.2f} s")
    return path


def gap_at_contact_fraction(
    surface: np.ndarray, contact_fraction: float
) -> tuple[np.ndarray, float, float]:
    """Truncate a surface against a rigid plane at a target contact fraction."""
    if not 0.0 < contact_fraction < 1.0:
        raise ValueError("contact fraction must lie strictly between zero and one")
    threshold = float(np.quantile(surface, contact_fraction, method="linear"))
    gap = np.asarray(surface, dtype=np.float64) - threshold
    np.maximum(gap, 0.0, out=gap)
    actual_contact = float(np.count_nonzero(gap == 0.0) / gap.size)
    return gap, threshold, actual_contact


def bruggeman_transmissivity(
    filtered_gap: np.ndarray,
    m0: float = 1.0,
    percolation_limit: float = 0.5,
) -> float:
    r"""Return normalized 2-D Bruggeman transmissivity ``K / m0**(3/2)``.

    This follows the established ``fluid-percolation/plot_results.py``
    convention.  The input contains only the boundary-spanning component;
    other cells are zero.  The solved self-consistency equation is

    ``A_c * mean_c(h**3 / (h**3 + K_prime*m0**(3/2))) = p_c``.

    Here ``A_c`` is the conductive area fraction, ``mean_c`` averages over its
    positive gaps, and the standard 2-D threshold is ``p_c=0.5``.  The input
    array is not modified.
    """
    values = np.asarray(filtered_gap, dtype=np.float64)
    if values.ndim != 2 or np.any(values < 0.0) or not np.all(np.isfinite(values)):
        raise ValueError("filtered gap must be a finite, nonnegative 2-D array")
    if not np.isfinite(m0) or m0 <= 0.0:
        raise ValueError("m0 must be finite and positive")
    if not 0.0 < percolation_limit < 1.0:
        raise ValueError("percolation limit must lie between zero and one")

    positive = values[values > 0.0]
    conductive_area = float(positive.size / values.size)
    if conductive_area <= percolation_limit:
        return 0.0
    conductivity = positive**3
    lower_conductivity = float(np.min(conductivity))
    upper_conductivity = float(np.max(conductivity))
    if lower_conductivity <= 0.0 or upper_conductivity <= 0.0:
        return 0.0
    m0_32 = float(m0) ** 1.5
    log_lower = math.log(1e-6 * lower_conductivity / m0_32)
    log_upper = math.log(1e6 * upper_conductivity / m0_32)

    def residual(log_normalized_effective: float) -> float:
        normalized_effective = math.exp(log_normalized_effective)
        fraction = conductivity / (
            conductivity + normalized_effective * m0_32
        )
        return float(
            conductive_area * np.mean(fraction, dtype=np.float64)
            - percolation_limit
        )

    if residual(log_lower) < 0.0:
        return 0.0
    return float(
        math.exp(
            brentq(
                residual,
                log_lower,
                log_upper,
                xtol=1e-10,
                maxiter=200,
            )
        )
    )


def _load_surface(output: Path, task: SurfaceTask) -> np.ndarray:
    path = surface_path(output, task)
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; run the corresponding generate task first"
        )
    return np.load(path, mmap_mode="r", allow_pickle=False)


def run_simulation(
    task: SimulationTask,
    *,
    output: Path,
    solver: str,
    rtol: float,
    prescribed_mean_flux: float,
    force: bool = False,
) -> Path:
    """Run one pressure-driven or prescribed-mean-flux cell problem."""
    from reynoldsflow import transport

    result_path = simulation_result_path(output, task)
    if result_path.exists() and not force:
        print(f"skip existing {result_path}")
        return result_path
    if prescribed_mean_flux <= 0.0:
        raise ValueError("prescribed mean flux must be positive")

    surface_task = SurfaceTask(task.rolloff_ratio, task.realization)
    surface = _load_surface(output, surface_task)
    gap, threshold, actual_contact = gap_at_contact_fraction(
        surface, task.contact_fraction
    )
    m0 = float(np.var(surface, dtype=np.float64))
    m0_32 = m0**1.5
    mean_gap = float(np.mean(gap, dtype=np.float64))
    if mean_gap <= 0.0:
        raise RuntimeError("the truncated gap has zero mean separation")

    started = time.perf_counter()
    diagnostics = None
    if task.mode == "pressure":
        prepared = transport.prepare_fluid_problem(
            gap,
            boundary_mode="pressure",
            p_west=0.0,
            p_east=1.0,
        )
        unit_gradient = 1.0
        boundary_mode = "west/east Dirichlet; transverse periodic"
    elif task.mode == "mean-flux":
        # Linearity lets one unit-gradient solve determine the gradient needed
        # for any specified average flux.  Both directions are periodic here.
        prepared = transport.prepare_fluid_problem(
            gap,
            boundary_mode="periodic",
            pressure_gradient=1.0,
        )
        unit_gradient = 1.0
        boundary_mode = "fully periodic; prescribed macroscopic mean flux"
    else:
        raise ValueError(f"unknown simulation mode {task.mode!r}")

    if prepared is None:
        filtered = pressure = flux = None
        effective = 0.0
        total_flux = 0.0
        conservation_error = None
        converged_path = False
    else:
        filtered, pressure, flux, diagnostics = prepared.solve_with_diagnostics(
            gap, solver=solver, rtol=rtol
        )
        total_flux, conservation_error = transport.compute_total_flux(
            filtered,
            flux,
            gap.shape[0],
            boundary_mode="periodic" if task.mode == "mean-flux" else "pressure",
        )
        effective = float(total_flux / unit_gradient)
        converged_path = True
    elapsed = time.perf_counter() - started

    if task.mode == "mean-flux":
        required_gradient = (
            prescribed_mean_flux / effective if effective > 0.0 else None
        )
        resulting_flux = prescribed_mean_flux if effective > 0.0 else 0.0
    else:
        required_gradient = unit_gradient
        resulting_flux = total_flux

    payload = {
        "schema_version": SCHEMA_VERSION,
        "task": asdict(task),
        "surface_file": str(surface_path(output, surface_task)),
        "threshold_plane_height": threshold,
        "target_contact_fraction": task.contact_fraction,
        "actual_contact_fraction": actual_contact,
        "mean_gap": mean_gap,
        "surface_m0": m0,
        "normalization": "effective_transmissivity / surface_m0**(3/2)",
        "boundary_conditions": boundary_mode,
        "periodic_gauge": (
            "strongest-cell point gauge" if task.mode == "mean-flux" else None
        ),
        "solver": solver,
        "solver_used": diagnostics.solver if diagnostics is not None else None,
        "rtol": rtol,
        "solver_converged": (
            diagnostics.converged if diagnostics is not None else None
        ),
        "solver_iterations": (
            diagnostics.iterations if diagnostics is not None else None
        ),
        "solver_relative_residual": (
            diagnostics.relative_residual if diagnostics is not None else None
        ),
        "solver_convergence_reason": (
            diagnostics.convergence_reason if diagnostics is not None else None
        ),
        "percolating_or_winding_path": converged_path,
        "unit_solve_pressure_gradient": unit_gradient,
        "unit_solve_total_flux": total_flux,
        "effective_transmissivity": effective,
        "normalized_transmissivity": effective / m0_32,
        "prescribed_mean_flux": (
            prescribed_mean_flux if task.mode == "mean-flux" else None
        ),
        "required_mean_pressure_gradient": required_gradient,
        "resulting_mean_flux": resulting_flux,
        "relative_flux_conservation_error": conservation_error,
        "solve_seconds": elapsed,
    }
    _atomic_json(result_path, payload)
    print(f"wrote {result_path} in {elapsed:.2f} s")
    del filtered, pressure, flux, gap, prepared
    gc.collect()
    return result_path


def run_theory(
    task: TheoryTask, *, output: Path, force: bool = False
) -> Path:
    """Evaluate Bruggeman theory for one surface/load pair."""
    from reynoldsflow import transport

    result_path = theory_result_path(output, task)
    if result_path.exists() and not force:
        print(f"skip existing {result_path}")
        return result_path
    surface_task = SurfaceTask(task.rolloff_ratio, task.realization)
    surface = _load_surface(output, surface_task)
    gap, threshold, actual_contact = gap_at_contact_fraction(
        surface, task.contact_fraction
    )
    m0 = float(np.var(surface, dtype=np.float64))
    m0_32 = m0**1.5
    mean_gap = float(np.mean(gap, dtype=np.float64))
    started = time.perf_counter()
    filtered_gap = transport.connectivity_analysis(gap)
    if filtered_gap is None:
        conductive_area = 0.0
        normalized_effective = 0.0
    else:
        conductive_area = float(np.count_nonzero(filtered_gap > 0.0) / gap.size)
        normalized_effective = bruggeman_transmissivity(filtered_gap, m0=m0)
    effective = normalized_effective * m0_32
    elapsed = time.perf_counter() - started
    _atomic_json(
        result_path,
        {
            "schema_version": SCHEMA_VERSION,
            "task": asdict(task),
            "surface_file": str(surface_path(output, surface_task)),
            "threshold_plane_height": threshold,
            "target_contact_fraction": task.contact_fraction,
            "actual_contact_fraction": actual_contact,
            "mean_gap": mean_gap,
            "surface_m0": m0,
            "conductive_area_fraction": conductive_area,
            "normalization": "effective_transmissivity / surface_m0**(3/2)",
            "bruggeman_equation": (
                "A_c*mean_c(gap**3/(gap**3 + K_prime*m0**(3/2))) = 0.5"
            ),
            "effective_transmissivity": effective,
            "normalized_transmissivity": normalized_effective,
            "theory_seconds": elapsed,
        },
    )
    print(f"wrote {result_path} in {elapsed:.2f} s")
    return result_path


def _load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _mean_sem(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(tuple(values), dtype=float)
    mean = float(np.mean(array))
    sem = float(np.std(array, ddof=1) / math.sqrt(array.size)) if array.size > 1 else 0.0
    return mean, sem


def _rms_error(values: Iterable[float]) -> float:
    """Return the population RMS deviation from an ensemble mean."""
    array = np.asarray(tuple(values), dtype=float)
    mean = np.mean(array)
    return float(np.sqrt(np.mean((array - mean) ** 2)))


def plot_transmissivity_rms_error(
    *,
    output: Path,
    ratios: Sequence[int],
    realizations: int,
    contacts: Sequence[float],
    allow_incomplete: bool = False,
) -> list[Path]:
    """Plot realization-to-realization RMS error in normalized transmissivity."""
    import matplotlib.pyplot as plt

    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for contact in contacts:
        fig, axis = plt.subplots(figsize=(6.4, 4.5), constrained_layout=True)
        for mode, label, marker in (
            ("pressure", "Prescribed pressure", "o"),
            ("mean-flux", "Prescribed mean flux", "s"),
        ):
            errors = []
            for ratio in ratios:
                values = []
                for realization in range(realizations):
                    task = SimulationTask(ratio, realization, contact, mode)
                    path = simulation_result_path(output, task)
                    if path.exists():
                        values.append(
                            _load_json(path)["normalized_transmissivity"]
                        )
                    elif not allow_incomplete:
                        raise FileNotFoundError(f"missing result {path}")
                errors.append(_rms_error(values) if values else math.nan)
            axis.plot(ratios, errors, marker=marker, label=label)

        theory_errors = []
        for ratio in ratios:
            values = []
            for realization in range(realizations):
                task = TheoryTask(ratio, realization, contact)
                path = theory_result_path(output, task)
                if path.exists():
                    values.append(
                        _load_json(path)["normalized_transmissivity"]
                    )
                elif not allow_incomplete:
                    raise FileNotFoundError(f"missing theory result {path}")
            theory_errors.append(_rms_error(values) if values else math.nan)
        axis.plot(
            ratios,
            theory_errors,
            marker="^",
            linestyle="--",
            label="Bruggeman (2-D self-consistent)",
        )
        axis.set(
            xlabel=r"Roll-off ratio $k_1/k_0$",
            ylabel=r"RMS error in normalized transmissivity $K'$",
            title=f"Geometrical contact area = {100 * contact:.0f}%",
            xticks=list(ratios),
        )
        axis.set_yscale("log")
        axis.grid(True, alpha=0.3)
        axis.legend()
        stem = figure_dir / f"transmissivity_rms_error_{_contact_tag(contact)}"
        for suffix in (".png", ".pdf"):
            path = stem.with_suffix(suffix)
            fig.savefig(path, dpi=220, bbox_inches="tight")
            written.append(path)
            print(f"wrote {path}")
        plt.close(fig)
    return written


def plot_results(
    *,
    output: Path,
    ratios: Sequence[int],
    realizations: int,
    contacts: Sequence[float],
    allow_incomplete: bool = False,
) -> list[Path]:
    """Create normalized, unnormalized, and normalized semilog-x plots."""
    import matplotlib.pyplot as plt

    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    plot_fields = (
        (
            "normalized_transmissivity",
            r"Normalized transmissivity $K'=K/m_0^{3/2}$",
            "transmissivity",
            "linear",
            "log",
        ),
        (
            "normalized_transmissivity",
            r"Normalized transmissivity $K'=K/m_0^{3/2}$",
            "transmissivity_semilogx",
            "log",
            "linear",
        ),
        (
            "effective_transmissivity",
            r"Transmissivity $K$",
            "transmissivity_raw",
            "linear",
            "log",
        ),
    )
    for field, ylabel, stem_prefix, xscale, yscale in plot_fields:
        for contact in contacts:
            fig, axis = plt.subplots(
                figsize=(6.4, 4.5), constrained_layout=True
            )
            for mode, label, marker in (
                ("pressure", "Prescribed pressure", "o"),
                ("mean-flux", "Prescribed mean flux", "s"),
            ):
                means: list[float] = []
                sems: list[float] = []
                for ratio in ratios:
                    values = []
                    for realization in range(realizations):
                        task = SimulationTask(
                            ratio, realization, contact, mode
                        )
                        path = simulation_result_path(output, task)
                        if path.exists():
                            values.append(_load_json(path)[field])
                        elif not allow_incomplete:
                            raise FileNotFoundError(f"missing result {path}")
                    if not values:
                        means.append(math.nan)
                        sems.append(math.nan)
                    else:
                        mean, sem = _mean_sem(values)
                        means.append(mean)
                        sems.append(sem)
                axis.errorbar(
                    ratios,
                    means,
                    yerr=sems,
                    marker=marker,
                    capsize=3,
                    linewidth=1.5,
                    label=label,
                )

            theory_means: list[float] = []
            theory_sems: list[float] = []
            for ratio in ratios:
                values = []
                for realization in range(realizations):
                    task = TheoryTask(ratio, realization, contact)
                    path = theory_result_path(output, task)
                    if path.exists():
                        values.append(_load_json(path)[field])
                    elif not allow_incomplete:
                        raise FileNotFoundError(f"missing theory result {path}")
                if not values:
                    theory_means.append(math.nan)
                    theory_sems.append(math.nan)
                else:
                    mean, sem = _mean_sem(values)
                    theory_means.append(mean)
                    theory_sems.append(sem)
            axis.errorbar(
                ratios,
                theory_means,
                yerr=theory_sems,
                marker="^",
                linestyle="--",
                capsize=3,
                linewidth=1.5,
                label="Bruggeman (2-D self-consistent)",
            )
            axis.set(
                xlabel=r"Roll-off ratio $k_1/k_0$",
                ylabel=ylabel,
                title=f"Geometrical contact area = {100 * contact:.0f}%",
                xticks=list(ratios),
            )
            axis.set_xscale(xscale)
            axis.set_yscale(yscale)
            if xscale == "log":
                axis.set_xticks(
                    list(ratios), labels=[str(ratio) for ratio in ratios]
                )
            axis.grid(True, alpha=0.3)
            axis.legend()
            stem = figure_dir / f"{stem_prefix}_{_contact_tag(contact)}"
            for suffix in (".png", ".pdf"):
                path = stem.with_suffix(suffix)
                fig.savefig(path, dpi=220, bbox_inches="tight")
                written.append(path)
                print(f"wrote {path}")
            plt.close(fig)
    written.extend(
        plot_transmissivity_rms_error(
            output=output,
            ratios=ratios,
            realizations=realizations,
            contacts=contacts,
            allow_incomplete=allow_incomplete,
        )
    )
    return written


def _select_task(tasks: Sequence, task_id: int | None) -> Sequence:
    if task_id is None:
        return tasks
    if task_id < 0 or task_id >= len(tasks):
        raise ValueError(f"task id must be in [0, {len(tasks) - 1}]")
    return [tasks[task_id]]


def _common_study_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--rolloff-ratios",
        type=lambda text: _csv_numbers(text, int),
        default=DEFAULT_ROLLOFF_RATIOS,
        help="comma-separated k1/k0 values (default: 1,2,4,6,8)",
    )
    parser.add_argument("--realizations", type=int, default=DEFAULT_REALIZATIONS)
    parser.add_argument(
        "--contacts",
        type=lambda text: _csv_numbers(text, float),
        default=DEFAULT_CONTACT_FRACTIONS,
        help="comma-separated geometrical contact fractions",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="print the production task map")
    _common_study_arguments(plan)
    plan.add_argument("--size", type=int, default=DEFAULT_SIZE)

    generate = subparsers.add_parser("generate", help="generate rough surfaces")
    _common_study_arguments(generate)
    generate.add_argument("--size", type=int, default=DEFAULT_SIZE)
    generate.add_argument("--hurst", type=float, default=DEFAULT_HURST)
    generate.add_argument(
        "--short-wave-ratio", type=int, default=DEFAULT_SHORT_WAVE_RATIO
    )
    generate.add_argument(
        "--short-wave-reference",
        choices=("k0", "k1"),
        default="k0",
        help=(
            "interpret --short-wave-ratio as k2/k0 (default) or k2/k1"
        ),
    )
    generate.add_argument("--base-seed", type=int, default=DEFAULT_BASE_SEED)
    generate.add_argument("--task-id", type=int)
    generate.add_argument("--force", action="store_true")

    simulate = subparsers.add_parser("simulate", help="run Reynolds-flow tasks")
    _common_study_arguments(simulate)
    simulate.add_argument("--task-id", type=int)
    simulate.add_argument("--solver", default="auto")
    simulate.add_argument("--rtol", type=float, default=1e-12)
    simulate.add_argument("--prescribed-mean-flux", type=float, default=1.0)
    simulate.add_argument("--force", action="store_true")

    theory = subparsers.add_parser("theory", help="run Bruggeman tasks")
    _common_study_arguments(theory)
    theory.add_argument("--task-id", type=int)
    theory.add_argument("--force", action="store_true")

    plot = subparsers.add_parser("plot", help="plot ensemble averages")
    _common_study_arguments(plot)
    plot.add_argument("--allow-incomplete", action="store_true")
    return parser


def _validate_common(args: argparse.Namespace) -> None:
    if args.realizations < 1:
        raise ValueError("realizations must be positive")
    if not args.rolloff_ratios:
        raise ValueError("at least one roll-off ratio is required")
    if not args.contacts or any(not 0.0 < item < 1.0 for item in args.contacts):
        raise ValueError("all contact fractions must lie between zero and one")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _validate_common(args)
    output = args.output.resolve()

    if args.command == "plan":
        surfaces = surface_tasks(args.rolloff_ratios, args.realizations)
        simulations = simulation_tasks(
            args.rolloff_ratios, args.realizations, args.contacts
        )
        theories = theory_tasks(
            args.rolloff_ratios, args.realizations, args.contacts
        )
        surface_gib = args.size**2 * np.dtype(np.float32).itemsize / 2**30
        print(f"output: {output}")
        print(f"surface tasks: {len(surfaces)} (IDs 0..{len(surfaces)-1})")
        print(
            f"simulation tasks: {len(simulations)} "
            f"(IDs 0..{len(simulations)-1})"
        )
        print(f"theory tasks: {len(theories)} (IDs 0..{len(theories)-1})")
        print(f"surface storage: about {len(surfaces) * surface_gib:.2f} GiB")
        for task_id, task in enumerate(simulations):
            print(f"simulate {task_id:03d}: {task}")
        return 0

    if args.command == "generate":
        tasks = _select_task(
            surface_tasks(args.rolloff_ratios, args.realizations), args.task_id
        )
        for task in tasks:
            generate_surface(
                task,
                output=output,
                size=args.size,
                hurst=args.hurst,
                short_wave_ratio=args.short_wave_ratio,
                short_wave_reference=args.short_wave_reference,
                base_seed=args.base_seed,
                force=args.force,
            )
        return 0

    if args.command == "simulate":
        tasks = _select_task(
            simulation_tasks(
                args.rolloff_ratios, args.realizations, args.contacts
            ),
            args.task_id,
        )
        for task in tasks:
            run_simulation(
                task,
                output=output,
                solver=args.solver,
                rtol=args.rtol,
                prescribed_mean_flux=args.prescribed_mean_flux,
                force=args.force,
            )
        return 0

    if args.command == "theory":
        tasks = _select_task(
            theory_tasks(args.rolloff_ratios, args.realizations, args.contacts),
            args.task_id,
        )
        for task in tasks:
            run_theory(task, output=output, force=args.force)
        return 0

    if args.command == "plot":
        plot_results(
            output=output,
            ratios=args.rolloff_ratios,
            realizations=args.realizations,
            contacts=args.contacts,
            allow_incomplete=args.allow_incomplete,
        )
        return 0
    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(2) from error
