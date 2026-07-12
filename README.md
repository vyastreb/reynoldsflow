<p align="center">
  <img src="https://raw.githubusercontent.com/vyastreb/reynoldsflow/master/extras/logo.png" alt="ReynoldsFlow logo" width="560">
</p>

# ReynoldsFlow

<p align="center">
  <a href="https://pypi.org/project/reynoldsflow/"><img src="https://img.shields.io/pypi/v/reynoldsflow.svg" alt="PyPI version"></a>
  <a href="https://pypi.org/project/reynoldsflow/"><img src="https://img.shields.io/pypi/pyversions/reynoldsflow.svg" alt="Supported Python versions"></a>
  <a href="https://github.com/vyastreb/reynoldsflow/actions/workflows/tests.yml"><img src="https://github.com/vyastreb/reynoldsflow/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://opensource.org/licenses/BSD-3-Clause"><img src="https://img.shields.io/badge/License-BSD_3--Clause-blue.svg" alt="BSD-3-Clause license"></a>
</p>

ReynoldsFlow is a finite-volume Python solver for steady incompressible flow in
thin gaps with contact and complex percolating geometry. It solves Cartesian
and polar forms of the Reynolds equation, retains all boundary-spanning fluid
components, assembles only active degrees of freedom, and reconstructs
conservative face fluxes from the same conductances used by the linear system.

The package is intended for rough-contact leakage calculations and related
elliptic transport problems. Version 0.1.0 emphasizes numerical conservation,
explicit convergence diagnostics, deterministic benchmarks, and safe optional
solver selection.

## Mathematical model

For an isoviscous incompressible fluid with local gap `g` and pressure `p`, the
dimensionless Cartesian problem is

```text
div(g³ grad(p)) = 0.
```

The Cartesian discretization is cell-centered on the unit square. Axis 0 is
the transport direction, axis 1 is periodic, and reservoir pressures are
applied at the west and east boundary faces. Internal face conductivity is the
harmonic mean of adjacent `g³` values. Cells with `g = 0` are impermeable;
negative or non-finite gaps are invalid input.

The polar solver discretizes the corresponding conservative annular operator
on `(r, theta)`, with configurable radial pressures and periodic or symmetry
angular boundary conditions.

## Main features

- Conservative Cartesian and polar finite-volume discretizations.
- Periodic connectivity analysis retaining every independent spanning channel.
- Compact sparse systems with one unknown per active percolating cell.
- Exact-size two-pass CSR assembly accelerated with Numba.
- Conservative face flux, total-flow integration, and conservation diagnostics.
- Direct and AMG-preconditioned iterative solvers with checked true residuals.
- Explicit errors for invalid input, unavailable backends, unknown solvers,
  breakdown, and non-convergence.
- Deterministic correctness and performance benchmarks.

## Installation

The portable installation includes SciPy and PyAMG:

```bash
pip install reynoldsflow
```

Optional native backends can be installed separately:

```bash
pip install "reynoldsflow[pardiso]"   # pypardiso / oneMKL
pip install "reynoldsflow[cholesky]" # scikit-sparse / CHOLMOD
pip install "reynoldsflow[petsc]"    # petsc4py binding
pip install "reynoldsflow[solvers]"  # request every optional Python binding
```

PETSc, MUMPS, Hypre, SuiteSparse, and oneMKL are native libraries. The PETSc
installation itself must have been built with the requested Hypre or MUMPS
support. Conda or a system/HPC package manager is often more reliable than
building these stacks through pip. Optional native backends are selected
explicitly; `solver="auto"` does not probe them because a broken MPI or MKL
installation may terminate below Python before a fallback is possible.

For plotting examples and development tools:

```bash
pip install "reynoldsflow[plot]"
pip install "reynoldsflow[dev]"
```

## Quick start

This example computes flow around a circular impermeable inclusion:

```python
import matplotlib.pyplot as plt
import numpy as np

from reynoldsflow import transport

n = 256
coordinate = (np.arange(n, dtype=float) + 0.5) / n
x, y = np.meshgrid(coordinate, coordinate, indexing="ij")
gaps = (np.hypot(x - 0.5, y - 0.5) > 0.2).astype(float)

filtered_gaps, pressure, flux = transport.solve_fluid_problem(
    gaps,
    solver="auto",
    p_west=0.0,
    p_east=1.0,
)

if pressure is not None:
    total_flux, conservation_error = transport.compute_total_flux(
        filtered_gaps, flux, n
    )
    print(f"Q = {total_flux:.8g}")
    print(f"relative conservation error = {conservation_error:.3e}")
    plt.imshow(pressure.T, origin="lower", extent=(0, 1, 0, 1))
    plt.colorbar(label="pressure")
    plt.show()
```

`solve_fluid_problem` returns `(filtered_gaps, pressure, flux)`. A normal
non-percolating field returns `(None, None, None)`; invalid data and solver
failures raise explicit exceptions. The default iterative tolerance is
`rtol=1e-12`.

For annular domains, use
`reynoldsflow.transport_polar.solve_fluid_problem_polar`.

## Linear solvers

No backend is universally optimal. Sparse-direct methods are attractive as
moderate-size references but exhibit fill-driven time and memory growth.
Multigrid-preconditioned CG methods have higher setup costs and generally offer
the better route to large systems.

| Solver string | Method and preconditioner | Dependency | Advantages | Limitations / best use |
|---|---|---|---|---|
| `auto` | CG + Ruge–Stuben AMG | SciPy, PyAMG | Portable and safe default; no native optional stack | Same implementation as `scipy.amg-rs`; slow on the largest rough-contact cases |
| `scipy.amg-rs` | CG + Ruge–Stuben AMG | SciPy, PyAMG | Low memory; portable; explicit iterations and residual | Reached 531 s at `5120²` and exceeded the `6144²` timeout |
| `scipy.amg-smooth_aggregation` | CG + smoothed-aggregation AMG | SciPy, PyAMG | Portable alternative; competitive on the largest release case | More hierarchy memory; performance is problem-dependent |
| `scipy-spsolve` | Sparse LU (SuperLU) | SciPy | Robust direct reference available in the base install | Used 23.1 GiB at `4096²`; intended for small/moderate diagnostics |
| `cholesky` | Sparse Cholesky (CHOLMOD) | scikit-sparse, SuiteSparse | Very fast direct reference for the SPD operator | Native install; fill growth; performance depends strongly on BLAS |
| `pardiso` | Sparse Cholesky (oneMKL Pardiso) | pypardiso, oneMKL | Accurate direct solve; tunable shared-memory parallelism | Native runtime; thread-sensitive; factor memory grows rapidly |
| `petsc-cg.hypre` | CG + Hypre BoomerAMG | PETSc, Hypre | Few iterations; strongest large-scale option in current tests | PETSc/MPI installation and cold initialization cost |
| `petsc-cg.gamg` | CG + PETSc GAMG | PETSc | PETSc-native algebraic multigrid | Iteration count and setup are more problem-dependent than Hypre |
| `petsc-mumps` | Multifrontal direct solve | PETSc, MUMPS | Accurate direct reference inside PETSc | Used 16.8 GiB at `5120²`; `6144²` was killed under memory pressure |

Explicit native backends never silently fall back to another algorithm.

## Reproducible v0.1.0 performance

The following figure replaces the legacy performance plots as the release
baseline. It uses the deterministic `rough-contact` case at `256²`, `512²`,
`1024²`, `2048²`, `4096²`, `5120²`, and `6144²`; compact active-DOF systems;
`rtol=1e-12`; and one native thread. Every solver ran in an isolated process.
Runtime is the median of two fresh-solver, steady-process, end-to-end runs
after one cold run. It includes matrix assembly, preconditioner construction
or direct factorization, solution, and flux postprocessing. Peak RSS includes
imports, native runtime state, JIT state, the gap field, matrix, solver data,
and outputs.

![Rough-contact solver runtime and memory scaling](https://raw.githubusercontent.com/vyastreb/reynoldsflow/master/docs/img/rough_contact_solver_scaling_v0.1.0.png)

The `6144²` case contained 23,301,121 active DOFs and 116,378,457 matrix
nonzeros. Five backends completed it:

| Solver | Steady end-to-end time (s) | Peak RSS (GiB) | Iterations |
|---|---:|---:|---:|
| `petsc-cg.hypre` | 52.18 | 15.85 | 16 |
| `pardiso` | 72.40 | 21.22 | direct |
| `scipy.amg-smooth_aggregation` | 111.90 | 15.05 | 26 |
| `cholesky` | 153.52 | 14.36 | direct |
| `petsc-cg.gamg` | 214.16 | 13.40 | 156 |

Curves stop at each backend's largest successful size. SuperLU stopped at
`4096²` (23.10 GiB; `5120²` was projected above available RAM). MUMPS passed
`5120²` (101.26 s, 16.77 GiB) but received `SIGKILL` at `6144²`. Ruge–Stuben
passed `5120²` (531.47 s, 8.26 GiB) but exceeded the 5400 s process timeout at
`6144²`. Failed or unattempted points are not plotted as timings.

These values describe one matrix family and one binary environment; they are
not universal backend rankings.

### Benchmark host and numerical stack

- Machine: 13th Gen Intel Core i7-13700H, 14 physical cores / 20 logical CPUs,
  32 GiB RAM, Linux 6.8 x86-64.
- Python 3.12.13; NumPy 2.4.2; SciPy 1.17.1; Numba 0.64.0;
  scikit-image 0.26.0; PyAMG 5.3.0.
- BLAS/LAPACK: conda-forge Netlib 3.11.0 for SciPy, CHOLMOD, PETSc, and MUMPS.
- Pardiso: pypardiso 0.4.7 with Intel oneMKL 2025.3.
- CHOLMOD: scikit-sparse 0.4.16 with SuiteSparse 7.10.1.
- PETSc stack: petsc4py 3.24.4, PETSc 3.24.5, OpenMPI 5.0.10,
  Hypre 3.1.0, and MUMPS 5.8.2; one MPI rank.
- `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`.

Commands and full stage definitions are documented in
[`benchmarks/README.md`](https://github.com/vyastreb/reynoldsflow/blob/master/benchmarks/README.md).
The compact-system and tolerance validation is summarized in
[`docs/performance-0.1.0.md`](https://github.com/vyastreb/reynoldsflow/blob/master/docs/performance-0.1.0.md).

## Numerical validation

The test suite covers analytical Cartesian and polar solutions, matrix
symmetry, conservative section fluxes, periodic connectivity, multiple
spanning channels, compact/full-grid agreement, prepared-topology safeguards,
solver diagnostics, and subprocess-isolated native backends.

```bash
python -m pytest -q
python -m pytest -q --run-backend tests/integration/test_optional_backends.py
```

Performance is never accepted without checking the true algebraic residual,
finite pressure, total flow, and boundary-flux conservation.

## Scope and limitations

- Steady, incompressible, isoviscous lubrication flow.
- Prescribed immobile gap field; no fluid–structure coupling.
- No cavitation, compressibility, inertia, or transient storage model.
- Cartesian public solves currently require square arrays.
- Native backend availability and performance depend on the local binary stack.

## Illustrations

The following earlier large rough-contact simulations are retained as
qualitative examples; they are not part of the v0.1.0 benchmark baseline.

![Rough-contact flow on an 8000 by 8000 grid](https://raw.githubusercontent.com/vyastreb/reynoldsflow/master/docs/img/illustration.jpg)

![Rough-contact flow on a 20000 by 20000 grid](https://raw.githubusercontent.com/vyastreb/reynoldsflow/master/docs/img/illustration_2.jpg)

## Project information

### Citation

If ReynoldsFlow contributes to scientific work, please cite the software using
the metadata in
[`CITATION.cff`](https://github.com/vyastreb/reynoldsflow/blob/master/CITATION.cff).

### Credits

- Author: Vladislav A. Yastrebov, CNRS, Mines Paris – PSL, Centre des Matériaux.
- Development period: September 2025 – July 2026.
- License: BSD 3-Clause; see
  [`LICENSE`](https://github.com/vyastreb/reynoldsflow/blob/master/LICENSE).
- Repository: [github.com/vyastreb/reynoldsflow](https://github.com/vyastreb/reynoldsflow).
- Changelog:
  [`CHANGELOG.md`](https://github.com/vyastreb/reynoldsflow/blob/master/CHANGELOG.md).

AI-assisted development is acknowledged for Cursor and GitHub Copilot;
ChatGPT 4o and 5; Claude 3.7, 4, and 4.5; and OpenAI Codex (GPT-5). Codex
assisted with the numerical audit, regression tests, benchmark methodology,
documentation, and v0.1.0 release engineering. Scientific and numerical
decisions and all generated code were reviewed by the author.
