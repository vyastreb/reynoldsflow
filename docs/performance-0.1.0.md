# ReynoldsFlow 0.1.0 numerical and performance report

Date: 2026-07-11

This report defines the reproducible release baseline. Historical benchmark
tables and figures remain in the repository for provenance but are not used to
assess version 0.1.0.

## Benchmark environment

- 13th Gen Intel Core i7-13700H, 14 physical cores / 20 logical CPUs, 32 GiB
  RAM, Linux 6.8 x86-64.
- Python 3.12.13, NumPy 2.4.2, SciPy 1.17.1, Numba 0.64.0,
  scikit-image 0.26.0, and PyAMG 5.3.0.
- Netlib BLAS/LAPACK 3.11.0 for SciPy, CHOLMOD, PETSc, and MUMPS.
- pypardiso 0.4.7 with Intel oneMKL 2025.3.
- scikit-sparse 0.4.16 with SuiteSparse 7.10.1.
- petsc4py 3.24.4, PETSc 3.24.5, OpenMPI 5.0.10, Hypre 3.1.0, and
  MUMPS 5.8.2; one MPI rank.
- `OMP_NUM_THREADS=MKL_NUM_THREADS=OPENBLAS_NUM_THREADS=1`.

## Method

The deterministic Cartesian `rough-contact` case was run at `256²`, `512²`,
`1024²`, `2048²`, and `4096²` with seed 23349, compact active-DOF assembly,
and `rtol=1e-12`. Every backend ran in a separate subprocess. Each process
made one cold run followed by two steady runs.

Reported runtime is the median steady end-to-end wall time: connectivity,
assembly, sparse-format conversion, solver setup/factorization, linear solve,
face flux, cell flux, and boundary integration. Case generation is excluded.
Peak RSS is the process high-water mark and includes imports, native/JIT state,
input/output arrays, sparse matrices, preconditioners or factors, and all
completed stages.

![Rough-contact runtime and memory scaling](img/rough_contact_solver_scaling_v0.1.0.png)

The exact plotted data are stored in
[`benchmarks/results/rough-contact-scaling-v0.1.0.csv`](../benchmarks/results/rough-contact-scaling-v0.1.0.csv).

## Largest release case

The `4096²` case retained 10,722,930 active DOFs (63.9% of the grid) and
53,531,200 matrix nonzeros.

| Solver | Steady total (s) | Linear stage (s) | Peak RSS (GiB) | Iterations |
|---|---:|---:|---:|---:|
| `petsc-cg.hypre` | 24.13 | 22.59 | 7.42 | 16 |
| `cholesky` | 30.28 | 28.71 | 7.53 | direct |
| `pardiso` | 31.71 | 30.13 | 9.64 | direct |
| `scipy.amg-smooth_aggregation` | 63.01 | 61.43 | 7.02 | 38 |
| `petsc-mumps` | 71.32 | 69.79 | 12.13 | direct |
| `petsc-cg.gamg` | 74.46 | 72.92 | 6.27 | 109 |
| `scipy-spsolve` | 153.50 | 151.89 | 23.10 | direct |
| `scipy.amg-rs` | 224.59 | 223.01 | 5.95 | 50 |

All backends completed and passed the true algebraic residual check. On the
largest case, the maximum true relative residual was `7.70e-13` and the
maximum boundary-flux conservation error was `4.81e-10`. Across the full
five-size sweep, the corresponding maxima were `8.54e-13` and `7.72e-8`.

These results are specific to this matrix family and binary environment.
Direct solvers are competitive at this scale, but their factor fill is
superlinear. Hypre is the preferred PETSc configuration for larger systems;
PyAMG remains the portable base-dependency path.

## Correctness changes relative to the pre-0.1 code

| Regression case | Previous behavior | Version 0.1.0 |
|---|---:|---:|
| Cartesian unit gap, `24 x 24`, expected `Q=1` | `Q=1.92` | `Q=1.0000000000000018` |
| Cartesian unit-gap conservation | approximately `4e-11` in an iterative smoke test | `2.89e-15` with direct reference |
| Polar unit gap, `16 x 32`, boundary conservation error | `4.96e-2` | `1.82e-14` |
| Polar `24 x 48`, analytical total-flow relative error | inconsistent postprocessing | `8.5e-5` |

The Cartesian and polar stored operators are symmetric after boundary
elimination and polar row scaling. Flux reconstruction uses the same harmonic
face conductances as matrix assembly.

Compact active-DOF and full-grid reference systems agree on active-cell
pressure to approximately `3e-13` in deterministic regression tests. The
compact system has exactly one unknown per retained positive spanning cell.

## Tolerance calibration

On the compact `512²` rough case, the direct reference produced
`Q=0.002964149471992414`.

| Iterative tolerance | Total flow | Relative flow error | Conservation error |
|---:|---:|---:|---:|
| `1e-10` | `0.002964152415423008` | approximately `9.9e-7` | `1.98e-6` |
| `1e-12` | `0.002964149475495322` | approximately `1.2e-9` | `1.61e-9` |

This sensitivity motivates the version 0.1.0 default `rtol=1e-12`.

## Reproduction

```bash
for n in 256 512 1024 2048 4096; do
  python -m benchmarks.benchmark_suite \
    --case rough-contact --size "$n" --rtol 1e-12 \
    --repeat 3 --threads 1 \
    --output "benchmarks/results/rough-contact-${n}.json"
done

python -m benchmarks.plot_solver_scaling \
  benchmarks/results/rough-contact-{256,512,1024,2048,4096}.json \
  --output docs/img/rough_contact_solver_scaling_v0.1.0.png \
  --csv-output benchmarks/results/rough-contact-scaling-v0.1.0.csv
```

Raw subprocess reports are ignored because they contain bulky repeated data;
the compact CSV and generated figure are versioned release artifacts.
