# Cartesian circle solver benchmark — 2000 x 2000

Environment: `fluidpaper`, Python 3.12.13, one controlled native thread,
`rtol=1e-8`. The deterministic centered-circle case has 3,497,348 active
degrees of freedom and 17,479,540 matrix nonzeros. Each solver ran in an
isolated process. The cold total includes first imports, native runtime
initialization, and Numba compilation; the steady value is the second complete
run.

The historical values come from the original README and were produced on a
different, insufficiently recorded software/hardware environment. They are
included for orientation, not treated as a strict regression threshold.

| Solver | Historical total (s) | Current cold total (s) | Current steady total (s) | Steady linear solve (s) | Iterations | Peak RSS (GiB) |
|---|---:|---:|---:|---:|---:|---:|
| `petsc-cg.hypre` | 4.46 | 8.656 | **4.241** | **3.952** | 7 | 2.60 |
| `petsc-cg.gamg` | 11.96 | 10.966 | 7.947 | 7.641 | 16 | 2.19 |
| `scipy.amg-smooth_aggregation` | 15.48 | 10.126 | 9.162 | 8.828 | 13 | 2.45 |
| `pardiso` | 8.53 | 13.061 | 11.804 | 11.491 | direct | 3.59 |
| `scipy.amg-rs` | 8.96 | 13.380 | 12.503 | 12.161 | 8 | 2.09 |
| `cholesky` | 20.61 | 27.747 | 26.234 | 25.863 | direct | 2.57 |
| `petsc-mumps` | 26.14 | 44.180 | 39.570 | 39.237 | direct | 4.78 |
| `scipy-spsolve` | — | 111.521 | 109.248 | 108.884 | direct | 8.94 |
| `petsc-gmres.ilu` | 134.98\* | — | **two-run worker exceeded 600 s** | — | — | — |

\* The historical row was labeled `petsc-cg.ilu`. The current canonical solver
is GMRES+ILU because ILU does not preserve the symmetry contract required by
CG, so this row is not an algorithm-for-algorithm comparison.

All eight completed solvers passed their algebraic residual checks and agreed
on total flux near `0.776450225`. Hypre most closely reproduces the historical
best result: 4.241 s steady today versus 4.46 s historically. Direct sparse
factorizations show superlinear time and memory growth. The GMRES/ILU timeout
confirms that it is not a viable production method for this problem size.

Raw reports:

- `cartesian-circle-2000-fluidpaper-2026-07-11.json`: results through GAMG;
  GMRES/ILU then timed out before the original suite could append its status.
- `cartesian-circle-2000-petsc-mumps-fluidpaper-2026-07-11.json`: separately
  resumed MUMPS result with identical benchmark settings.
