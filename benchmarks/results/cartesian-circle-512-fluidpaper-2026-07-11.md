# Cartesian circle solver benchmark — 512 x 512

Environment: `fluidpaper`, Python 3.12.13, one controlled native thread,
`rtol=1e-8`. The deterministic circle case has 229,216 active degrees of
freedom and 1,144,240 matrix nonzeros. Each solver ran in an isolated process.
The cold total includes first imports, native runtime initialization, and Numba
compilation. The steady values are medians of the remaining two runs.

| Solver | Cold total (s) | Steady total (s) | Steady linear solve (s) | Iterations |
|---|---:|---:|---:|---:|
| `petsc-cg.hypre` | 4.557 | **0.228** | **0.208** | 6 |
| `scipy.amg-rs` | 1.280 | 0.262 | 0.243 | 6 |
| `cholesky` | 1.361 | 0.374 | 0.350 | direct |
| `petsc-cg.gamg` | 3.295 | 0.422 | 0.402 | 13 |
| `scipy.amg-smooth_aggregation` | 1.486 | 0.469 | 0.451 | 12 |
| `pardiso` | 1.800 | 0.659 | 0.639 | direct |
| `petsc-mumps` | 5.546 | 1.207 | 1.186 | direct |
| `scipy-spsolve` | 2.718 | 1.707 | 1.684 | direct |

All listed solvers passed their algebraic residual checks.
