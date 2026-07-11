# Direct-solver performance diagnosis

## Conclusion

There is no evidence that the corrected ReynoldsFlow matrix implementation
caused a direct-solver regression. The old and current tables were not
like-for-like. Three effects explain the apparent slowdown:

1. The historical table was probably generated with the `2048 x 2048`
   self-affine rough-contact sweep, whereas the July 2026 reproduction used a
   `2000 x 2000` centered circle. Sparse-direct cost depends on elimination
   fill and separator structure, not just the nominal grid dimension.
2. The historical run did not control native threads. The reproducible July
   table deliberately used one thread. Restoring 20 threads reduces current
   Pardiso from 11.80 s to 9.47 s on the circle and to 9.44 s on the
   reconstructed rough case, close to the historical 8.53 s.
3. The current `fluidpaper` CHOLMOD and PETSc/MUMPS builds link against
   reference Netlib BLAS/LAPACK. They do not have an optimized parallel BLAS.
   The historical binary environment was not recorded and no longer exists.

The corrected matrix is smaller (active DOFs only), symmetric positive
definite, and passed to Pardiso using the documented one-triangle SPD storage.
Reverting to the historical nonsymmetric-boundary matrix or its Pardiso
configuration would trade correctness for an invalid comparison.

## Evidence from git history

Commit `1bf0f09` added the README timing table. Its parent configured
`tests/test_solvers.py` with:

- `N0 = 2048`;
- seed `23349`;
- a periodic self-affine field with `Hurst = 0.5`;
- `k_low = 1/N0`, `k_high = 12/N0`;
- contact offset `delta = 0.3`.

The timing commit changed `N0` back to 128 immediately after adding the table.
That is stronger provenance evidence than the README's rounded description of
"2000 x 2000". The same README showed a circle only as its minimal usage
example; it did not record the timing geometry explicitly.

The old test timed `time.time()` around the entire public solve. Numba kernels
were compiled automatically at module import, so its values are best compared
with current steady end-to-end totals, not current cold totals.

## Controlled results

All current values use `fluidpaper`, `rtol=1e-8`, and the compact corrected
operator. Times are steady end-to-end wall times in seconds.

| Backend | Historical, environment unknown | Circle 2000, 1 thread | Reconstructed rough 2048, 1 thread | Reconstructed rough 2048, 20 threads |
|---|---:|---:|---:|---:|
| Pardiso | 8.53 | 11.80 | 12.57 | 9.44 |
| CHOLMOD | 20.61 | 26.23 | 23.67 | not beneficial on current Netlib BLAS build |
| PETSc/MUMPS | 26.14 | 39.57 | 31.28 | not materially beneficial in a one-rank run |

The reconstructed rough field has 3,121,307 active DOFs (74.4% of the grid)
and 15,593,449 matrix nonzeros. The circle has 3,497,348 active DOFs (87.4%)
and 17,479,540 nonzeros. The smaller rough elimination graph particularly
reduces MUMPS and CHOLMOD time.

Pardiso carries its own oneMKL runtime and responds to `MKL_NUM_THREADS`.
Twenty threads are not universally optimal on the hybrid i7-13700H, but they
show that thread policy accounts for most of Pardiso's apparent regression.

## Current binary environment

The July 2026 measurements ran on an Intel Core i7-13700H (20 logical CPUs,
24 MiB L3). The nearby historical README scaling results mention a Xeon
Platinum 8488C, but do not prove that every row in the small table used that
machine. Consequently hardware-normalized claims cannot be made.

The `fluidpaper` environment contains:

- `libblas 3.11.0 *_netlib` and `liblapack 3.11.0 *_netlib`;
- CHOLMOD 5.3.1 / SuiteSparse 7.10.1 linked to that `libblas.so`;
- PETSc 3.24.5 configured with `BLASLAPACK_LIB = -llapack -lblas`;
- MUMPS 5.8.2 running through PETSc with MPI world size 1;
- pypardiso 0.4.7 using its separate oneMKL runtime.

This distinction matters because CHOLMOD and MUMPS perform dense BLAS-3 work
inside sparse frontal matrices. Their official documentation recommends an
optimized BLAS and warns that a single-threaded/reference BLAS can be much
slower. Merely raising `OMP_NUM_THREADS` cannot make reference Netlib BLAS into
a parallel optimized implementation.

## Reproducibility policy

Future direct-solver reports must record:

- exact deterministic case and active graph statistics;
- CPU model and logical/physical topology;
- cold and steady timing separately;
- native thread variables and affinity policy;
- BLAS/LAPACK implementation;
- PETSc, MUMPS, SuiteSparse, scikit-sparse, and pypardiso versions;
- matrix DOFs, nonzeros, factor memory where available, and process peak RSS.

Compare direct backends only on identical matrices and binary environments.
For the production rough-contact workflow, PETSc/Hypre remains the preferred
scalable solver; direct methods are secondary references for moderate sizes.

## Primary references

- [PETSc: use of BLAS/LAPACK in PETSc and external packages](https://petsc.org/main/manual/blas-lapack/)
- [Official SuiteSparse build and BLAS/OpenMP guidance](https://github.com/DrTimothyAldenDavis/SuiteSparse)
- [Intel oneMKL Pardiso parameter reference](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-c/2025-0/pardiso-iparm-parameter.html)
- [Intel oneMKL Pardiso symmetric-storage reference](https://www.intel.com/content/www/us/en/docs/onemkl/developer-reference-fortran/2023-1/onemkl-pardiso-parameters-in-tabular-form.html)
