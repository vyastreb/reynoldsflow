# Changelog

## 0.1.1 — 2026-07-12

### Performance and documentation

- Extended the deterministic rough-contact CPU/RAM scaling dataset from
  `4096 x 4096` to `6144 x 6144` grid points (23.3 million active DOFs).
- Recorded solver-specific capacity limits: SuperLU's projected memory wall,
  MUMPS's `6144 x 6144` `SIGKILL`, and Ruge–Stuben's 5400 s timeout.
- Clarified that steady end-to-end timings rebuild preconditioners or direct
  factors on every run while excluding only first-process initialization.
- Regenerated the release figure, compact CSV, performance report, citation,
  and PyPI metadata for version 0.1.1.

## 0.1.0 — 2026-07-12

### Numerical correctness

- Replaced the Cartesian boundary and flux treatment with a consistent
  cell-centered finite-volume formulation.
- Added conservative Cartesian and polar face fluxes derived from the same
  conductances used by matrix assembly.
- Row-scaled the polar operator to a symmetric stored form and corrected
  annular/symmetry-sector flux integration.
- Removed implicit polar channel dilation; dilation is now an explicit geometry
  modification.
- Retained every independent spanning component across periodic seams.
- Tightened the default iterative tolerance to `1e-12` after direct total-flow
  calibration on constricted rough fields.

### Sparse systems and performance

- Added compact active-cell DOF mappings and eliminated blocked identity rows.
- Replaced overallocated COO triplets with exact-size two-pass CSR assembly.
- Removed unused Cartesian coordinate grids and automatic import-time Numba
  compilation.
- Added deterministic staged, scaling, and subprocess-isolated solver
  benchmarks with separate cold and steady timing.
- Added machine, BLAS, native thread, backend-version, residual, iteration,
  conservation, and peak-RSS metadata to benchmark reports.
- Added the reproducible v0.1.0 rough-contact CPU/RAM scaling dataset through
  `4096 x 4096` grid points (10.7 million active DOFs) and its generated
  figure.

### Solvers and API

- Added a shared solver registry with explicit aliases and diagnostics.
- Implemented `scipy-spsolve` as a portable direct reference backend.
- Made `auto` reliably select base SciPy/PyAMG; native backends are explicit.
- Added explicit invalid-input, unavailable-backend, unknown-solver, and
  convergence errors. `None` results are reserved for normal no-percolation.
- Added configurable Cartesian `p_west` and `p_east` reservoir pressures.
- Corrected the Pardiso SPD interface to pass one matrix triangle and release
  its one-shot factor workspace.
- Removed the unsupported PETSc ILU configuration and misleading legacy alias.

### Tests, packaging, and documentation

- Added analytical and discrete conservation tests, compact/full-grid
  agreement, matrix symmetry, periodic connectivity, multiple-channel,
  validation, solver-diagnostic, and native-backend coverage.
- Isolated optional native-backend tests and benchmarks in subprocesses so a
  broken MPI/MKL runtime cannot abort the main test process.
- Added Python 3.10–3.12 CI, modern SPDX/package metadata, backend-specific
  optional extras, and source/wheel release checks.
- Rewrote the README around the numerical model, verified capabilities,
  solver tradeoffs, limitations, and reproducible v0.1.0 results.
- Added PyPI-compatible absolute image URLs, the ReynoldsFlow logo, and
  versioned release assets.
- Removed stale `fluxflow` metadata and verified-dead numerical kernels.
