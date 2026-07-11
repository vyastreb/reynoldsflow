# Changelog

## 0.1.0 — 2026-07-10

### Numerical correctness

- Replaced Cartesian boundary and flux postprocessing with a consistent
  cell-centered finite-volume formulation.
- Added conservative face fluxes in Cartesian and polar geometries.
- Row-scaled the polar operator to a symmetric stored form.
- Corrected polar sector quadrature and single-sample angular extent.
- Removed implicit polar channel dilation; dilation is now an explicit option.
- Retained every independent spanning component across periodic seams.
- Tightened the default iterative tolerance to `1e-12` based on direct
  total-flow comparisons on constricted fields.

### Performance

- Removed unused full-grid Cartesian coordinate allocations and import-time JIT
  compilation.
- Added compact active-cell DOF mappings for partially blocked domains.
- Replaced overallocated COO triplets and format conversion with exact-size,
  two-pass CSR assembly.
- Added prepared Cartesian and polar problems for fixed-topology sequences.
- Added opt-in AMG hierarchy reuse with convergence and residual checks.

### Solvers and API

- Added a shared solver registry with explicit aliases and diagnostics.
- Implemented `scipy-spsolve` as a portable direct reference backend.
- Made `auto` reliably select base SciPy/PyAMG; native backends are explicit.
- Added explicit invalid-input, unavailable-backend, unknown-solver, and
  convergence errors. `None` results are reserved for no percolation.
- Added finite input and geometry validation.
- Added configurable Cartesian `p_west` and `p_east` reservoir pressures while
  preserving `0` and `1` defaults.

### Development

- Added deterministic staged and evolution benchmarks.
- Added safe pytest markers and subprocess-isolated native-backend tests.
- Removed stale `fluxflow` package metadata and legacy numerical kernels.
