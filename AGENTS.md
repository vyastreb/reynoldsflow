# ReynoldsFlow repository guide

## Purpose and scope

This repository implements finite-difference solvers for the incompressible Reynolds/thin-film equation

```text
div(g^3 grad(p)) = 0
```

on Cartesian and annular polar grids. `g <= 0` denotes solid/contact; positive `g` is an open fluid gap. The code first keeps a boundary-to-boundary percolating component, then assembles and solves a sparse linear system, reconstructs the pressure field, and computes `q = -g^3 grad(p)`.

The package is early-stage scientific software (`reynoldsflow` 0.1.1, Python >=3.10). Optimize it carefully: numerical equivalence, connectivity, boundary conditions, flux conservation, and peak memory matter at least as much as wall time.

## Canonical repository map

- `src/reynoldsflow/transport.py`: Cartesian solver and its public API.
- `src/reynoldsflow/transport_polar.py`: polar/annular solver and its public API.
- `src/reynoldsflow/_connectivity.py`: shared periodic spanning-component analysis.
- `src/reynoldsflow/_active_dofs.py`: compact grid/DOF mappings and reconstruction.
- `src/reynoldsflow/_linear_solvers.py`: shared solver registry and diagnostics.
- `src/reynoldsflow/_validation.py`: geometry-aware input validation.
- `src/reynoldsflow/__init__.py`: intentionally empty; users import the two modules explicitly.
- `tests/analytical_test.py`: four small, deterministic matrix/pressure regression tests. This is the reliable fast suite.
- `tests/test_solve.py`: large Cartesian Pardiso run (`1000 x 1000`), closer to a manual performance test than a unit test.
- `tests/test_evolution.py`: 40 rough-contact solves and percolation evolution; requires a PETSc-capable setup as currently written.
- `tests/test_solvers.py`: optional-backend compatibility/performance sweep. It writes `test_solvers.result` and expects every optional backend.
- `tests/polar_flow.py`: very large manual annulus generator/runner; it is not collected by pytest and currently exits before postprocessing.
- `scripts/fluid_plot.py`: CLI/functions for plotting saved `gap`, `pressure`, and `flux` arrays from `.npz` files.
- `README.md`: user-facing overview and historical performance numbers.
- `pyproject.toml`: authoritative packaging and dependency metadata.
- `docs/test_solvers.md`: notes for the optional solver sweep; some solver spellings are stale.

Large local/legacy directories such as `EXTRAs/`, `Beauty/`, `TESTs/`, `dev/`, `REF/`, `dist/`, and generated `.npz`/images are gitignored and are not canonical implementation sources. Do not edit or delete them unless the task explicitly includes them. The current package metadata lives under `src/reynoldsflow.egg-info/`; the old `src/fluxflow.egg-info/` path, if present, is stale and should stay out of the implementation path.

## Data layout and boundary conditions

### Cartesian

- Input is assumed to be a square `gaps` array with shape `(n, n)`; axis 0 is `x` (west to east) and axis 1 is periodic `y`.
- Reservoir pressures default to west `p=0` and east `p=1` and are configurable with `p_west`/`p_east`.
- The `y` direction is periodic, including connectivity across columns `0` and `-1`.
- Cartesian values are cell-centered with `dx=dy=1/n`; reservoir values lie on the physical west/east faces, half a cell from the boundary-cell centers. Thus a constant-gap pressure profile has values strictly between 0 and 1.
- `solve_fluid_problem(gaps, solver, ...)` returns `(filtered_gaps, pressure, flux)` or `(None, None, None)` when there is no percolation or a caught solve error.
- `flux[..., 0]` is `q_x`; `flux[..., 1]` is `q_y`.
- `prepare_fluid_problem(gaps)` caches topology/CSR structure for sequences with an unchanged positive mask.

### Polar

- Input shape is `(n_r, n_theta)`; axis 0 is radius and axis 1 is angle.
- Inner and outer radial rows are nodal Dirichlet boundaries, defaulting to `p_inner=1`, `p_outer=0`.
- `theta_bc="auto"` selects periodic boundaries only for an angular extent close to `2*pi`; a sector otherwise receives symmetry/zero-angular-flux boundaries.
- Periodic grids use `dtheta = theta_extent / n_theta`; symmetry grids include both angular endpoints and use `theta_extent / (n_theta - 1)`.
- Polar dilation defaults to zero because conservative open/blocked faces already enforce impermeability. Positive `dilation_iterations` explicitly solves on an expanded channel and then masks flux back to the original component.
- `solve_fluid_problem_polar(...)` returns `(filtered_gaps, pressure, flux, dr, dtheta)` or five `None` values on a caught failure/no percolation.
- `flux[..., 0]` is radial `q_r`; `flux[..., 1]` is tangential `q_theta`.
- `prepare_fluid_problem_polar(...)` provides the corresponding fixed-topology sequence API.

## Solver pipeline

The Cartesian and polar entry points follow the same broad sequence:

1. Label positive-gap cells with 4-connectivity.
2. Merge labels across a periodic seam with union-find where applicable.
3. Retain every component touching both pressure boundaries and zero every other component.
4. In the polar solver only, optionally dilate that selected channel.
5. Map positive percolating cells to compact int32 DOFs (fully open grids bypass mapping overhead).
6. Assemble exact-size CSR in two Numba passes. Open/open face conductivity is the harmonic mean of `g^3`.
7. Solve through the shared backend registry, reconstruct full-grid pressure, calculate conservative face flux and the legacy cell-shaped visualization flux, then integrate boundary faces.

Important implementation details:

- Public solves use one unknown per positive percolating cell; the full-grid matrix builders remain as numerical references.
- Polar assembly discretizes `(1/r) d_r(r g^3 d_r p) + (1/r^2) d_theta(g^3 d_theta p)` and eliminates radial Dirichlet neighbors into the RHS.
- CSR indices are `int32` with overflow checks; values and RHS are `float64`.
- Numba warmup is explicit rather than automatic. First-call compilation and steady-state solve timing must be measured separately.
- `setup_logging()` is idempotent within each solver module.

## Backends and exact solver strings

Base dependencies include NumPy, SciPy, Numba, scikit-image, and PyAMG. `pypardiso`, `petsc4py`, and `scikit-sparse` are optional under `reynoldsflow[solvers]`, even though the legacy `requirements.txt` lists them all.

- `scipy.amg-rs`: SciPy iterative solve with Ruge-Stuben AMG; available in the base install.
- `scipy.amg-smooth_aggregation` (alias `scipy.amg-sa`): SciPy iterative solve with smoothed-aggregation AMG.
- `pardiso`: MKL Pardiso through `pypardiso`.
- `cholesky`: CHOLMOD through `scikit-sparse`.
- `petsc-cg.hypre`, `petsc-cg.gamg`: PETSc iterative paths.
- `petsc-mumps`: PETSc/MUMPS direct solve.
- `auto`/`none`: use portable `scipy.amg-rs`. Native PETSc/Pardiso stacks are explicit-only because a broken MPI/MKL installation can terminate below Python before fallback.

Solver parsing and aliases are centralized in `_linear_solvers.py`. `scipy-spsolve` is implemented. Unknown names, unavailable explicit backends, and non-convergence raise explicit ReynoldsFlow errors.

The Pardiso backend declares the operator as real SPD (`mtype=2`) and must pass
only one matrix triangle to MKL. Passing the complete symmetric CSR matrix can
segfault below Python on larger problems.

Both geometry matrices are symmetric and share the same CG-capable backend layer. PETSc remains an opt-in environment-specific backend.

## Validation commands

Install for development with:

```bash
python -m pip install -e '.[dev]'
```

Safe default regression command:

```bash
python -m pytest -q
```

This runs base-dependency tests and explicitly skips workloads marked `slow`, `backend`, or `benchmark`. Opt in with:

```bash
python -m pytest -q --run-slow -m slow
python -m pytest -q --run-slow --run-backend -m backend
python -m pytest -q --run-benchmark -m benchmark
```

Run every canonical backend in a crash-isolated, thread-controlled performance
suite with:

```bash
python -m benchmarks.benchmark_suite \
  --case circle --size 512 --rtol 1e-8 --repeat 3 --threads 1
```

The original analytical subset can still be run with:

```bash
python -m pytest -q tests/analytical_test.py
```

All four analytical tests passed in the inspected baseline. They directly solve the assembled matrices with SciPy and cover constant and linearly varying gaps in Cartesian and polar coordinates. Additional unit tests cover conservative Cartesian/polar face flux, matrix symmetry, periodic connectivity, multiple spanning channels, sector quadrature, single-sample angular extent, iterative convergence diagnostics, and solver aliases. Optional native backends are exercised in subprocess-isolated opt-in integration tests. Dilation remains only partially covered.

The large Pardiso example, evolution run, and optional-solver sweep require explicit flags and an environment with the intended backend and memory budget. `tests/polar_flow.py` remains a separate manual workload.

For optimization work, record both elapsed time and peak RSS, separate stages (connectivity, assembly, format conversion/preconditioner, solve, flux), and report whether Numba was cold or warm. Use fixed arrays/seeds and compare pressure, total flux, conservation error, percolating mask, iteration/convergence status, and matrix `nnz`; wall time alone is insufficient.

## Known correctness and API risks

Treat these as issues to test before or while optimizing, not as behavior to preserve blindly:

- Public input validation rejects non-2D, rectangular, too-small, nonnumeric,
  NaN, and infinite gap fields before numerical kernels run.
- Connectivity now retains all spanning labels with shared union-find seam merging. Further optimization must preserve this multi-channel behavior.
- Cartesian assembly and flux reconstruction now share a cell-centered finite-volume convention and conservative face conductivities.
- Polar symmetry-sector integration uses endpoint half weights, and `n_theta=1` uses `theta_extent`.
- Polar dilation changes the geometry used for pressure/conductance while returning the undilated gap mask; quantify this effect before treating dilation as a pure postprocessing aid.
- The polar operator is row-scaled to a symmetric conservative stored form; backend-specific symmetry and positive-definiteness assumptions still need integration coverage.
- Iterative solves default to `rtol=1e-12` and record iterations and relative residual. Constricted-flow benchmarks showed that `1e-10` algebraic tolerance could still leave approximately `1e-6` relative total-flow error; native-backend integration coverage remains opt-in.
- Public entry points return `None` only for normal no-percolation and propagate invalid input, backend, convergence, and numerical errors.
- README/default-solver claims, boundary-pressure text, old test comments, and some documented solver spellings do not all match current code. Update documentation whenever the API is corrected.

## Highest-value optimization targets

In roughly the order they should be investigated:

1. Quantify peak memory at production sizes; small-process RSS is dominated by imports/JIT and does not expose the full CSR/compact benefit.
2. Optimize the one-shot assembly, preconditioner construction, solve, and
   flux pipeline. The primary rough-contact workflow does not repeatedly solve
   the same operator.
3. Treat X-flow and Y-flow as distinct operators: their Dirichlet and periodic
   boundary directions differ, so a factorization or preconditioner must not be
   reused without rebuilding or explicit validation.
4. Explore matrix-free/geometric multigrid and distributed PETSc only against the compact CSR baseline.
5. Remove or isolate dead legacy COO, gradient, dilation, and filtering helpers after compatibility/reference coverage is no longer needed.

Do not start with micro-optimizing Numba arithmetic: connectivity, full-grid DOF count, triplet over-allocation, format copies, and solver/preconditioner choice offer much larger gains.

## Change discipline

- Keep axis conventions, pressure orientation, periodic seam behavior, and polar grid spacing explicit in new APIs and tests.
- Prefer small deterministic correctness tests before large rough-surface benchmarks.
- Add flux and connectivity regressions before modifying those stages; current analytical coverage is pressure/matrix-only.
- For iterative solvers, assert convergence and record residual/iterations. Never accept a faster unconverged result.
- Compare cold and warmed runs separately, and avoid including plotting, random-field generation, or JIT compilation in solver timings unless stated.
- Preserve user data and ignored historical artifacts. Generated matrices (`transport_matrix*.npz`, `transport_rhs*.npz`), plots, and benchmark result files should remain uncommitted.
- Use `rg`/`rg --files` for source discovery, `apply_patch` for edits, and inspect `git diff` before handing work back.
