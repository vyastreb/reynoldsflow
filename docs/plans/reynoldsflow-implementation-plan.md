# ReynoldsFlow improvement implementation plan

Status: active

Started: 2026-07-10

## Objective

Make the Cartesian and polar Reynolds solvers numerically trustworthy, substantially reduce their peak memory and runtime on contact-rich problems, and provide reliable backend selection and diagnostics without abruptly breaking the existing public tuple-based API.

Correctness gates every performance change. A faster implementation is accepted only when its connectivity mask, pressure, conservative face flux, total flow, residual, and convergence status agree with an appropriate reference.

## Numerical contract

The following decisions guide the implementation and must be reflected in tests and user documentation.

### Cartesian geometry

- Treat the gap array as cell-centered finite volumes on `[0, 1] x [0, 1]`.
- Axis 0 is the transport direction `x`; axis 1 is periodic `y`.
- Use `dx = 1 / n_x` and `dy = 1 / n_y`.
- West/east reservoir pressures are located on physical boundary faces, half a cell from boundary cell centers.
- Use the harmonic mean of `g^3` on open/open internal faces.
- Use the same face coefficients for matrix assembly and flux reconstruction.
- Preserve the current west `p=0`, east `p=1` convention initially; make pressures configurable in the corrected API.

### Polar geometry

- Retain the current `(n_r, n_theta)` layout and nodal radial boundary rows for compatibility.
- Axis 0 is radius; axis 1 is angle.
- Multiply interior equations by `r_i` (or derive the equivalent finite-volume form) so radial neighbor couplings are symmetric.
- Integrate radial face flux with the correct circumference/sector measure.
- Use periodic angular sampling for a full annulus and endpoint-aware quadrature for symmetry sectors.

### Connectivity and failures

- Keep the union of every independent component spanning the two pressure boundaries.
- A non-percolating field remains a normal no-solution result for compatibility.
- Invalid inputs, unavailable backends, unknown solvers, breakdowns, and non-convergence become explicit errors rather than `None` results.

## Target internal structure

Keep `reynoldsflow.transport` and `reynoldsflow.transport_polar` as compatible public modules. Extract shared responsibilities gradually:

```text
src/reynoldsflow/
  transport.py              Cartesian public API and numerical orchestration
  transport_polar.py        Polar public API and numerical orchestration
  _connectivity.py          Periodic component merging and spanning masks
  _linear_solvers.py        Backend registry, selection, and diagnostics
  _exceptions.py            Publicly meaningful error types
```

Do not split geometry-specific assembly into more modules until its corrected form and performance profile are stable.

## Milestone 0 — Reproducible baseline

Status: completed (initial base-dependency harness)

### Deliverables

- Deterministic Cartesian and polar benchmark cases.
- A base-dependency staged benchmark runner.
- JSON-compatible results containing environment metadata, timings, sizes, `nnz`, flux, conservation error, and peak process RSS.
- Separate cold-import/JIT and warmed measurements in performance reports.

### Required cases

- constant open gap;
- gap varying along the transport direction;
- circular obstruction;
- periodic seam-crossing channel;
- two independent spanning channels;
- non-percolating field;
- deterministic rough/contact field;
- constant and radially varying annuli.

### Acceptance

- Quick cases run on a base installation without optional solvers.
- Random-field creation and plotting are outside solver timings.
- Benchmark results identify geometry, case, size, versions, solver, active cells, DOFs, and `nnz`.

## Milestone 1 — Safe and meaningful tests

Status: completed (core regression structure; backend coverage continues in Milestone 6)

### Deliverables

- Register `unit`, `integration`, `backend`, `slow`, and `benchmark` pytest markers.
- Make ordinary `python -m pytest -q` skip optional-backend and large workloads.
- Preserve explicit opt-in commands for each excluded category.
- Add strict expected-failure characterizations for confirmed defects before fixing them.

### Required regression coverage

- Cartesian constant and variable-gap pressure;
- conservative Cartesian face flux and total flow;
- flux agreement on every transport-normal section;
- periodic translation invariance;
- union of multiple spanning channels;
- transitive periodic seam merging;
- non-percolating fields;
- invalid shapes, values, dimensions, and radii;
- polar logarithmic pressure and radial flux;
- polar matrix symmetry and positive definiteness where applicable;
- symmetry-sector integration;
- direct/iterative backend agreement and convergence reporting.

### Acceptance

- Default tests use only base dependencies and complete quickly.
- Known defects appear as strict `xfail`, not silently absent coverage.
- Fixing a defect creates an `XPASS(strict)` until the marker is deliberately removed.

### Commands

```bash
# Safe base-dependency suite
python -m pytest -q

# Large tests (a test carrying both markers requires both flags)
python -m pytest -q --run-slow -m slow
python -m pytest -q --run-slow --run-backend -m backend
python -m pytest -q --run-benchmark -m benchmark
```

## Milestone 2 — Behavior-preserving cleanup

Status: completed

### Tasks

- Remove the unused Cartesian `linspace` and two full `meshgrid` arrays.
- Make logging setup idempotent.
- Add a package `NullHandler`.
- Replace automatic import-time warming with an explicit benchmark warmup API.
- Remove unused imports and classify dead numerical helpers.
- Normalize solver parsing and add public type hints.

### Acceptance

- Existing masks, matrices, pressure, and flux remain unchanged.
- Fast tests pass.
- Cartesian peak RSS drops by the two removed float64 grids (about `16*N` bytes for `N=n^2`).

## Milestone 3 — Conservative Cartesian discretization

Status: completed

### Assembly

For internal faces, assemble finite-volume conductances

```text
a_e/w = k_face * dy / dx
a_n/s = k_face * dx / dy
```

with `k_face` the harmonic mean of adjacent `g^3`. At a west/east reservoir face, use the half-cell distance:

```text
a_boundary = 2 * k_cell * dy / dx
```

Add the known pressure contribution to the RHS.

### Flux

- Compute internal and boundary face flux from the same conductances.
- Use half-cell pressure distance at Dirichlet faces.
- Integrate total flux on faces with `dy=1/n_y`.
- Preserve the existing cell-centered visualization array by averaging adjacent face values internally; retain exact boundary-face values for boundary integration.
- Consider exposing face flux in a later additive API.

### Acceptance

- Unit gap gives the analytical dimensionless conductance.
- A 1D variable-gap channel has constant discrete face flux.
- Direct-solver inlet, outlet, and internal-section totals agree near machine precision.
- Iterative conservation error scales with requested tolerance.
- Refinement converges to the continuous analytical solution.

## Milestone 4 — Correct periodic connectivity

Status: completed

### Tasks

- Add shared `find_spanning_mask(gaps, transport_axis, periodic_axis=None)`.
- Label 4-connected local components.
- Union component labels joined across a periodic seam.
- Compress the root lookup table.
- Select every root touching both pressure boundaries.
- Build the result in one vectorized lookup; remove full-grid Python relabel loops and repeated whole-array replacement.

### Acceptance

- Parallel independent channels are all retained and their flows add.
- Transitive seam unions work in both geometries.
- No Python loop scales with the total cell count.
- Time and peak memory do not regress on representative fields.

## Milestone 5 — Conservative polar discretization

Status: completed

### Tasks

- Row-scale/derive the radial equations to remove stored `1/r_i` asymmetry.
- Keep known Dirichlet rows out of interior couplings and place their contributions in the RHS.
- Compute radial and angular face flux consistently with assembly.
- Correct symmetry-sector endpoint weights and `n_theta=1` angular measure.
- Compare current dilation, no dilation, and interface-aware flux treatment; remove dilation from the physical solve if it is unnecessary.

### Acceptance

- `A` is symmetric to floating-point precision.
- Constant gap reproduces logarithmic pressure with refinement convergence.
- Integrated radial flow is radius-independent.
- CG and a direct reference agree.
- Full-annulus and symmetry-sector totals use correct angular measures.

## Milestone 6 — Solver registry and explicit diagnostics

Status: completed (optional native backends remain opt-in integration tests)

### Tasks

- Add a shared backend registry describing dependency, matrix requirements, sparse format, direct/iterative type, and preconditioners.
- Implement real SciPy `spsolve` support.
- Retain existing strings as deprecated aliases and introduce unambiguous canonical names.
- Make `auto` inspect installed dependencies and matrix properties, always falling back to base SciPy/PyAMG.
- Return/record iteration count, residual norm, convergence reason, and selected backend.
- Add explicit invalid-input, unavailable-solver, unknown-solver, and convergence errors.

### Acceptance

- `auto` works in a clean base installation.
- Unknown or unavailable solvers produce actionable errors.
- Iterative non-convergence cannot be mistaken for success.
- Available backends agree with a direct reference within declared tolerances.

## Milestone 7 — Compact active-DOF systems

Status: completed

### Tasks

- Build `grid_to_dof` (`-1` inactive) and `dof_to_grid` mappings from the spanning mask.
- Eliminate identity rows for blocked cells.
- Assemble only active-active faces and active reservoir contributions.
- Reconstruct compatible full-grid pressure and flux results after solving.
- Add an optional future compact-output mode for extreme grids.

### Acceptance

- `ndof == count_nonzero(spanning_mask)`.
- Compact and full-grid reference solutions agree on active cells.
- Contact-rich cases show substantial peak-memory reduction.
- Fully open cases incur only small mapping overhead.

## Milestone 8 — Direct CSR assembly

Status: completed

### Tasks

- Replace fixed `5*N` COO triplets with a two-pass Numba CSR builder.
- First pass counts entries per active row and prefix-sums `indptr`.
- Second pass fills exact-size `indices`, `data`, and RHS arrays.
- Select `int32` or `int64` indices with explicit overflow guards.
- Preserve optional saved-matrix formats without retaining unnecessary copies.

### Acceptance

- CSR equals the reference matrix on exhaustive small cases.
- Symmetry and numerical results remain unchanged.
- Row-index storage and COO-to-CSR peak copies are eliminated.
- Assembly time and peak RSS improve on representative cases.

## Milestone 9 — Repeated-solve reuse

Status: completed (topology reuse; AMG hierarchy reuse is opt-in)

### Tasks

- Introduce prepared geometry/topology objects for evolution sequences.
- Reuse active mappings, CSR structure, neighbor topology, and backend matrix objects when the open mask is unchanged.
- Evaluate safe reuse of AMG hierarchy or symbolic direct factorization.
- Detect topology changes and invalidate every dependent cache.

### Acceptance

- Cached sequences agree with independent fresh solves.
- Cache invalidation has direct tests.
- Repeated-step time improves measurably on the evolution workload.

## Milestone 10 — Advanced scaling and release

Status: completed for 0.1.0; native backend results remain environment-specific

### Candidates

- matrix-free stencil operators;
- geometric multigrid;
- distributed PETSc assembly;
- partitioned/parallel connected components;
- parallel Numba assembly;
- compact or out-of-core result output.

Each candidate must beat compressed CSR with an appropriate preconditioner on a documented workload; complexity alone is not a reason to adopt it.

### Release gate

- Synchronize README equations, pressure orientation, grids, solver names, defaults, exceptions, and optional dependencies.
- Remove stale tracked `fluxflow.egg-info` in a dedicated packaging change. Completed: the old metadata path is no longer part of the tracked package state.
- Test installation from a clean environment on supported Python versions.
- Publish before/after time, memory, iteration, residual, and accuracy tables.
- Keep generated matrices, plots, archives, and benchmark results uncommitted.

## Dependency order

```text
baseline and safe tests
        |
behavior-preserving cleanup
        |
Cartesian correctness ----- connectivity correctness
        |                          |
polar correctness           active-DOF mapping
        |                          |
solver registry ------------ direct CSR assembly
                   |
           repeated-solve reuse
                   |
         advanced scaling/release
```

## Work log

- 2026-07-10: Repository audit completed and summarized in `AGENTS.md`.
- 2026-07-10: Fast analytical baseline: four tests passing.
- 2026-07-10: Implementation plan started; Milestones 0 and 1 selected as the first batch.
- 2026-07-10: Added deterministic staged Cartesian/polar benchmark harness and JSON/environment reporting.
- 2026-07-10: Default test suite made safe: six passed, three opt-in skips, and two strict known-defect xfails.
- 2026-07-10: `24 x 24` Cartesian unit-gap reference reproduced `Q_total=1.92` with a `1.37e-15` linear residual, isolating the error to flux/grid postprocessing rather than the linear solve.
- 2026-07-10: `16 x 32` constant polar reference produced a `4.96e-2` boundary-flux conservation error with a `4.45e-16` linear residual, providing a refinement baseline for the polar flux milestone.
- 2026-07-10: Removed unused Cartesian coordinate grids, duplicate logging handlers, and automatic import-time Numba compilation.
- 2026-07-10: Replaced Cartesian reservoir/flux treatment with a cell-centered finite-volume scheme. The `24 x 24` unit-gap result is now `Q_total=1.0000000000000018` with `2.89e-15` conservation error.
- 2026-07-10: Added conservative Cartesian face flux and verified constant flux for a one-dimensional variable gap.
- 2026-07-10: Replaced both connectivity implementations with shared union-find seam merging and retained the union of all spanning components.
- 2026-07-10: Row-scaled the polar operator to symmetry and added conservative radial/angular face fluxes, symmetry-sector endpoint weights, and correct single-sample angular extent.
- 2026-07-10: The corrected `16 x 32` polar constant-gap reference reduced conservation error from `4.96e-2` to `1.82e-14`; SciPy AMG at `24 x 48` agreed with analytical total flow to `8.5e-5` relative error.
- 2026-07-10: Added shared solver registry, SciPy direct support, legacy aliases, residual/iteration diagnostics, explicit backend/convergence errors, and portable SciPy/PyAMG `auto` selection. PETSc and Pardiso remain explicit because broken native installations can terminate below Python before fallback.
- 2026-07-10: Added finite numeric/shape/geometry validation while preserving normal `None` results for non-percolating fields.
- 2026-07-10: Compact active-DOF Cartesian and polar systems match full-grid reference pressures to approximately `3e-13` and preserve polar symmetry.
- 2026-07-10: Replaced fixed-size COO triplets and format conversion with exact-size two-pass CSR assembly for full and active systems; CSR matrices/RHS match the former COO builders in both geometries and angular boundary modes.
- 2026-07-10: On the `256 x 256` rough-contact benchmark (62.46% active), compact AMG reduced DOFs from 65,536 to 40,936 and warmed solve time from 0.854 s to 0.360 s (2.37x), with matching total flow.
- 2026-07-10: Polar dilation now defaults to zero; positive iterations are an explicit geometry modification rather than an invisible numerical aid.
- 2026-07-10: Safe suite after Milestones 0–8: 42 passed and three explicit slow/backend skips in 19.68 s; no expected failures remain.
- 2026-07-10: Added prepared Cartesian/polar problem APIs that reuse connectivity, active mappings, CSR row/column structure, and reject any open/closed topology change.
- 2026-07-10: Added opt-in AMG hierarchy reuse with mandatory convergence/residual checks and explicit cache clearing.
- 2026-07-10: On a three-step `256 x 256` rough sequence, fresh total time was approximately 5.14 s versus 4.28 s including preparation for prepared topology. Reusing the first AMG hierarchy was not consistently faster because later iteration counts rose from roughly 66–69 to 91; it therefore remains opt-in rather than the default.
- 2026-07-10: On the `512 x 512` rough reference, `rtol=1e-10` left about `1e-6` relative total-flow error versus a direct solve despite a passing algebraic residual. Tightening to `rtol=1e-12` reduced total-flow error to about `1.2e-9` and conservation error to `1.6e-9`; `1e-12` is now the library default.
- 2026-07-10: On the `1024 x 1024` rough benchmark (60.4% active), compact CSR reduced DOFs from 1,048,576 to 633,360, AMG solve time from 9.55 s to 5.45 s (1.75x), and peak RSS from about 640 MB to 628 MB. Dense outputs, imports, and the AMG hierarchy limit the process-level memory reduction.
- 2026-07-10: Removed verified-dead COO/gradient kernels and stale `fluxflow` metadata; explicit warmup compiles live kernels only.
- 2026-07-10: Added subprocess-isolated native-backend checks, a Python 3.10–3.12 CI matrix, version 0.1.0 metadata/changelog, deterministic performance report, and source/wheel manifests.
- 2026-07-10: Added configurable Cartesian reservoir pressures (`p_west`, `p_east`) across one-shot and prepared APIs.
- 2026-07-10: Release-candidate safe suite: 61 passed and nine explicit slow/backend skips. Source and wheel distributions build successfully; the wheel was smoke-tested from an isolated virtual environment.
- 2026-07-11: Corrected the MKL Pardiso SPD interface to pass one matrix triangle; the former full-symmetric input could segfault on the `512 x 512` case.
- 2026-07-11: Added a subprocess-isolated all-backend performance suite with controlled thread counts and separate cold-start and steady-state reporting. All nine canonical backends completed the `512 x 512` circle case in the `fluidpaper` environment.
