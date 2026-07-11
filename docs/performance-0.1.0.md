# ReynoldsFlow 0.1.0 correctness and performance report

Date: 2026-07-10

Host: 20 logical CPUs, Linux x86-64, Python 3.12.2

Core stack: NumPy 1.26.4, SciPy 1.17.1, Numba 0.64.0, PyAMG 5.2.1

All data below comes from deterministic cases in `benchmarks/`. Timings and
peak RSS include the limitations noted with each table and should not be
generalized to other surfaces or solver stacks without rerunning the commands.

## Correctness fixes

| Case | Previous result | 0.1.0 result |
|---|---:|---:|
| Cartesian unit gap, `24 x 24`, expected `Q=1` | `Q=1.92` | `Q=1.0000000000000018` |
| Cartesian unit-gap conservation error | approximately `4e-11` in iterative smoke test | `2.89e-15` with direct reference |
| Polar unit gap, `16 x 32`, boundary conservation error | `4.96e-2` | `1.82e-14` |
| Polar `24 x 48`, analytical total-flow relative error | not conservatively reconstructed | `8.5e-5` |

The Cartesian and polar matrices are symmetric after boundary elimination/row
scaling. Conservative face flux is derived from the same conductances used by
assembly.

## Compact versus full-grid AMG

Deterministic `rough-contact` case, `solver=scipy.amg-rs`, `rtol=1e-10` for
performance comparison. `active` is the retained spanning-cell fraction.

| Grid | Active | System | DOFs | `nnz` | Solve (s) | Peak RSS (MB) |
|---:|---:|---|---:|---:|---:|---:|
| `256²` | 62.46% | compact | 40,936 | 199,506 | 0.360 warmed | 275 |
| `256²` | 62.46% | full | 65,536 | 224,106 | 0.854 warmed | 272 |
| `512²` | 62.41% | compact | 163,605 | 807,387 | 2.96 warmed | 351 |
| `512²` | 62.41% | full | 262,144 | 905,926 | 3.50 warmed | 354 |
| `1024²` | 60.40% | compact | 633,360 | 3,149,082 | 5.45 | 628 |
| `1024²` | 60.40% | full | 1,048,576 | 3,564,298 | 9.55 | 640 |

At `1024²`, compact DOFs reduced the linear solve time by 1.75x. Process RSS
fell much less than the DOF count because Python imports, dense input/output
arrays, and the AMG hierarchy dominate at this scale. The former COO triplet
peak is absent from both 0.1.0 rows because both now use direct CSR; the
full-grid option is retained only as a benchmark/reference path.

## Tolerance calibration

On the compact `512²` rough case, the SciPy direct reference produced
`Q=0.002964149471992414`.

| Solver tolerance | Total flow | Relative flow error | Conservation error |
|---:|---:|---:|---:|
| `1e-10` | `0.002964152415423008` | approximately `9.9e-7` | `1.98e-6` |
| `1e-12` | `0.002964149475495322` | approximately `1.2e-9` | `1.61e-9` |

This sensitivity motivated the 0.1.0 default `rtol=1e-12`. Users may choose a
looser value for exploratory performance runs.

## Fixed-topology evolution

For a three-step `256²` sequence:

- fresh topology/assembly/preconditioner total: approximately 5.14 s;
- prepared topology total including one-time preparation: approximately 4.28 s;
- reusing the first AMG hierarchy raised later iteration counts from about
  66–69 to 91 and was not consistently faster.

Prepared topology is therefore useful and safe by default, while AMG hierarchy
reuse remains explicit and guarded by convergence/residual checks.

## Commands

```bash
python -m benchmarks.benchmark_solver \
  --geometry cartesian --case rough-contact --size 1024 \
  --solver scipy.amg-rs --rtol 1e-10

python -m benchmarks.benchmark_solver \
  --geometry cartesian --case rough-contact --size 1024 \
  --solver scipy.amg-rs --rtol 1e-10 --full-grid-system

python -m benchmarks.benchmark_evolution \
  --geometry cartesian --size 256 --steps 3 --solver scipy.amg-rs
```

Native PETSc/Pardiso/CHOLMOD comparisons are deliberately absent: the inspected
host's installed MPI/MKL stacks are not safe to initialize. Optional backends
now have subprocess-isolated tests for compatible machines.
