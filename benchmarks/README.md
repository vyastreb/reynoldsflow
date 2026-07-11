# ReynoldsFlow benchmarks

The staged runner uses deterministic fields and defaults to SciPy's direct
solver, so it is available with base dependencies. SciPy/PyAMG or explicit
optional backends can also be selected. Case construction and plotting are
excluded from reported stage timings. Process peak RSS includes imports, JIT
state, case data, and all completed stages.

Quick Cartesian reference:

```bash
python -m benchmarks.benchmark_solver --geometry cartesian --case constant --size 128
```

Polar reference without geometry dilation:

```bash
python -m benchmarks.benchmark_solver \
  --geometry polar --case constant --size 64 --n-theta 128 \
  --dilation-iterations 0
```

Write ignored JSON results and collect cold-process time/RSS:

```bash
/usr/bin/time -v python -m benchmarks.benchmark_solver \
  --geometry cartesian --case rough-contact --size 512 --repeat 3 \
  --output benchmarks/results/cartesian-rough-512.json
```

The first recorded run is the cold run. With `--repeat 3`, the report also
contains medians for the remaining two steady-state runs. This distinction is
important because the cold run includes Numba compilation, PyAMG hierarchy
setup, and first import/initialization of optional native runtimes. To measure
only warmed runs, request explicit warmups that are discarded:

```bash
python -m benchmarks.benchmark_solver \
  --geometry cartesian --case circle --size 512 \
  --solver petsc-cg.hypre --rtol 1e-8 --warmup 1 --repeat 3
```

Run all canonical backends in separate subprocesses so an MPI or native-solver
failure cannot terminate the suite. Thread counts are pinned and recorded; use
the same value when comparing machines or revisions:

```bash
python -m benchmarks.benchmark_suite \
  --case circle --size 512 --rtol 1e-8 --repeat 3 --threads 1 \
  --output benchmarks/results/cartesian-circle-512.json
```

List available deterministic cases:

```bash
python -m benchmarks.benchmark_solver --list-cases
```

Compare the active-DOF implementation to the legacy full-grid system in
separate processes:

```bash
python -m benchmarks.benchmark_solver \
  --geometry cartesian --case rough-contact --size 512 \
  --solver scipy.amg-rs --rtol 1e-10
python -m benchmarks.benchmark_solver \
  --geometry cartesian --case rough-contact --size 512 \
  --solver scipy.amg-rs --rtol 1e-10 --full-grid-system
```

The suite reports unavailable backends, Python failures, timeouts, and native
crashes independently while continuing with the remaining solvers.
