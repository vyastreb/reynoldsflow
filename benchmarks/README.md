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

The runner currently establishes a correctness/performance reference. Optional
backend comparisons and subprocess-isolated cold/warm suites will be added in
later milestones.

Repeated fixed-topology sequences can compare fresh assembly/preconditioning,
prepared CSR topology, and prepared topology plus AMG hierarchy reuse:

```bash
python -m benchmarks.benchmark_evolution --size 256 --steps 5 --fresh
python -m benchmarks.benchmark_evolution --size 256 --steps 5
python -m benchmarks.benchmark_evolution \
  --size 256 --steps 5 --reuse-preconditioner
```

Prepared problems require the open/closed mask to remain unchanged and raise an
error when the topology changes. Reusing AMG is opt-in and every solve still
checks convergence and the final residual.
