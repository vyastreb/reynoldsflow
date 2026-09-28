# Rough-surface homogenization study

This directory contains the restartable workflow and retained numerical results
for the Reynolds-flow homogenization study.  Two related studies were carried
out:

- the initial study used fixed `k2/k0 = 64`, five realizations, and contact
  fractions 10%, 20%, and 30%;
- the final representative-volume study used fixed `k2/k1 = 64`, ten
  realizations, and 30% geometrical contact.  This is the preferred dataset
  documented below.

The command-line production defaults still describe the initial study:

- periodic `4096 x 4096` self-affine surfaces with `H = 0.8`;
- five roll-off ratios `k1/k0 = 1, 2, 4, 6, 8`, with `k0 = 2*pi/L` and
  `k2/k0 = 64`;
- five paired realizations per roll-off (the same random seed is reused across
  roll-offs at a fixed grid size for each realization);
- geometrical contact fractions 10%, 20%, and 30%;
- a west/east pressure-reservoir problem (`p0=0`, `p1=1`, transverse periodic)
  and a fully periodic cell problem with prescribed average flux;
- the isotropic 2-D Bruggeman self-consistent estimate based on the same sampled
  local gap conductivities.

The five explicitly listed roll-offs are used even though the original request
said "4 types".  The recursive-looking list entry was interpreted as
`k1 = {k0, 2*k0, 4*k0, 6*k0, 8*k0}`.

## Final fixed-bandwidth study

### Design

The final study has the following common parameters:

- Hurst exponent `H = 0.8`;
- `rfgen 0.2.3`, `plateau=True`, `noise=True`, base seed `20260923`;
- periodic, isotropic, zero-mean surfaces normalized to unit sample RMS;
- fixed spectral bandwidth `k2/k1 = 64`;
- 30% geometrical contact, imposed by the surface-height quantile;
- ten realizations per reported roll-off;
- both west/east prescribed pressure and fully periodic macroscopic-gradient
  cell problems;
- PETSc CG with Hypre, requested relative tolerance `2e-11`;
- Bruggeman's isotropic two-dimensional self-consistent estimate evaluated from
  each realization's filtered local gap distribution.

The grid was increased with roll-off to avoid losing short-wave resolution:

| `k1/k0` | Preferred grid | `k2/k0` | Cells per shortest wavelength |
| ---: | ---: | ---: | ---: |
| 1 | 2048 x 2048 | 64 | 32.00 |
| 2 | 2048 x 2048 | 128 | 16.00 |
| 4 | 2048 x 2048 | 256 | 8.00 |
| 6 | 2048 x 2048 | 384 | 5.33 |
| 8 | 2048 x 2048 | 512 | 4.00 |
| 12 | 4096 x 4096 | 768 | 5.33 |
| 16 | 4096 x 4096 | 1024 | 4.00 |
| 20 | 5120 x 5120 | 1280 | 4.00 |

### Physical domain-scaling interpretation

The roll-off ratio is also the dimensionless physical domain size.  Let

```text
L1 = 2*pi/k1
r  = k1/k0.
```

Keeping the physical roughness spectrum fixed while increasing the sampled
domain gives

```text
Lr    = r*L1
k0(r) = 2*pi/Lr = k1/r
k1    = constant
k2    = 64*k1 = constant.
```

Therefore increasing `k1/k0` at fixed `k2/k1` is precisely an RVE sequence in
which the physical domain grows while the roll-off and short-wave wavelengths
remain fixed.  This is the physical interpretation of the final study.  The
initial fixed-`k2/k0` study is not such an RVE sequence because its physical
bandwidth changes with `r`.

The equivalence can be seen directly from the `rfgen` inputs.  On an `N x N`
grid the code supplies

```text
k_low_grid  = r/N
k_high_grid = 64*r/N.
```

With physical pixel size `dx = Lr/N`, these correspond to the physical angular
wavenumbers

```text
2*pi*(r/N)/dx      = 2*pi/L1 = k1
2*pi*(64*r/N)/dx   = 64*k1   = k2,
```

independent of `r`.

Constant physical grid spacing requires `N proportional to Lr`, hence
`N proportional to r`.  Four cells per shortest wavelength correspond to

```text
N = 4*(k2/k1)*r = 256*r.
```

The points `(r,N) = (8,2048), (12,3072), (16,4096), (20,5120)` form this
strict constant-resolution sequence.  Ratios 1--6 at `N=2048` are
over-resolved.  The preferred ratio-12 result at `N=4096` is also
over-resolved, with 5.33 rather than 4 cells per shortest wavelength; it is
useful statistically but is not part of the strict `N/r=256` sequence.

An earlier ratio-12 ensemble used `3072 x 3072`, or four cells per shortest
wavelength.  It is retained as a mesh/ensemble sensitivity result but is not
used in the preferred curve.

### Preferred normalized results

The table reports ensemble mean +/- standard error of the mean over ten
realizations.  `K' = K/m0**(3/2)`, with the measured sample `m0` used in every
realization.

| `k1/k0` | Prescribed pressure `K'` | Periodic/mean-flux `K'` | Bruggeman `K'` |
| ---: | ---: | ---: | ---: |
| 1 | 0.211786 +/- 0.067784 | 0.192105 +/- 0.058173 | 0.109198 +/- 0.016092 |
| 2 | 0.153181 +/- 0.032690 | 0.124333 +/- 0.028085 | 0.094784 +/- 0.006217 |
| 4 | 0.116874 +/- 0.022759 | 0.094380 +/- 0.015758 | 0.091601 +/- 0.002860 |
| 6 | 0.095934 +/- 0.017570 | 0.079611 +/- 0.011077 | 0.093652 +/- 0.001862 |
| 8 | 0.083272 +/- 0.011006 | 0.068805 +/- 0.007250 | 0.093265 +/- 0.001496 |
| 12 | 0.073360 +/- 0.007495 | 0.064740 +/- 0.006746 | 0.089313 +/- 0.001378 |
| 16 | 0.070505 +/- 0.005267 | 0.062342 +/- 0.004933 | 0.089025 +/- 0.001161 |
| 20 | 0.067897 +/- 0.004184 | 0.062728 +/- 0.003499 | 0.089896 +/- 0.000818 |

For the earlier `3072 x 3072` ratio-12 ensemble, pressure and periodic results
were `0.063254 +/- 0.005806` and `0.055935 +/- 0.004782`.  The preferred finer
ensemble is approximately 16% higher for both boundary formulations, but the
difference is not statistically resolved by a two-sided Welch test
(`p approximately 0.30`).

### Convergence interpretation

With the preferred fine ratio-12 data, the pressure sequence at ratios
12, 16, and 20 is `0.07336, 0.07050, 0.06790`; the periodic sequence is
`0.06474, 0.06234, 0.06273`.  The pressure estimate is monotonic over these
three points, while the periodic estimate is effectively flat between 16 and
20.  A one-way exploratory ANOVA cannot distinguish the three large-domain
means (`p = 0.80` for pressure and `p = 0.94` for periodic).  This is evidence
of a plateau near `K' approximately 0.063--0.068`, not a proof of an asymptotic
limit.  The pressure/periodic difference is still 8.2% relative to the periodic
mean at ratio 20.

Bruggeman theory has converged near `K' = 0.09` but remains above both resolved
simulations.  It uses the one-point gap distribution and therefore does not
capture the spatially correlated bottlenecks controlling the numerical flow.

All 160 solves in the preferred curve reported convergence.  Across this set,
the largest recomputed relative residual was `4.31e-11`, the largest relative
flux-conservation error was `1.27e-8`, and the largest iteration count was 19.
For the fine ratio-12 subset, the corresponding maxima were `1.99e-11`,
`1.01e-8`, and 18.

### Retained net results

The preferred combined results and figures are in:

```text
output/fixed_k2_over_k1_64_extended_fine_r12/
```

It contains compact per-realization simulation and theory JSON files plus PNG
and PDF figures.  In particular:

- `transmissivity_c030p0.*`: normalized `K'` with linear `k1/k0` and log `K'`;
- `transmissivity_semilogx_c030p0.*`: normalized `K'` with log `k1/k0` and
  linear `K'`;
- `transmissivity_raw_c030p0.*`: unnormalized `K`;
- `transmissivity_rms_error_c030p0.*`: population RMS ensemble dispersion.

The original per-grid result directories and the coarse ratio-12 comparison are
also retained as small JSON records.  Aggregate folders contain copies of those
JSON files, so duplicate records are expected.

On 2026-09-24, 150 generated `.npy` surface arrays totaling 5.703 GiB were
deliberately removed at the user's request.  No pressure or flux maps had been
saved.  Surface metadata JSON files were retained because they are small and
record generation provenance.  Consequently, `surface_file` entries in the
retained result JSON files point to intentionally absent arrays.  Regenerate
the surfaces before rerunning a solve or recomputing theory.

## Physical conventions

`rfgen` uses normalized cycles per sample, so the physical mode ratios map to
`k_low=(k1/k0)/N` and `k_high=(k2/k0)/N`.  `plateau=True` creates the roll-off.
Each surface is saved with zero mean and unit RMS height.  A rigid plane is put
at the requested surface quantile and

```text
gap = max(surface - plane_height, 0).
```

The code solves `div(gap**3 grad(p)) = 0`; viscosity and the common factor 12
therefore cancel in normalized comparisons.  Following the existing
`fluid-percolation/plot_results.py`, it reports

```text
normalized transmissivity = K' = K / m0**(3/2),
```

where `m0` is the variance (zeroth spectral moment) of the undeformed surface.
The generated surfaces have unit RMS height, but the measured `m0` is retained
in every result rather than assumed to be exactly one.

For the prescribed-flux case, one fully periodic unit-macroscopic-gradient
solve determines `K`.  Linearity then gives the mean pressure gradient required
for the requested average flux (default 1).  This avoids a redundant second
linear solve.  The arbitrary periodic pressure constant is removed by fixing
the cell with the largest gap in each connected component.  Choosing this
well-conditioned point is more robust here than fixing the first grid cell.

### Domain size, flux, and transmissivity

Writing physical coordinates as `x = L*x_hat` removes the explicit domain size
from

```text
div(h**3 grad(p)) = 0,
```

because both derivatives contribute the same factor `1/L**2`.  The effective
transmissivity is defined by

```text
mean(q_x) = -K_eff * DeltaP/L
```

in the code's units, where the conventional factor `1/(12*mu)` is omitted.
For a square domain of width and length `L`, the total flux is therefore

```text
Q = mean(q_x)*L = -K_eff*DeltaP.
```

Consequently:

- at fixed pressure drop, mean flux density decreases as `1/L`, but total flux
  through a square is independent of the explicit factor `L`;
- at fixed macroscopic gradient, mean flux density is `K_eff` times that
  gradient and total flux grows with the transverse width;
- at prescribed mean flux, the required gradient is `mean(q_x)/K_eff`, while
  the corresponding pressure drop grows with `L`;
- the plotted `K_eff/m0**(3/2)` acquires no extra factor of `L`.

Thus explicitly assigning `Lr = r*L1` changes the dimensional interpretation
of pressure gradient and flux density, but it does not change the normalized
transmissivity curve already computed from the fixed-bandwidth sequence.

Before applying Bruggeman theory, disconnected/trapped open regions are removed
with the same boundary-spanning connectivity analysis as the pressure solve.
Theory then follows the established implementation and solves

```text
A_conductive * mean_conductive(
    gap**3 / (gap**3 + K_B_prime*m0**(3/2))
) = 0.5.
```

This is the standard isotropic two-dimensional self-consistent equation and
predicts loss of conductivity at 50% zero-gap area.

## Caveats and limits

- **The mesh comparison is not paired.** Reusing a seed at another FFT size
  does not reproduce the same continuous surface: array shape changes the RNG
  consumption and Fourier modes.  The 16% ratio-12 coarse/fine shift combines
  mesh and ensemble sampling and must not be called a discretization correction.
- **Ten realizations are still a small ensemble.** Low-roll-off results have
  large coefficients of variation and the reported error bars are SEMs, not
  confidence intervals.  The Welch tests and ANOVA are exploratory and assume
  conditions that are difficult to verify with ten samples.
- **Pairing is grid-local.** Realization indices share seeds and are useful for
  paired comparisons only for roll-offs generated on the same grid.  Ratios on
  different grids are separate ensembles even when their indices match.
- **RMS normalization is neutral for the reported `K'`.** Every field was
  shifted to zero mean and scaled to unit sample RMS.  For the present
  quantile-defined contact geometry, a uniform height scaling `h -> a*h`
  scales the gap by `a`, the Reynolds transmissivity by `a**3`, and
  `m0**(3/2)` by the same `a**3`.  Hence `K'=K/m0**(3/2)` is exactly invariant
  apart from numerical error.  The scaling does affect raw `K`, and the
  cancellation would not generally hold for an absolute separation, imposed
  mechanical load, elastic deformation, adhesion, or another non-homogeneous
  contact law.
- **Resolution is empirical.** The largest cases use only four cells per
  shortest wavelength.  A five-realization paired ratio-8 downsampling check
  found about 0.5% transmissivity error, but that does not guarantee the same
  error for every roll-off, load, or realization.
- **The apparent plateau is not complete boundary-condition convergence.** The
  two numerical estimates still differ by 8.2% at ratio 20.  Larger domains or
  more realizations could shift the inferred limit.
- **"KUBC" and "PBC" are homogenization analogies.** The pressure problem has
  west/east Dirichlet reservoirs and transverse periodicity.  The periodic
  problem imposes a macroscopic unit pressure gradient, obtains `K`, and uses
  linearity to rescale to the requested mean flux.  It does not prescribe a
  spatially uniform microscopic flux boundary condition.
- **Only one transport direction was solved.** Statistical isotropy is assumed,
  but finite realizations can be anisotropic; the orthogonal tensor component
  was not independently averaged.
- **Contact is purely geometrical.** The gap is truncated at a height quantile.
  There is no elastic deformation, load-displacement law, adhesion, or contact
  mechanics, so "30% load" here means 30% geometrical contact area only.
- **The Reynolds model has its usual limits.** Results solve
  `div(gap**3 grad(p)) = 0` for an incompressible thin film.  Viscosity and the
  conventional factor 12 are omitted, and inertia, compressibility, slip,
  cavitation, and non-Newtonian effects are absent.
- **Connectivity is filtered.** Only boundary-spanning or periodically winding
  open components contribute.  Trapped positive-gap regions are set to zero
  before both numerical homogenization and Bruggeman postprocessing.
- **Bruggeman is a one-point mean-field model.** It assumes isotropic 2-D
  self-consistency and a 50% zero-conductivity threshold.  It ignores spatial
  correlation, channel topology, and critical bottlenecks, so agreement with
  the simulations is not expected near percolation-controlled flow.
- **Raw and normalized values are nearly equal by construction.** Unit-RMS
  normalization makes `m0` close to one, although every JSON result uses the
  measured value rather than assuming it exactly.
- **Deleted arrays limit immediate reproducibility.** The compact results and
  figures are preserved, but fields cannot be inspected or re-postprocessed
  without deterministic regeneration using the recorded parameters and a
  compatible `rfgen`/NumPy environment.

## Running the study

From the repository root, inspect the complete task mapping first:

```bash
python3 -m homogenization_sims.study plan
```

Run all stages serially with:

```bash
python3 -m homogenization_sims.study generate
python3 -m homogenization_sims.study simulate --solver auto --rtol 1e-12
python3 -m homogenization_sims.study theory
python3 -m homogenization_sims.study plot
```

At production size, prefer job arrays.  Generation task IDs are 0--24,
simulation IDs are 0--149, and theory IDs are 0--74:

```bash
python3 -m homogenization_sims.study generate --task-id "$TASK_ID"
python3 -m homogenization_sims.study simulate --task-id "$TASK_ID" \
  --solver petsc-cg.hypre --rtol 2e-11
python3 -m homogenization_sims.study theory --task-id "$TASK_ID"
```

Every task writes one atomic result and skips an existing result unless
`--force` is passed.  Outputs live in `homogenization_sims/output/`, which is
gitignored.  Generated surfaces dominate persistent storage and are not kept in
the current archived results.  Solver peak memory is much larger than a single
surface and depends on the backend.  Run production solves serially unless the
node's available memory has been measured.

For an inexpensive end-to-end check:

```bash
python3 -m homogenization_sims.study generate \
  --output /tmp/reynoldsflow-homogenization-smoke --size 128 \
  --rolloff-ratios 1,2 --realizations 2 --contacts 0.1
python3 -m homogenization_sims.study simulate \
  --output /tmp/reynoldsflow-homogenization-smoke \
  --rolloff-ratios 1,2 --realizations 2 --contacts 0.1 \
  --solver scipy-spsolve
python3 -m homogenization_sims.study theory \
  --output /tmp/reynoldsflow-homogenization-smoke \
  --rolloff-ratios 1,2 --realizations 2 --contacts 0.1
python3 -m homogenization_sims.study plot \
  --output /tmp/reynoldsflow-homogenization-smoke \
  --rolloff-ratios 1,2 --realizations 2 --contacts 0.1
```

To keep the self-affine bandwidth fixed instead, interpret the cutoff as
`k2/k1`.  For example, the 2048-square, 30%-contact convergence study uses:

```bash
python3 -m homogenization_sims.study generate \
  --output homogenization_sims/output/size2048_fixed_k2_over_k1_64 \
  --size 2048 --rolloff-ratios 1,2,4,6,8 --realizations 10 \
  --contacts 0.3 \
  --short-wave-ratio 64 --short-wave-reference k1
python3 -m homogenization_sims.study simulate \
  --output homogenization_sims/output/size2048_fixed_k2_over_k1_64 \
  --rolloff-ratios 1,2,4,6,8 --realizations 10 --contacts 0.3 \
  --solver petsc-cg.hypre --rtol 2e-11
python3 -m homogenization_sims.study theory \
  --output homogenization_sims/output/size2048_fixed_k2_over_k1_64 \
  --rolloff-ratios 1,2,4,6,8 --realizations 10 --contacts 0.3
python3 -m homogenization_sims.study plot \
  --output homogenization_sims/output/size2048_fixed_k2_over_k1_64 \
  --rolloff-ratios 1,2,4,6,8 --realizations 10 --contacts 0.3
```

Here ``k2/k0`` grows as ``64 * k1/k0``.  The surface metadata records both
cutoff ratios and the shortest wavelength in grid cells.

The plotting stage writes normalized ``K/m0**(3/2)`` and unnormalized ``K``
figures with logarithmic vertical axes.  It also writes a normalized-only
semilog-x figure with logarithmic ``k1/k0`` and linear transmissivity axes.
Error bars are standard errors over the available realizations.  It also
writes one RMS-error figure per load, using the population RMS deviation of
transmissivity from its ensemble mean.  By default plotting requires the
complete requested ensemble; use
`--allow-incomplete` only for progress inspection.

### Reproducing the preferred archived curve

Use these case definitions for the preferred data:

| Ratios | Grid size | Output directory |
| --- | ---: | --- |
| `1,2,4,6,8` | 2048 | `output/size2048_fixed_k2_over_k1_64` |
| `12` | 4096 | `output/size4096_fixed_k2_over_k1_64_ratio12_fine` |
| `16` | 4096 | `output/size4096_fixed_k2_over_k1_64_ratio16` |
| `20` | 5120 | `output/size5120_fixed_k2_over_k1_64_ratio20` |

For each row, set `case_output`, `case_size`, and `case_ratios`, then run the
three computational stages serially from the repository root:

```bash
case_output="homogenization_sims/output/REPLACE_WITH_CASE_DIRECTORY"
case_size=REPLACE_WITH_GRID_SIZE
case_ratios="REPLACE_WITH_COMMA_SEPARATED_RATIOS"

conda run -n fluidpaper env PYTHONPATH=src \
  python -m homogenization_sims.study generate \
  --output "$case_output" --size "$case_size" \
  --rolloff-ratios "$case_ratios" --realizations 10 --contacts 0.3 \
  --hurst 0.8 --short-wave-ratio 64 --short-wave-reference k1 \
  --base-seed 20260923

conda run -n fluidpaper env PYTHONPATH=src \
  python -m homogenization_sims.study simulate \
  --output "$case_output" --rolloff-ratios "$case_ratios" \
  --realizations 10 --contacts 0.3 \
  --solver petsc-cg.hypre --rtol 2e-11

conda run -n fluidpaper env PYTHONPATH=src \
  python -m homogenization_sims.study theory \
  --output "$case_output" --rolloff-ratios "$case_ratios" \
  --realizations 10 --contacts 0.3
```

The separate plotting stage is normally run only after compact JSON files from
all four cases have been collected in one aggregate `results/` and `theory/`
directory.  Copy the fine ratio-12 JSON files last if an aggregate initially
contains the coarse ratio-12 records.  The retained preferred aggregate is
already assembled, so its figures can be regenerated directly with:

```bash
conda run -n fluidpaper env PYTHONPATH=src \
  MPLCONFIGDIR=/tmp/matplotlib-homogenization \
  python -m homogenization_sims.study plot \
  --output homogenization_sims/output/fixed_k2_over_k1_64_extended_fine_r12 \
  --rolloff-ratios 1,2,4,6,8,12,16,20 \
  --realizations 10 --contacts 0.3
```

Because the heavy `.npy` fields were deleted, `generate` recreates them when
run in the retained per-grid directories.  The later `simulate` and `theory`
stages skip existing JSON files unless `--force` is supplied.  To perform an
independent reproduction without altering the archived results, point
`case_output` to new directories instead.
