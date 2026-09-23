# CRAN comments

## Update

This is an update of tulpa 0.5.0, published on 2026-09-21. It comes three days
later because 0.5.0 returns wrong numbers, without an error, in the reported
hyperparameter posterior. Every item below is a correctness fix; there is no
new user-facing surface.

* A refined outer axis read the wrong box for a cell in a row it had not
  refined. A slice point re-tiles only its own row, but the box-uniform
  interval and `tulpa_hyper_draws()` laid one partition over every distinct
  value on the axis, so an untouched cell was drawn on the narrow box beside
  the slice points while holding its whole base box's mass. On a reference
  fit the `phi_pos` draws' interquartile range held 0.453 of the fit's own
  measure instead of 0.5 (gcol33/tulpa#858).

* The copy coefficient's default outer axis carried five nodes, a node ratio
  of 2.34, so a posterior over the amplitude sat on two adjacent nodes. It is
  now declared at nine, the resolution the other outer axes of such a fit run
  at: simulation-based calibration on the reference fixture goes from 9 of 11
  and 7 of 11 parameters inside the family-wise band to 11 of 11 at both
  configurations (gcol33/tulpa#858).

* Hyperparameter draws lost the posterior's correlation between outer axes.
  Independent within-cell jitter added the full box variance of both axes in
  the direction a strongly correlated posterior pins down, so a product of
  anticorrelated scales came out too wide. On an analytic two-scale posterior
  with log correlation -0.9 (exact log-product sd 0.224) the draws' sd goes
  from 0.281 to 0.232 (gcol33/tulpa#859).

* That coupling fell back to independent draws on a coarse grid under a strong
  correlation, because the off-ridge cells' masses fell below the rank test of
  the quadratic that sets its target. At 3 x 3 nodes and log correlation
  -0.97 the log product's sd goes from 0.580 to 0.182 against an exact 0.122
  (gcol33/tulpa#860).

* A pair's within-cell coupling could take a conditional dependence of the
  opposite sign from the posterior's: a curved posterior at log correlation
  -0.76 was coupled at +0.55 inside the cell, and the log product read 19.6%
  wide at 5 x 5 nodes. Such a pair is now conditionally independent inside the
  cell and reads 1.8% wide (gcol33/tulpa#861).

* The subspace-debias closure declined on every grid or joint fit. It grows the
  corrected set over strongly coupled coordinates, because a coordinate coupled
  to a member of the set and left out of it is carried linearly -- the error
  the correction exists to remove -- and it needs the joint precision, which
  those drivers computed per cell and discarded. The modal cell's copy is now
  assembled from the scratch the inner solves already return
  (gcol33/tulpa#862).

Reporting added in the same cycle: a nested fit carries the share of each
fixed-effect marginal that the integrated hyperparameter contributed, the
collapsed-grid regime carries its reading beside its code, and a flagged inner
layer names the correction it did not run. These are new columns on
`diagnostics()`, not changes to any estimate.

## R CMD check results

0 errors | 0 warnings | 0 notes locally.

The expected NOTE on the incoming check is "Days since last update: 3",
explained above.

## Test environments

* local: Windows 11, R 4.6.1, `R CMD check --as-cran` including the PDF manual
  (Status: OK)
* win-builder: R-devel and R-release (4.6.1)
* GitHub Actions, on every push: ubuntu-latest (R-release and R-devel),
  macos-latest (R-release), windows-latest (R-release and R-devel)
* Ubuntu 24.04 (WSL), gcc 13, `-fsanitize=undefined`

## Notes

* The package contains a large compiled codebase (C++ inference kernels);
  installed size may exceed the default threshold on some platforms.

* Long-running model fits in examples are wrapped in \donttest{}; each
  retains a small runnable form where feasible. Recovery and sampler tests
  are gated by testthat's skip_on_cran() and by the package's own tier
  variables (NOT_CRAN, TULPA_SLOW_TESTS).

* Nine examples are in \dontrun{}. Seven reference symbols the user supplies
  (a fit produced by a consumer model package, a per-model E-step / M-step
  callback pair, a compiled latent block, a mesh-backed SPDE spec, an
  outer-grid inner fitter, a joint fit's own sparse blocks) and so have
  nothing to execute. One (spatiotemporal_effects) needs a
  Knorr-Held interaction block, which no backend in this package fits, so the
  fit can only come from a companion model package. One (tulpa_cache_clear)
  would delete the caller's own cached builds.

* tgmrf_cpp() compiles a user-supplied C++ latent block and caches the result
  under tools::R_user_dir("tulpa", "cache"). Nothing is written unless a
  compile happens, the contents are user-manageable through
  tulpa_cache_clear(older_than = ), and no example, vignette or test writes
  there.

* The vignettes set `eval` from NOT_CRAN in their setup chunk, so the code is
  shown but not run on the check farm.

* Intra-chain OpenMP teams are capped at two threads under
  `_R_CHECK_LIMIT_CORES_` (`tulpa_omp_team_size()`, src/omp_threads.h).

* Internal helpers that seed the RNG run inside `.with_preserved_seed()`,
  which restores the caller's `.Random.seed` on exit.

## Downstream dependencies

No reverse dependencies on CRAN.
