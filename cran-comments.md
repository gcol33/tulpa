# CRAN comments

## Update

This is an update of tulpa 0.6.0, published on 2026-09-24. It follows closely
because 0.6.0 returns wrong numbers without an error on several paths. The main
items:

* `mode = "ess"` did not mix on hierarchical models: each sweep left the
  intercept, the covariates and the random-effect scales where they started
  (gcol33/tulpa#877). It now draws them every sweep.

* `mode = "smc"` bridged to the wrong target, and `mode = "ess"` did not update
  the log-SD and correlation parameters of a random-slope block. Both now
  target the stated posterior.

* Fixed-effect standard errors under an intrinsic field (ICAR, BYM2, RW1 / RW2)
  and the per-component scaling of `spatial_bym2()` on a disconnected graph are
  corrected; `weights =` no longer changes which backend `mode = "auto"`
  selects.

* An explicit `control$integration = "grid_adaptive"` ran a different
  integration design at four or more hyperparameter axes and reported it
  without saying so (gcol33/tulpa#914).

* `family = Gamma()` fits the inverse link it names, and `VarCorr()` /
  `ranef()` report the quantities a fit actually used or sampled.

Messages: a one-level categorical predictor is named in the error rather than
reaching `model.matrix()`'s bare contrasts error, and a refused outer grid
names the axes that produced its cell count (gcol33/tulpa#913).

## R CMD check results

0 errors | 0 warnings | 0 notes locally.

The expected NOTE on the incoming check is "Days since last update", explained
above.
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
