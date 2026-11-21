# CRAN comments

## Update

This is an update of tulpa 0.7.5, published on 2026-10-19. The main items:

* `control$device` selects `"auto"`, `"cpu"` or `"cuda"`. The CUDA routes are in
  a separate package, tulpaGPU, suggested and available from
  https://gcol33.r-universe.dev (Additional_repositories). tulpa loads it only
  when a fit asks for `"auto"` or `"cuda"` and it is installed; without it every
  fit runs on the CPU as before, which is what the CRAN machines exercise.

* An outer hyperparameter axis is closed inside the interval its block
  defines it on (a BYM2 mixing weight on (0, 1), a proper-CAR `rho` on its
  eigenvalue interval, an AR1 `rho` on (-1, 1)). A BYM2 grid starting near
  zero could otherwise run its lowest level below zero and stop with an error
  (gcol33/tulpa#1188).

* A precision axis under the penalized-complexity prior has no posterior
  mean; it now reports `Inf` with the reason instead of a sum over the cells
  the grid happened to hold (gcol33/tulpa#1189).

* The elliptical-slice sampler draws the intercept level exactly also when a
  random-intercept term enters several linear predictors of a model
  package's likelihood (gcol33/tulpa#1190).

* `tulpa_re_cov_nested()` screens its tensor grid with a bounded cheap pass,
  as the other nested-Laplace entry points do (gcol33/tulpa#1192).

The full list is in NEWS.md under 0.7.6 to 0.8.0.

## R CMD check results

RESULTS

## Test environments

* win-builder: R-devel and R-release
* GitHub Actions, on every push: ubuntu-latest (R-release and R-devel),
  macos-latest (R-release), windows-latest (R-release and R-devel)

## Notes

* The package contains a large compiled codebase (C++ inference kernels);
  installed size may exceed the default threshold on some platforms.

* Long-running model fits in examples are wrapped in \donttest{}; each
  retains a small runnable form where feasible. Recovery and sampler tests
  are gated by testthat's skip_on_cran() and by the package's own tier
  variables (NOT_CRAN, TULPA_SLOW_TESTS). To keep the check time down, CRAN
  runs the structural tests of one file per subsystem; every file runs on
  GitHub Actions on four platforms.

* tgmrf_cpp() compiles a user-supplied C++ latent block and caches the result
  under tools::R_user_dir("tulpa", "cache"). Nothing is written unless a
  compile happens, the contents are user-manageable through
  tulpa_cache_clear(older_than = ), and no example, vignette or test writes
  there.

* The vignettes read the fits they show from vignettes/fits/, built by
  data-raw/build-article-fits.R, so no model is fitted on the check farm.

* Intra-chain OpenMP teams are capped at two threads under
  `_R_CHECK_LIMIT_CORES_` (`tulpa_omp_team_size()`, src/omp_threads.h).

* Internal helpers that seed the RNG restore the caller's `.Random.seed` on
  exit.

## Downstream dependencies

No reverse dependencies on CRAN.
