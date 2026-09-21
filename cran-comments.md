# CRAN comments

## Resubmission

This replaces the tulpa 0.5.0 upload of 2026-09-20, which the incoming check
archived on "Overall checktime 12 min > 10 min". Uwe Ligges has since triggered
further checks on it; I have asked for that upload to be dropped in favour of
this one.

* It fixes a bug found after that upload, listed below with the others: a
  copy-scale amplitude was reported, and drawn, below zero.
* The check time is reduced. The tests CRAN runs now leave out model fits and
  sampler runs, which belong to the package's own recovery tier and run in CI,
  and read the structural identity checks on smaller grids. win-builder
  r-devel reports a check time of 491s (Status: OK), against 724s for the
  archived upload.

## Update

This is an update of tulpa 0.2.0, published on 2026-09-09.

It fixes the gcc-UBSAN issue shown on the 0.2.0 check page: an Eigen LLT object
was copied before its first factorization, so `LLT.h:66` loaded an
uninitialized `ComputationInfo` ("load of value 32119, which is not a valid
value for type 'ComputationInfo'"). Per-thread workspaces are now constructed
in place. Checked under gcc with `-fsanitize=undefined
-ftrivial-auto-var-init=pattern`, which reproduces the report on the code
before the fix and reports nothing after it.

The update also comes early because 0.2.0 carries defects that return wrong
results without an error, all fixed here:

* Matern 5/2 spatial fields were fitted with the squared-exponential kernel in
  every sampler mode (a covariance code read under two numberings).
* Resuming a checkpointed MCMC fit on a different data set of the same
  dimensions returned the earlier fit's draws, because the checkpoint
  fingerprint covered only the dimensions.
* An `offset()` term was dropped on the nested-Laplace route.
* A copy amplitude's posterior was reported, and drawn, below zero. The copy
  scale carries a declared point mass at zero beside a continuum on the
  positive half-line; the reporting geometry gave that level an ordinary cell
  and mirrored its edge half a node step below it, so the reported 2.5% bound
  and a fifth of the draws left the parameter's support.

The version is 0.5.0 rather than 0.2.1 because development continued after
0.2.0 was submitted; NEWS.md carries an entry for each version in between.

Default hyperpriors on the outer integration grid changed to proper priors, so
the same call can return different numbers from 0.2.0; NEWS.md documents each
change. The Title now expands the package name.

## R CMD check results

0 errors | 0 warnings | 0 notes locally.

The expected NOTE on the incoming check is "Days since last update: 12",
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
