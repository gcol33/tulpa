# Test tiers -- single source of truth for which cost level runs where.
#
#   Tier 1  structural   (no gate)           shape / class / dispatch / formula /
#                                             closed-form / finite-difference-gradient
#                                             checks; sub-second; runs everywhere
#                                             including CRAN.
#   Tier 2  recovery      skip_on_cran()      one (or a few) fits to convergence,
#                                             parameter recovery vs a single simulated
#                                             truth, equivalence (dense == sparse,
#                                             serial == parallel, cpp == R). Seconds.
#                                             Runs by default locally + CI; skipped on
#                                             CRAN.
#   Tier 3  full          skip_if_not_slow()  MCMC / HMC sampler chains, multi-seed
#                                             coverage aggregates, multi-recovery loops.
#                                             Minutes. Opt-in; skipped by default
#                                             everywhere (including CRAN).
#
# Fast smoke mode: set TULPA_FAST=1 to collapse the suite to tier 1 only. Every
# tier-2 fit and tier-3 sampler then skips (reported as skips, never silently
# dropped), so a whole-suite plumbing / dispatch / closed-form run finishes in
# seconds. The default (TULPA_FAST unset) is unchanged, so the local dev loop,
# CI, and CRAN see exactly what they did before.
#
# Run profiles:
#   devtools::test()                                   # tiers 1 + 2  (default dev loop)
#   Sys.setenv(TULPA_FAST = "1"); test()               # tier 1 only  (fast smoke)
#   Sys.setenv(TULPA_SLOW_TESTS = "true"); test()      # tiers 1 + 2 + 3  (full validation)
#   R CMD check / CRAN                                 # tier 1 only
#
# Put exactly one tier gate as the first line of any test_that() block whose cost
# is a model fit or an MCMC sample; leave structural blocks ungated so CRAN still
# exercises them. Fast mode rides the tier-2 / tier-3 gates, so a new fit placed
# behind skip_on_cran() / skip_if_not_slow() is collapsed by TULPA_FAST=1 with no
# extra wiring.

# Fast smoke gate. Set TULPA_FAST=1 to keep only the ungated tier-1 structural
# tests; everything that fits a model or samples a chain skips.
skip_if_fast <- function() {
  if (identical(Sys.getenv("TULPA_FAST"), "1")) {
    testthat::skip("TULPA_FAST set: tier-1 structural smoke only")
  }
}

# Tier 3 full-validation gate: samplers, multi-seed coverage, multi-recovery
# loops. Fast mode takes precedence so TULPA_FAST=1 wins even if TULPA_SLOW_TESTS
# is also set.
skip_if_not_slow <- function() {
  skip_if_fast()
  if (!identical(tolower(Sys.getenv("TULPA_SLOW_TESTS")), "true")) {
    testthat::skip(
      "full-validation tier: set TULPA_SLOW_TESTS=true to run samplers / coverage loops"
    )
  }
}

# Tier 2 recovery gate: single-fit recovery / equivalence checks. Wraps the
# testthat builtin so all three tier gates have one definition here, and folds in
# fast smoke mode -- TULPA_FAST=1 skips tier 2 alongside CRAN. testthat::skip_on_cran()
# is the builtin (gated on the NOT_CRAN env var, which devtools sets locally and
# CRAN leaves unset).
skip_on_cran <- function() {
  skip_if_fast()
  testthat::skip_on_cran()
}

# Publish the gates into the global environment. This is what makes the
# skip_on_cran() above take precedence over the identically named testthat
# builtin when a test file calls it bare: globalenv is searched ahead of the
# attached testthat package, whereas a helper-scoped binding is not. Mirrors the
# namespace-aliasing in helper-internal.R.
for (.nm in c("skip_if_fast", "skip_if_not_slow", "skip_on_cran")) {
  assign(.nm, get(.nm), envir = globalenv())
}
rm(.nm)
