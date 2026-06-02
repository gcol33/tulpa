# Speed-tier control for the test suite.
#
# The recovery / coverage / sampler tests fit many models (and the HMC/NUTS
# tests run the sampler), which dominates the suite's wall-clock. For fast dev
# iteration set the TULPA_FAST environment variable to "1" -- the heavy
# statistical loops then skip (reported as skips, never silently dropped) and
# only the structural / closed-form / gradient unit tests run.
#
#   Sys.setenv(TULPA_FAST = "1"); devtools::test()   # fast: structure only
#   Sys.unsetenv("TULPA_FAST");   devtools::test()   # full: + recovery loops
#
# The default (variable unset) runs everything, so CRAN and CI always see the
# full recovery suite. Place skip_if_fast() as the first line of any test_that()
# block whose cost is a model fit or an HMC sample.
skip_if_fast <- function() {
  if (identical(Sys.getenv("TULPA_FAST"), "1")) {
    testthat::skip("TULPA_FAST set: skipping slow recovery/coverage loop")
  }
}
