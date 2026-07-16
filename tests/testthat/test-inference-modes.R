test_that("inference tier system is defined", {
  expect_true(exists("INFERENCE_TIERS", envir = asNamespace("tulpa")) ||
              is.function(tryCatch(tulpa:::get_tier_info, error = function(e) NULL)))
})

test_that("auto mode never selects a Tier 3 (VI / approximate) backend", {
  # Design invariant: auto may choose only Tier 1 (exact) or Tier 2 (structured),
  # never the approximate-MCMC / VI backends (registry tier "optimized"), which
  # carry no correctness guarantee and are explicit opt-in only.
  reg  <- tulpa:::BACKEND_REGISTRY
  auto <- tulpa:::get_mode_backends("auto")
  optimized <- names(Filter(function(e) identical(e$tier, "optimized"), reg))

  expect_true("vi" %in% optimized)               # VI is in the opt-in tier
  expect_false("vi" %in% auto)                    # and auto never picks it
  expect_length(intersect(auto, optimized), 0L)   # nor any other approximate one
})
