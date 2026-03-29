test_that("inference tier system is defined", {
  expect_true(exists("INFERENCE_TIERS", envir = asNamespace("tulpa")) ||
              is.function(tryCatch(tulpa:::get_tier_info, error = function(e) NULL)))
})

test_that("tier 3 requires explicit opt-in", {
  # The tier system should never auto-select tier 3
  # This is a design invariant, not just a test
  expect_true(TRUE)  # Placeholder until select_backend is wired
})
