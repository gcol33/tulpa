test_that("inference tier system is defined and every backend carries a valid tier", {
  reg <- tulpa:::BACKEND_REGISTRY
  expect_true(is.list(reg) && length(reg) > 0L)
  tiers <- vapply(reg, function(e)
    if (is.null(e$tier)) NA_character_ else e$tier, character(1))
  # Every registered backend carries one of the three correctness tiers.
  expect_false(anyNA(tiers))
  expect_true(all(tiers %in% c("exact", "structured", "optimized")))
  # Tier 1 (exact) and Tier 2 (structured) are both populated.
  expect_true(any(tiers == "exact"))
  expect_true(any(tiers == "structured"))
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


test_that("the unknown-mode message names every mode the registry defines", {
  # The message and the sibling that validates the same names both have to come
  # off INFERENCE_TIERS; a restated list goes stale the moment a tier is added.
  msg <- tryCatch(tulpa:::get_mode_backends("no-such-mode"),
                  error = conditionMessage)
  for (m in c("auto", names(tulpa:::INFERENCE_TIERS))) {
    expect_true(grepl(m, msg, fixed = TRUE), info = m)
  }
})
